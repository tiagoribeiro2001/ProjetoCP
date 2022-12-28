#include "../include/utils.h"

// ---------------------------------------     Variáveis Globais     -------------------------------------------

#define NUMBER_CLUSTERS 4
#define NUMBER_POINTS 10000000

// Set up GPU data

float* gpu_points;
float* gpu_centroid;
int* gpu_cluster_attribution;

// ---------------------------------------     Funções     -------------------------------------------


// Funcao que inicializa a lista de pontos e os clusters de forma aleatoria
__host__ void inicializa(float *points, float *centroid){

    srand(10);

    // Preenche a lista de pontos
    for(int i = 0; i < NUMBER_POINTS; i++) {
        points[i * 2] = (float) rand() / RAND_MAX;
        points[i * 2 + 1] = (float) rand() / RAND_MAX;
    }

    // Os primeiros centroides sao as primeiras amostras
    for(int i = 0; i < NUMBER_CLUSTERS; i++) {
        centroid[i * 2] = points[i * 2];
        centroid[i * 2 + 1] = points[i * 2 + 1];
    }
}

// Funcao que atribui todas as amostras ao seu devido cluster
__global__ void atribuiCluster(float *points, float *centroid, int *cluster_attribution){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= NUMBER_POINTS) return;
    
    // Fórmula da distancia euclidiana
    float min = ((centroid[0] - points[index * 2]) * (centroid[0] - points[index * 2])) + ((centroid[1] - points[index * 2 + 1]) * (centroid[1] - points[index * 2 + 1]));
    int c = 0;
    // Percorre os vários clusters para comparar a distancia da amostra ao seu centroid
    for(int j = 1; j < NUMBER_CLUSTERS; j++){
        float dist = ((centroid[j * 2] - points[index * 2]) * (centroid[j * 2] - points[index * 2])) + ((centroid[j * 2 + 1] - points[index * 2 + 1]) * (centroid[j * 2 + 1] - points[index * 2 + 1]));
        if (dist < min){
            c = j;
            min = dist;
        }
    }

    // Atualiza o cluster do ponto
    cluster_attribution[index] = c;

}

// Funcao que calcula o centroid de cada cluster
__host__ void calculaCentroid(float *points, float *centroid, float *sum, int *size, int *cluster_attribution){

    // Reseta os somatorios
    for(int i = 0; i < NUMBER_CLUSTERS; i++){
        sum[i * 2] = 0;
        sum[i * 2 + 1] = 0;    
        size[i] = 0;
    }

    // Faz os somatorios
    for (int i = 0; i < NUMBER_POINTS; i++) {
        int clust = cluster_attribution[i];
        sum[clust * 2] +=  points[i * 2];
        sum[clust * 2 + 1] += points[i * 2 + 1];
        size[clust]++;
    }

    // O centroid e calculado a partir da media de todos os pontos do cluster
    for(int i = 0; i < NUMBER_CLUSTERS; i++){
        float mediaX = sum[i * 2] / size[i];
        float mediaY = sum[i * 2 + 1] / size[i];
        centroid[i * 2] = mediaX;
        centroid[i * 2 + 1] = mediaY;
    }
}

__host__ int verificaConverge(int *cluster_attribution, int *prev_cluster_attribution){
    int conv = 1;

    for(int i = 0; i < NUMBER_POINTS; i++){
        if (cluster_attribution[i] != prev_cluster_attribution[i]){
        conv = 0;
        }
    }

    return conv;
}

// Funcao que imprime informacao relativa ao estado final do programa
__host__ void printInfo(int iteration, float *centroid, int *size){
    printf("N = %d, K = %d\n", NUMBER_POINTS, NUMBER_CLUSTERS);
    for (int i = 0; i < NUMBER_CLUSTERS; i++){
        printf("Center: (%.3f, %.3f) : Size %d\n", centroid[i * 2], centroid[i * 2 + 1], size[i]);
    }
    printf("Iterations: %d\n", iteration - 1);
}


// -----------------------------------------     Main     ------------------------------------------

int main(int argc, char*argv[]){

    clock_t start, end;
    double elapsed;

    start = clock();

    // Inicializacao de variaveis
    int iteration = 0;
    int convergiu = 0;
    float *points;
    float *centroid;
    float *sum;
    int *size;
    int *cluster_attribution;
    int *prev_cluster_attribution;

    // Aloca memoria
    points = (float *) malloc(sizeof(float) * NUMBER_POINTS * 2);
    centroid = (float *) malloc(sizeof(float) * NUMBER_CLUSTERS * 2);
    sum = (float *) malloc(sizeof(float) * NUMBER_CLUSTERS * 2);
    size = (int *) malloc(sizeof(int) * NUMBER_CLUSTERS);
    cluster_attribution = (int *) malloc(sizeof(int) * NUMBER_POINTS);
    prev_cluster_attribution = (int *) malloc(sizeof(int) * NUMBER_POINTS);

    // Insere valores nos arrays dos pontos e centroides
    inicializa(points, centroid);
    
    // Mallocar as variáveis
    cudaMalloc(&gpu_points, NUMBER_POINTS * 2 * sizeof(float));
    cudaMalloc(&gpu_centroid, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_cluster_attribution, NUMBER_POINTS * sizeof(int));
    
    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_points, points, NUMBER_POINTS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroid, centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cluster_attribution, cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyHostToDevice);

    // Determina o numero maximo de threads por bloco da maquina
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int max_threads_per_block = prop.maxThreadsPerBlock;

    int numberThreads = min(max_threads_per_block, NUMBER_POINTS);
    int numberBlocks = (NUMBER_POINTS / numberThreads) + 1;


    // Algoritmo de Lloyd
    while (iteration < 51 && convergiu != 1) {

        atribuiCluster<<<numberBlocks, numberThreads>>>(gpu_points, gpu_centroid, gpu_cluster_attribution);

        // Copia os valores do cluster_attribution para o prev_cluster_attribution
        memcpy(prev_cluster_attribution, cluster_attribution, NUMBER_POINTS * sizeof(int));

        // Atualiza os valores na memoria com as novas atribuicoes dos clusters
        cudaMemcpy(cluster_attribution, gpu_cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyDeviceToHost);
        
        calculaCentroid(points, centroid, sum, size, cluster_attribution);

        // Envia os novos centroides para a GPU
        cudaMemcpy(gpu_centroid, centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);

        convergiu = verificaConverge(cluster_attribution, prev_cluster_attribution);

        iteration++;
    }

    // Libertar a memoria da GPU
    cudaFree(gpu_points);
    cudaFree(gpu_centroid);
    cudaFree(gpu_cluster_attribution);

    printInfo(iteration, centroid, size);

    // Libertar a memoria
    free(points);
    free(centroid);
    free(sum);
    free(size);
    free(cluster_attribution);
    free(prev_cluster_attribution);

    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %f\n", elapsed);
    return 0;
}