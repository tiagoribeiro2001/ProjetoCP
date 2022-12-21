#include "../include/utils.h"

// ---------------------------------------     Variáveis Globais     -------------------------------------------

#define NUMBER_CLUSTERS 4
#define NUMBER_POINTS 1000000

// Set up GPU data

int* gpu_convergiu;
float* gpu_points;
float* gpu_centroid;
float* gpu_sum;
int* gpu_size;
int* gpu_cluster_attribution;
int* gpu_prev_cluster_attribution;

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
__global__ void atribuiCluster(float *points, float *centroid, float *sum, int *size, int *cluster_attribution, int *prev_cluster_attribution){


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= NUMBER_POINTS) return;

    // Percorre a lista de amostras e atribui-as a um cluster
    
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
__global__ void calculaCentroid(float *points, float *centroid, float *sum, int *size, int *cluster_attribution){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= NUMBER_CLUSTERS) return;

    sum[index * 2] = 0;
    sum[index * 2 + 1] = 0;    
    size[index] = 0;

    for (int i = 0; i < NUMBER_POINTS; i++) {
        int clust = cluster_attribution[i];
        if (clust == index){
            sum[clust * 2] +=  points[i * 2];
            sum[clust * 2 + 1] += points[i * 2 + 1];
            size[clust]++;
        }
    }

    // O centroid e calculado a partir da media de todos os pontos do cluster
    float mediaX = sum[index * 2] / size[index];
    float mediaY = sum[index * 2 + 1] / size[index];
    centroid[index * 2] = mediaX;
    centroid[index * 2 + 1] = mediaY;
}

__global__ void verificaConverge(int *convergiu, int *cluster_attribution, int *prev_cluster_attribution){
    int conv = 1;

    for(int i = 0; i < NUMBER_POINTS; i++){
        if (cluster_attribution[i] != prev_cluster_attribution[i]){
        conv = 0;
        }
    }

    *convergiu = conv;
}

// Funcao que imprime informacao relativa ao estado final do programa
__host__ void printInfo(int iteration, float *centroid, int *size){
    printf("N = %d, K = %d\n", NUMBER_POINTS, NUMBER_CLUSTERS);
    for (int i = 0; i < NUMBER_CLUSTERS; i++){
        printf("Center: (%.3f, %.3f) : Size %d\n", centroid[i * 2], centroid[i * 2 + 1], size[i]);
    }
    printf("Iterations: %d\n", iteration);
}


// -----------------------------------------     Main     ------------------------------------------

int main(int argc, char*argv[]){

    clock_t start, end;
    double elapsed;

    start = clock();

    int iteration = 0;

    int convergiu = 0;
    float *points;
    float *centroid;
    float *sum;
    int *size;
    int *cluster_attribution;
    int *prev_cluster_attribution;

    points = (float *) malloc(sizeof(float) * NUMBER_POINTS * 2);
    centroid = (float *) malloc(sizeof(float) * NUMBER_CLUSTERS * 2);
    sum = (float *) malloc(sizeof(float) * NUMBER_CLUSTERS * 2);
    size = (int *) malloc(sizeof(int) * NUMBER_CLUSTERS);
    cluster_attribution = (int *) malloc(sizeof(int) * NUMBER_POINTS);
    prev_cluster_attribution = (int *) malloc(sizeof(int) * NUMBER_POINTS);

    inicializa(points, centroid);
    
    // Mallocar as variáveis
    cudaMalloc(&gpu_convergiu, sizeof(int));
    cudaMalloc(&gpu_points, NUMBER_POINTS * 2 * sizeof(float));
    cudaMalloc(&gpu_centroid, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_sum, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_size, NUMBER_CLUSTERS * sizeof(int));
    cudaMalloc(&gpu_cluster_attribution, NUMBER_POINTS * sizeof(int));
    cudaMalloc(&gpu_prev_cluster_attribution, NUMBER_POINTS * sizeof(int));
    
    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_convergiu, &convergiu, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_points, points, NUMBER_POINTS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroid, centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sum, sum, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size, size, NUMBER_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cluster_attribution, cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_prev_cluster_attribution, prev_cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyHostToDevice);

    int numberThreads = min(1024, NUMBER_POINTS);
    int numberBlocks = (NUMBER_POINTS / numberThreads) + 1;


    // Algoritmo de Lloyd
    while (iteration < 20 || !convergiu) {

        cudaMemcpy(gpu_prev_cluster_attribution, gpu_cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyDeviceToDevice);

        atribuiCluster<<<numberBlocks, numberThreads>>>(gpu_points, gpu_centroid, gpu_sum, gpu_size, gpu_cluster_attribution, gpu_prev_cluster_attribution);

        calculaCentroid<<<NUMBER_CLUSTERS, 1>>>(gpu_points, gpu_centroid, gpu_sum, gpu_size, gpu_cluster_attribution);

        verificaConverge<<<1, 1>>>(gpu_convergiu, gpu_cluster_attribution, gpu_prev_cluster_attribution);

        cudaMemcpy(&convergiu, gpu_convergiu, sizeof(int), cudaMemcpyDeviceToHost);

        iteration++;
    }

    // Copy results back to host
    cudaMemcpy(centroid, gpu_centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size, gpu_size, NUMBER_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(gpu_convergiu);
    cudaFree(gpu_points);
    cudaFree(gpu_centroid);
    cudaFree(gpu_sum);
    cudaFree(gpu_size);
    cudaFree(gpu_cluster_attribution);
    cudaFree(gpu_prev_cluster_attribution);

    printInfo(iteration, centroid, size);

    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %f\n", elapsed);
    return 0;
}