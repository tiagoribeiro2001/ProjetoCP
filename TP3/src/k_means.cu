#include "../include/utils.h"

// ---------------------------------------     Variáveis Globais     -------------------------------------------

#define NUMBER_CLUSTERS 4
#define NUMBER_POINTS 10000000

// Contador da iteracoes
int iteration = 0;

// Set up GPU data

float* gpu_points;
float* gpu_centroid;
float* gpu_sum;
int* gpu_size;
int* gpu_cluster_attribution;

// ---------------------------------------     Funções     -------------------------------------------


// Funcao que inicializa a lista de pontos e os clusters de forma aleatoria
void inicializa(float *points, float *centroid, float *sum, int *size, int *cluster_attribution){

    srand(10);

    // Preenche a lista de pontos
    for(int i = 0; i < NUMBER_POINTS * 2; i += 2) {
        points[i] = (float) rand() / RAND_MAX;
        points[i + 1] = (float) rand() / RAND_MAX;
    }

    // Os primeiros centroides sao as primeiras amostras
    for(int i = 0; i < NUMBER_CLUSTERS; i += 2) {
        centroid[i] = points[i];
        centroid[i + 1] = points[i + 1];
    }
}

// Funcao que atribui todas as amostras ao seu devido cluster
__global__ void atribuiCluster(float *points, float *centroid, float *sum, int *size, int *cluster_attribution){

    printf("ola\n");

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Reseta as variaveis de todos os clusters a 0
    for(int i = 0; i < NUMBER_CLUSTERS; i++){
        size[i] = 0;
        sum[i * 2] = 0;
        sum[i * 2 + 1] = 0;
    }

    // Percorre a lista de amostras e atribui-as a um cluster
    if (index < NUMBER_POINTS){
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

    // Percorre os pontos e atualiza os valores do tamanho e somatorios das coordenadas dos clusters
    for(int i = 0; i < NUMBER_POINTS; i++){
        int c = cluster_attribution[i];
        size[c]++;
        sum[c * 2] += points[i * 2];
        sum[c * 2 + 1] += points[i * 2 + 1];
    }
}

// Funcao que calcula o centroid de cada cluster
void calculaCentroid(float *centroid, float *sum, int *size){
    for(int i = 0; i < NUMBER_CLUSTERS; i++){
        // O centroid e calculado a partir da media de todos os pontos do cluster
        float mediaX = sum[i * 2] / size[i];
        float mediaY = sum[i * 2 + 1] / size[i];
        centroid[i * 2] = mediaX;
        centroid[i * 2 + 1] = mediaY;
    }
}

// Funcao que imprime informacao relativa ao estado final do programa
void printInfo(float *centroid, int *size){
    printf("N = %d, K = %d\n", NUMBER_POINTS, NUMBER_CLUSTERS);
    for (int i = 0; i < NUMBER_CLUSTERS; i++){
        printf("Center: (%.3f, %.3f) : Size %d\n", centroid[i * 2], centroid[i * 2 + 1], size[i]);
    }
    printf("Iterations: %d\n", iteration);
}


// -----------------------------------------     Main     ------------------------------------------

int main(int argc, char*argv[]){

    float *points;
    float *centroid;
    float *sum;
    int *size;
    int *cluster_attribution;

    points = (float *) malloc(sizeof(float) * NUMBER_POINTS * 2);
    centroid = (float *) malloc(sizeof(float) * NUMBER_CLUSTERS * 2);
    sum = (float *) malloc(sizeof(float) * NUMBER_CLUSTERS * 2);
    size = (int *) malloc(sizeof(int) * NUMBER_CLUSTERS);
    cluster_attribution = (int *) malloc(sizeof(int) * NUMBER_POINTS);

    inicializa(points, centroid, sum, size, cluster_attribution);
    
    // Mallocar as variáveis globais
    cudaMalloc(&gpu_points, NUMBER_POINTS * 2 * sizeof(float));
    cudaMalloc(&gpu_centroid, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_sum, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_size, NUMBER_CLUSTERS * sizeof(int));
    cudaMalloc(&gpu_cluster_attribution, NUMBER_POINTS * sizeof(int));
    
    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_points, points, NUMBER_POINTS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroid, centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sum, sum, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size, size, NUMBER_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cluster_attribution, cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyHostToDevice);

    // Algoritmo de Lloyd
    atribuiCluster<<<1, 4>>>(points, centroid, sum, size, cluster_attribution);
    
    while(iteration < 20) {
        calculaCentroid(centroid, sum, size);
        atribuiCluster<<<1, 4>>>(points, centroid, sum, size, cluster_attribution);
        iteration++;
    }

    // Copy results back to host
    cudaMemcpy(gpu_points, points, NUMBER_POINTS * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_centroid, centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_sum, sum, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_size, size, NUMBER_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_cluster_attribution, cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(gpu_points);
    cudaFree(gpu_centroid);
    cudaFree(gpu_sum);
    cudaFree(gpu_size);
    cudaFree(gpu_cluster_attribution);

    printInfo(centroid, size);

    printf("Execution time: \n");
    return 0;
}