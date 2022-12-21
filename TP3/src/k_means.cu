#include "../include/utils.h"

// ---------------------------------------     Variáveis Globais     -------------------------------------------

#define NUMBER_CLUSTERS 4
#define NUMBER_POINTS 1000000
#define NUMBER_BLOCKS 32

// Set up GPU data

int* gpu_iteration;
int* gpu_convergiu;
float* gpu_points;
float* gpu_centroid;
float* gpu_sum;
int* gpu_size;
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
__global__ void atribuiCluster(int *iteration, int *convergiu, float *points, float *centroid, float *sum, int *size, int *cluster_attribution){

    // int conv = 1;

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

    /*
    if (iteration == 0 || cluster_attribution[index] != c){
        conv = 0;
    }
    */
    // Atualiza o cluster do ponto
    cluster_attribution[index] = c;

    // *convergiu = conv;
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

    int iteration = 0;
    int convergiu = 0;
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

    inicializa(points, centroid);
    
    // Mallocar as variáveis
    cudaMalloc(&gpu_iteration, sizeof(int));
    cudaMalloc(&gpu_convergiu, sizeof(int));
    cudaMalloc(&gpu_points, NUMBER_POINTS * 2 * sizeof(float));
    cudaMalloc(&gpu_centroid, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_sum, NUMBER_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&gpu_size, NUMBER_CLUSTERS * sizeof(int));
    cudaMalloc(&gpu_cluster_attribution, NUMBER_POINTS * sizeof(int));
    
    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_iteration, &iteration, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_convergiu, &convergiu, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_points, points, NUMBER_POINTS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroid, centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sum, sum, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size, size, NUMBER_CLUSTERS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cluster_attribution, cluster_attribution, NUMBER_POINTS * sizeof(int), cudaMemcpyHostToDevice);

    int numberThreads = (NUMBER_POINTS + NUMBER_BLOCKS - 1) / NUMBER_BLOCKS;
    if (numberThreads > 1024){
        numberThreads = 1024;
    } 

    // Algoritmo de Lloyd
    while (iteration < 20) {
        atribuiCluster<<<NUMBER_BLOCKS, numberThreads>>>(gpu_iteration, gpu_convergiu, gpu_points, gpu_centroid, gpu_sum, gpu_size, gpu_cluster_attribution);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("1 -> Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaDeviceSynchronize();

        cudaMemcpy(&convergiu, gpu_convergiu, sizeof(int), cudaMemcpyDeviceToHost);

        calculaCentroid<<<NUMBER_CLUSTERS, 1>>>(gpu_points, gpu_centroid, gpu_sum, gpu_size, gpu_cluster_attribution);
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("1 -> Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        iteration++;
    }

    // Copy results back to host
    cudaMemcpy(&iteration, gpu_iteration, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(centroid, gpu_centroid, NUMBER_CLUSTERS * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size, gpu_size, NUMBER_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(gpu_iteration);
    cudaFree(gpu_convergiu);
    cudaFree(gpu_points);
    cudaFree(gpu_centroid);
    cudaFree(gpu_sum);
    cudaFree(gpu_size);
    cudaFree(gpu_cluster_attribution);

    printInfo(iteration, centroid, size);

    printf("Execution time: \n");
    return 0;
}