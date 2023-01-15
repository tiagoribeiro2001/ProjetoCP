#include "../include/utils.h"

// ---------------------------------------     Variáveis Globais     -------------------------------------------

// Dados da GPU
float* gpu_points;
float* gpu_centroid;
int* gpu_cluster_attribution;

// ---------------------------------------     Funções     -------------------------------------------


// Funcao que inicializa a lista de pontos e os clusters de forma aleatoria
__host__ void inicializa(point *points, point *centroids, sumThread *sumThreads, int *sizes, int number_points, int number_clusters, int number_blocks, int number_threadspblock){

    srand(10);
    
    // Preenche a lista de pontos
    for(int i = 0; i < number_points; i++) {
        points[i].x = (float) rand() / RAND_MAX;
        points[i].y = (float) rand() / RAND_MAX;
    }

    // Os primeiros centroides sao as primeiras amostras
    for(int i = 0; i < number_clusters; i++) {
        centroids[i].x = points[i].x;
        centroids[i].y = points[i].y;
    }

    for(int i = 0; i < number_blocks * number_threadspblock; i++){
        sumThreads[i].sum_x = 0;
        sumThreads[i].sum_y = 0;
        sumThreads[i].size = 0;
    }

    for(int i = 0; i < number_clusters; i++){
        sizes[i] = 0;
    }
}

__global__ void atribuiCluster(point *points, point *centroid, sumThread *sumThreads, int number_points, int number_clusters, int number_blocks, int number_threadspblock){

    // Numero total de threads
    int total_threads = number_blocks * number_threadspblock;

    // Id da thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Percorre todos os pontos
    for (int i = index; i < number_points; i += total_threads){
        // Determina a distancia para o primeiro cluster
        float min = ((centroid[0].x - points[i].x) * (centroid[0].x - points[i].x)) + ((centroid[0].y - points[i].y) * (centroid[0].y - points[i].y));
        
        int c = 0;
        // Percorre os vários clusters para comparar a distancia da amostra ao seu centroid
        for(int j = 1; j < number_clusters; j++){
            float dist = ((centroid[j].x - points[i].x) * (centroid[j].x - points[i].x)) + ((centroid[j].y - points[i].y) * (centroid[j].y - points[i].y));
            if (dist < min){
                c = j;
                min = dist;
            }
        }

        // Soma ao cluster mais proximo, no espaco designado da thread
        sumThreads[index * number_clusters + c].sum_x += points[i].x;
        sumThreads[index * number_clusters + c].sum_y += points[i].y;
        sumThreads[index * number_clusters + c].size++;
    }
}

__global__ void calculaCentroids(point *centroid, sumThread* sumThreads, int * sizes, int number_clusters, int number_blocks, int number_threadspblock){

    // Id da thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Reinicializa as variaveis
    centroid[index].x = 0;
    centroid[index].y = 0;
    sizes[index] = 0;

    // Percorre o array clusters e efetua o somatorio dos somatorios efetuados por cada thread no kernel anterior
    for (int i = index; i < number_blocks * number_threadspblock * number_clusters; i += number_clusters){
        // Efetua o somatorio das coordenadas
        centroid[index].x += sumThreads[i].sum_x;
        centroid[index].y += sumThreads[i].sum_y;
        sizes[index] += sumThreads[i].size;

        sumThreads[i].sum_x = 0;
        sumThreads[i].sum_y = 0;
        sumThreads[i].size = 0;
    }

    centroid[index].x = centroid[index].x / sizes[index];
    centroid[index].y = centroid[index].y / sizes[index];

}

// Funcao que imprime informacao relativa ao estado final do programa
__host__ void printInfo(int iteration, point *centroid, int *size, int number_points, int number_clusters){
    printf("N = %d, K = %d\n", number_points, number_clusters);
    for (int i = 0; i < number_clusters; i++){
        printf("Center: (%.3f, %.3f) : Size %d\n", centroid[i].x, centroid[i].y, size[i]);
    }
    printf("Iterations: %d\n", iteration - 1);
}


// -----------------------------------------     Main     ------------------------------------------

int main(int argc, char*argv[]){

    // Verifica se o número de argumentos passados são os corretos
    if (argc != 5) {
        printf("Error -> Wrong number of arguments. \n");
        return -1;
    }

    clock_t start, end;
    double elapsed;

    start = clock();

    int number_points = atoi(argv[1]);
    int number_clusters = atoi(argv[2]);
    int number_blocks = atoi(argv[3]);
    int number_threadspblock = atoi(argv[4]);

    // Inicializacao de variaveis
    int iteration = 0;
    point *points;
    point *centroids;
    sumThread *sumThreads;
    int *sizes;

    // Aloca memoria
    points = (point *) malloc(sizeof(struct Point) * number_points);
    centroids = (point *) malloc(sizeof(struct Point) * number_clusters);
    sumThreads = (sumThread *) malloc(sizeof(struct SumThread) * number_blocks * number_threadspblock * number_clusters);
    sizes = (int *) malloc(sizeof(int) * number_clusters);
    
    // Inicializacao de variaveis da GPU
    point* gpu_points;
    point* gpu_centroids;
    sumThread* gpu_sumThreads;
    int* gpu_sizes;

    
    // Mallocar as variáveis na GPU
    cudaMalloc(&gpu_points, number_points * sizeof(struct Point));
    cudaMalloc(&gpu_centroids, number_clusters * sizeof(struct Point));
    cudaMalloc(&gpu_sumThreads, number_blocks * number_threadspblock * number_clusters * sizeof(struct SumThread));
    cudaMalloc(&gpu_sizes, number_clusters * sizeof(int));

    // Insere valores nos arrays dos pontos e centroides
    inicializa(points, centroids, sumThreads, sizes, number_points, number_clusters, number_blocks, number_threadspblock);

    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_points, points, number_points * sizeof(struct Point), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroids, centroids, number_clusters * sizeof(struct Point), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sumThreads, sumThreads, number_blocks * number_threadspblock * number_clusters * sizeof(struct SumThread), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sizes, sizes, number_clusters * sizeof(int), cudaMemcpyHostToDevice);

    while(iteration < 21) {
        atribuiCluster<<<number_blocks, number_threadspblock>>>(gpu_points, gpu_centroids, gpu_sumThreads, number_points, number_clusters, number_blocks, number_threadspblock);
        calculaCentroids<<<1, number_clusters>>>(gpu_centroids, gpu_sumThreads, gpu_sizes, number_clusters, number_blocks, number_threadspblock);
        iteration++;
    }

    cudaMemcpy(sizes, gpu_sizes, number_clusters * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, gpu_centroids, number_clusters * sizeof(struct Point), cudaMemcpyDeviceToHost);

    printInfo(iteration, centroids, sizes, number_points, number_clusters);

    // Libertar a memoria da GPU
    cudaFree(gpu_points);
    cudaFree(gpu_centroids);
    cudaFree(gpu_sumThreads);
    cudaFree(gpu_sizes);

    // Libertar a memoria
    free(points);
    free(centroids);
    free(sumThreads);
    free(sizes);

    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %f\n", elapsed);

    return 0;
}