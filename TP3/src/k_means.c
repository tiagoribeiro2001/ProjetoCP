#include "../include/utils.h"

// ---------------------------------------     Variáveis Globais     -------------------------------------------

// Contador da iteracoes
int iteration = 0;

// Numero de clusters
int number_clusters;

// Numero de amostras
int number_points;

// Numero de fios de execucao
int number_wires;

// Array que indica o cluster de cada ponto
int *clust;

// Array que indica a coordenada x de cada ponto
float *coordX;

// Array que indica a coordenada y de cada ponto
float *coordY;

// Array das coordenadas X dos centroides dos clusters
float *centroidX;

// Array das coordenadas Y dos centroides dos clusters
float *centroidY;

// Array dos tamanhos de cada cluster
int *size;

// Array dos somatórios da coordenada X dos pontos de cada cluster
float *sumX;

//Array dos somatórios da coordenada Y dos pontos de cada cluster
float *sumY;

// Set up GPU data
int* gpu_clust;
float* gpu_coordX;
float* gpu_coordY;
float* gpu_centroidX;
float* gpu_centroidY;
int* gpu_size;
float* gpu_sumX;
float* gpu_sumY;



// ---------------------------------------     Funções     -------------------------------------------


// Funcao que inicializa a lista de pontos e os clusters de forma aleatoria
__global__ void inicializa(int npontos, int nclusters) {

    srand(10);

    clust = (int *) malloc(sizeof(int) * npontos);
    coordX = (float *) malloc(sizeof(float) * npontos);
    coordY = (float *) malloc(sizeof(float) * npontos);

    centroidX = (float *) malloc(sizeof(float) * npontos);
    centroidY = (float *) malloc(sizeof(float) * npontos);
    size = (int *) malloc(sizeof(int) * npontos);
    sumX = (float *) malloc(sizeof(float) * npontos);
    sumY = (float *) malloc(sizeof(float) * npontos);

    // Preenche a lista de pontos
    for(int i = 0; i < number_points; i++) {
        coordX[i] = (float) rand() / RAND_MAX;
        coordY[i] = (float) rand() / RAND_MAX;
    }

    // Os primeiros centroides sao as primeiras amostras
    for(int i = 0; i < number_clusters; i++) {
        centroidX[i] = coordX[i];
        centroidY[i] = coordY[i];
    }
}

// Funcao que atribui todas as amostras ao seu devido cluster
__global__ void atribuiCluster(){

    int conv = 1;

    // Reseta as variaveis de todos os clusters a 0
    for(int i = 0; i < number_clusters; i++ ) {
        size[i] = 0;
        sumX[i] = 0;
        sumY[i] = 0;
    }

    // Percorre a lista de amostras e atribui-as a um cluster
    //#pragma omp parallel for num_threads(number_wires)
    for(int i = 0; i < number_points; i++){

        // Fórmula da distancia euclidiana
        float min = ((centroidX[0] - coordX[i]) * (centroidX[0] - coordX[i])) + ((centroidY[0] - coordY[i]) * (centroidY[0] - coordY[i]));
        int c = 0;
        // Percorre os vários clusters para comparar a distancia da amostra ao seu centroid
        for(int j = 1; j < number_clusters; j++){
            float dist = ((centroidX[j] - coordX[i]) * (centroidX[j] - coordX[i])) + ((centroidY[j] - coordY[i]) * (centroidY[j] - coordY[i]));
            if (dist < min){
                c = j;
                min = dist;
            }
        }

        // Atualiza o cluster do ponto
        clust[i] = c;
    }

    // Percorre os pontos e atualiza os valores do tamanho e somatorios das coordenadas dos clusters
    for(int i = 0; i < number_points; i++){
        int c = clust[i];
        size[c]++;
        sumX[c] += coordX[i];
        sumY[c] += coordY[i];
    }
}

// Funcao que calcula o centroid de cada cluster
__global__ void calculaCentroid(){
    for(int i = 0; i < number_clusters; i++){
        // O centroid e calculado a partir da media de todos os pontos do cluster
        float mediaX = sumX[i] / size[i];
        float mediaY = sumY[i] / size[i];
        centroidX[i] = mediaX;
        centroidY[i] = mediaY;
    }
}

// Funcao que imprime informacao relativa ao estado final do programa
void printInfo(){
    printf("N = %d, K = %d\n", number_points, number_clusters);
    for (int i = 0; i < number_clusters; i++){
        printf("Center: (%.3f, %.3f) : Size %d\n", centroidX[i], centroidY[i], size[i]);
    }
    printf("Iterations: %d\n", iteration);
}




// -----------------------------------------     Main     ------------------------------------------

int main(int argc, char*argv[]){

    // Verifica se o número de argumentos passados são os corretos
    if (argc != 4) {
        printf("Error -> Wrong number of arguments. \n");
        return -1;
    }

    // Guarda em variáveis globais os inputs dado pelo utilizador
    number_points = atoi(argv[1]);
    number_clusters = atoi(argv[2]);
    number_wires = atoi(argv[3]);

    double itime, ftime, exec_time;
    itime = omp_get_wtime();


    // Mallocar as variáveis globais
    cudaMalloc(&gpu_clust, number_points * sizeof(int));
    cudaMalloc(&gpu_coordX, number_points * sizeof(float));
    cudaMalloc(&gpu_coordY, number_points * sizeof(float));
    cudaMalloc(&gpu_centroidX, number_clusters * sizeof(float));
    cudaMalloc(&gpu_centroidY, number_clusters * sizeof(float));
    cudaMalloc(&gpu_size, number_clusters * sizeof(int));
    cudaMalloc(&gpu_sumX, number_clusters * sizeof(float));
    cudaMalloc(&gpu_sumY, number_clusters * sizeof(float));

    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_clust, clust, number_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_coordX, coordX, number_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroidX, centroidX, number_clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_centroidY, centroidY, number_clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size, size, number_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sumX, sumX, number_clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sumY, sumY, number_clusters * sizeof(float), cudaMemcpyHostToDevice);

    // Algoritmo de Lloyd
    inicializa<<<1, number_points>>>(number_points, number_clusters);
    atribuiCluster<<<1, 1>>>();
    while(iteration < 20) {
        calculaCentroid<<<1, number_clusters>>>();
        atribuiCluster<<<1, 1>>>();
        iteration++;
    }

    // Copy results back to host
    cudaMemcpy(gpu_clust, clust, number_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_coordX, coordX, number_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_centroidX, centroidX, number_clusters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_centroidY, centroidY, number_clusters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_size, size, number_clusters * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_sumX, sumX, number_clusters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_sumY, sumY, number_clusters * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(gpu_clust);
    cudaFree(gpu_coordX);
    cudaFree(gpu_coordY);
    cudaFree(gpu_centroidX);
    cudaFree(gpu_centroidY);
    cudaFree(gpu_size);
    cudaFree(gpu_sumX);
    cudaFree(gpu_sumY);

    printInfo();

    ftime = omp_get_wtime();
    exec_time = ftime - itime;

    printf("Execution time: %f\n", exec_time);
    return 0;
}