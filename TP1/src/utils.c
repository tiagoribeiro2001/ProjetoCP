#include "../include/utils.h"

#define N 10000000 // Numero de amostras
#define K 4 // Numero de clusters

// Inicialização do array de pontos
point pontos[N];

// Inicialização do array de clusters
cluster clusters[K];

// Funcao que inicializa a lista de pontos e os clusters de forma aleatoria
void inicializa() {

    srand(10);
    
    // Preenche a lista de pontos
    for(int i = 0; i < N; i++) {
        point ponto = malloc(sizeof(struct Point));
        ponto->x = (float) rand() / RAND_MAX;
        ponto->y = (float) rand() / RAND_MAX;
        pontos[i] = ponto;
    }

    // Preenche a lista dos clusters
    for(int i = 0; i < K; i++) {
        point centroid = malloc(sizeof(struct Point));
        cluster cluster = malloc(sizeof(struct Cluster));
        centroid->x = pontos[i]->x;
        centroid->y = pontos[i]->y;
        centroid->clust = i;
        cluster->centroid = centroid;
        cluster->sumX = 0;
        cluster->sumY = 0;
        cluster->size = -1;
        clusters[i] = cluster;
    }
}

// Funcao que atribui todas as amostras ao seu devido cluster
void atribuiCluster(int it, int *convergiu){
    int conv = 1;

    // Reseta as variaveis de todos os clusters a 0
    for(int i = 0; i < K; i++ ) {
        clusters[i]->size = 0;
        clusters[i]->sumX = 0;
        clusters[i]->sumY = 0;
    }

    // Percorre a lista de amostras e atribui-as a um cluster
    for(int i = 0; i < N; i++){
        // Fórmula da distancia euclidiana
        float min = ((clusters[0]->centroid->x - pontos[i]->x) * (clusters[0]->centroid->x - pontos[i]->x)) + ((clusters[0]->centroid->y - pontos[i]->y) * (clusters[0]->centroid->y - pontos[i]->y));
        int c = 0;
        // Percorre os vários clusters para comparar a distancia da amostra ao seu centroid
        for(int j = 1; j < K; j++){
            float dist = ((clusters[j]->centroid->x - pontos[i]->x) * (clusters[j]->centroid->x - pontos[i]->x)) + ((clusters[j]->centroid->y - pontos[i]->y) * (clusters[j]->centroid->y - pontos[i]->y));
            if (dist < min){
                c = j;
                min = dist;
            }
        }

        // Se todas as amostras não mudarem de cluster, então convergiu (este teste nao é feito na atribuicao inicial das amostras por clusters)
        if (it == 0 || pontos[i]->clust != c){
            conv = 0;
        }

        clusters[c]->size++;
        clusters[c]->sumX += pontos[i]->x;
        clusters[c]->sumY += pontos[i]->y;
        pontos[i]->clust = c;
    }
    *convergiu = conv;
}

// Funcao que calcula o centroid de cada cluster
void calculaCentroid(){
    for(int i = 0; i < K; i++){
        // O centroid e calculado a partir da media de todos os pontos do cluster
        float mediaX =  clusters[i]->sumX / clusters[i]->size;
        float mediaY =  clusters[i]->sumY / clusters[i]->size;
        clusters[i]->centroid->x = mediaX;
        clusters[i]->centroid->y = mediaY;
    }
}

// Funcao que imprime informacao relativa ao estado final do programa
void printInfo(int it){
    printf("N = %d, K = %d\n", N, K);
    for (int i = 0; i < K; i++){
        printf("Center: (%.3f, %.3f) : Size %d\n", clusters[i]->centroid->x, clusters[i]->centroid->y, clusters[i]->size);
    }
    printf("Iterations: %d\n", it);
}