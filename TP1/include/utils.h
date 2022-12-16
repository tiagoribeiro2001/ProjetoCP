#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Estrutura de dados do ponto
typedef struct Point{
    int clust;
    float x;
    float y;
} *point;

// Estrutura de dados do cluster
typedef struct Cluster{
    point centroid;
    int size;
    float sumX;
    float sumY;
} *cluster;

void inicializa();
void atribuiCluster(int, int*);
void calculaCentroid();
void printInfo(int);