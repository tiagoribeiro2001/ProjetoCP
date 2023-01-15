#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// Estrutura de dados do ponto
typedef struct Point{
    float x;
    float y;
} point;

// Estrutura de dados de cluster
typedef struct Cluster{
    float sum_x;
    float sum_y;
    int size;
} cluster;

__host__ void inicializa(point *, point *, cluster *, int *, int, int, int, int);
__global__ void atribuiCluster(point *, point *, cluster *, int, int, int, int);
__global__ void calculaCentroids(point *, cluster*, int *, int, int, int);
__host__ void printInfo(int, point *, int *, int, int);
