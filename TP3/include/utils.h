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
typedef struct SumThread{
    float sum_x;
    float sum_y;
    int size;
} sumThread;

__host__ void inicializa(point *, point *, sumThread *, int *, int, int, int);
__global__ void calculaCluster(point *, point *, sumThread *, int, int, int);
__global__ void calculaCentroids(point *, sumThread *, int *, int, int);
__host__ void printInfo(int, point *, int *, int, int);
