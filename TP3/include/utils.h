#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

__host__ void  inicializa(float *, float *);
__global__ void  atribuiCluster(float *, float *, float *, int *, int *, int *);
__global__ void calculaCentroid(float *, float *, float *, int *, int *);
__global__ void verificaConverge(int *, int *, int *);
__host__ void printInfo(int, float*, int *);