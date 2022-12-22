#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

__host__ void  inicializa(float *, float *);
__global__ void  atribuiCluster(float *, float *, int *);
__host__ void calculaCentroid(float *, float *, float *, int *, int *);
__host__ int verificaConverge(int *, int *);
__host__ void printInfo(int, float*, int *);