#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

void inicializa(float *, float *, float *, int *, int *);
void __global__ atribuiCluster(float *, float *, float *, int *, int *);
void calculaCentroid(float *, float *, int *);
void printInfo(float*, int *);