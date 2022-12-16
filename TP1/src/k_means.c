#include "../include/utils.h"

// Flag que indica 0 se o algoritmo não convergiu e 1 se convergiu
int convergiu = 0;

// Contador da iteracoes
int iteration = 0;

int main(){
    clock_t begin = clock();
    inicializa();
    atribuiCluster(iteration, &convergiu);
    while(!convergiu) {
        calculaCentroid();
        atribuiCluster(iteration, &convergiu);
        iteration++;
    }
    printInfo(iteration);
    clock_t end = clock();
    double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
    printf("Execution time: %f\n",time_spent);
    return 0;
}