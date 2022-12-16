CC       = gcc
BIN      = bin/
SRC      = src/
INCLUDES = include/
EXEC     = k_means
THREADS  = 16

CFLAGS = -O2 -fopenmp -std=c99 -Ofast -fomit-frame-pointer -ftree-vectorize -funroll-loops -msse4

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)$(EXEC)

clean:
	rm -r bin/*

runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) 1

runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)