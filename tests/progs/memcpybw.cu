#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(func) \
    { \
        if ((func) != cudaSuccess ) { \
            fprintf(stderr, "ERROR\n"); \
            exit(EXIT_FAILURE); \
        } \
    }

static inline double get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000
            + (end->tv_usec - begin->tv_usec) / 1000.0;
}

__global__ 
void null() 
{
}

int main(int argc, char *argv[])
{
    int i = 0;
    struct timeval t1, t2;
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( 1, 1 );
    char *pDev, *pHost;
    char *endpoint;
    size_t memsize;
    int num;
    size_t secSize;

    if (argc < 3) {
        fprintf(stderr, "prog memsize num\n");
        exit(EXIT_FAILURE);
    }

    memsize = strtol(argv[1], &endpoint, 10);
    if (*endpoint != '\0') {
        fprintf(stderr, "memsize has to be long int.\n");
        exit(EXIT_FAILURE);
    }

    num = (int)strtol(argv[2], &endpoint, 10);
    if (*endpoint != '\0') {
        fprintf(stderr, "num has to be int.\n");
        exit(EXIT_FAILURE);
    }

    secSize = memsize / num;

    /* Initialize phase to force migration */
    if ((pHost = (char *)malloc(sizeof(char) * memsize)) == NULL) {
        perror("MALLOC ERROR:");
        exit(EXIT_FAILURE);
    }

    CUDA_SAFE_CALL(cudaMalloc(&pDev, sizeof(char) * secSize));
    CUDA_SAFE_CALL(cudaMemcpy(pDev, pHost, sizeof(char) * secSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFree(pDev));
    gettimeofday(&t1, NULL);
    for (i = 0; i < num; i++) {
        CUDA_SAFE_CALL(cudaMalloc(&pDev, sizeof(char) * secSize));
        CUDA_SAFE_CALL(cudaMemcpy(pDev, pHost, sizeof(char) * secSize, cudaMemcpyHostToDevice));
    }
    gettimeofday(&t2, NULL);
    printf("Elapsed Time: %f\n", get_elapsed_time(&t1, &t2));
    while (i < 2000) {
        null<<<dimBlock, dimBlock>>>();
        i++;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
	return EXIT_SUCCESS;
}

