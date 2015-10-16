#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MEMSIZE (1 << 30)

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

int main()
{
    int i = 0;
    struct timeval t1, t2;
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( 1, 1 );
    char *pDev0, *pDev1, *pHost;

    /* Initialize phase to force migration */
    if ((pHost = (char *)malloc(sizeof(char) * MEMSIZE)) == NULL) {
        perror("MALLOC ERROR:");
        exit(EXIT_FAILURE);
    }

    CUDA_SAFE_CALL(cudaSetDevice(0));
    CUDA_SAFE_CALL(cudaMalloc(&pDev0, sizeof(char) * MEMSIZE));
    CUDA_SAFE_CALL(cudaMemcpy(pDev0, pHost, sizeof(char) * MEMSIZE, cudaMemcpyHostToDevice));
    while (i < 2000) {
        null<<<dimBlock, dimBlock>>>();
        i++;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(pDev0, pHost, sizeof(char) * MEMSIZE, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaSetDevice(1));
    CUDA_SAFE_CALL(cudaMalloc(&pDev1, sizeof(char) * MEMSIZE));
    CUDA_SAFE_CALL(cudaMemcpy(pDev1, pHost, sizeof(char) * MEMSIZE, cudaMemcpyHostToDevice));
    i = 0;
    while (i < 2000) {
        null<<<dimBlock, dimBlock>>>();
        i++;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(pDev1, pHost, sizeof(char) * MEMSIZE, cudaMemcpyHostToDevice));

    //CUDA_SAFE_CALL(cudaSetDevice(0));
    /* mhelper benchmark phase */
    for (int iter = 0; iter < 20; iter++) {
        int size = sizeof(char) * (1 << (10 + iter));
        gettimeofday(&t1, NULL);
        for (int j = 0; j < 1000; j++)
            CUDA_SAFE_CALL(cudaMemcpy(pDev1, pHost, size, cudaMemcpyHostToDevice));
        gettimeofday(&t2, NULL);
        printf("%d %f\n", size, get_elapsed_time(&t1, &t2));
    }

	return EXIT_SUCCESS;
}

