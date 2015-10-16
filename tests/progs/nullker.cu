#include <stdio.h>
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
 
int main()
{
    int i = 0;
    struct timeval t1, t2;
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( 1, 1 );

    /* Initialize phase to force migration */
    CUDA_SAFE_CALL(cudaSetDevice(0));
    while (i < 20) {
        null<<<dimBlock, dimBlock>>>();
        i++;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaSetDevice(1));
    i = 0;
    while (i < 20) {
        null<<<dimBlock, dimBlock>>>();
        i++;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //CUDA_SAFE_CALL(cudaSetDevice(0));
    /* mhelper benchmark phase */
    for (int iter = 0; iter < 15; iter++) {
        int j = (1 << (10 + iter)) - 1;
        i = 0;
        gettimeofday(&t1, NULL);
        while (i < j) {
            null<<<dimBlock, dimBlock>>>();
            i++;
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        gettimeofday(&t2, NULL);
        printf("%d %f\n", j + 1, get_elapsed_time(&t1, &t2));
    }

	return EXIT_SUCCESS;
}
