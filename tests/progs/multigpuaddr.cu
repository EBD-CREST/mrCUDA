#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(x) \
{ \
    if ((x) != cudaSuccess) { \
        fprintf(stderr, "Error!");   \
        exit(EXIT_FAILURE); \
    } \
}

int main()
{
    float *a, *b;
    CUDA_SAFE_CALL(cudaSetDevice(0));
    CUDA_SAFE_CALL(cudaMalloc(&a, sizeof(float)));
    CUDA_SAFE_CALL(cudaSetDevice(1));
    CUDA_SAFE_CALL(cudaMalloc(&b, sizeof(float)));
    printf("a on device 0 is %p\n", a);
    printf("b on device 1 is %p\n", b);
    return 0;
}

