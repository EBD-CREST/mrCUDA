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
    float *a;
    CUDA_SAFE_CALL(cudaMalloc(&a, sizeof(float)));
    printf("a is %p\n", a);
    getchar();
    return 0;
}

