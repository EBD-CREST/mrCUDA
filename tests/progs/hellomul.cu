#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(func) \
    { \
        if ((func) != cudaSuccess ) { \
            fprintf(stderr, "ERROR\n"); \
            exit(EXIT_FAILURE); \
        } \
    }
 
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}

__global__ 
void null() 
{
}

 
int main()
{
    int i = 0;
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
 
	printf("%s", a);
 
    CUDA_SAFE_CALL(cudaSetDevice(1));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&ad, csize )); 
	CUDA_SAFE_CALL(cudaMalloc( (void**)&bd, isize )); 
	CUDA_SAFE_CALL(cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice )); 
	CUDA_SAFE_CALL(cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice )); 

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
	
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	CUDA_SAFE_CALL(cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost )); 
	
	printf("%s\n", a);
	return EXIT_SUCCESS;
}

