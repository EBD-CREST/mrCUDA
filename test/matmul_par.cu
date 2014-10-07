#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define BS (16)
#define L (16)
#define M (16)
#define N (16)

__global__ void matmul(float *A, float *B, float *C,
                       int l, int m, int n)
{
    int i, j, k;
    float sum;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    sum = 0.0;
    for (k = 0; k < m; k++) {
        sum += A[i * m + k] * B[k * n + j];
    }
    C[i*n+j] = sum;
}

__global__ void thread_matrix(float *A,
                       int l, int n)
{
    int i, j;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    A[i * n + j] = i * n + j;
}

void matmul_cpu(float *A, float *B, float *C,
                       int l, int m, int n)
{
    int i, j, k;
    for (i = 0; i < l; i++) {
        for (j = 0; j < n; j++) {
            float sum = 0.0;
            for (k = 0; k < m; k++) {
                sum += A[i * m + k] * B[k * n + j];
            }
            C[i*n+j] = sum;
        }
    }
}

void print_matrix(float *A, int l, int n)
{
    int i, j;
    for (i = 0; i < l; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
}

int compare_matrix(float *A, float *B, int l, int n)
{
    int i, j;
    int ret = 0;
    for (i = 0; i < l; i++) {
        for (j = 0; j < n; j++) {
            if(A[i * n + j] != B[i * n + j])
                ret = -1;
        }
    }
    return ret;
}

void alloc_matrix(float **m_h, float **m_d, int h, int w)
{
    *m_h = (float *)malloc(sizeof(float) * h * w);
    cudaMalloc((void **)m_d, sizeof(float) * h * w);
}

void init_matrix(float *m, int h, int w)
{
    int i, j;
    for (i = 0; i < h; i++)
        for (j = 0; j < w; j++)
            m[i * w + j] = (float)(random() % 100);
}

int check_error(const char *err_msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s.\n",
                err_msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

double get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000
            + (end->tv_usec - begin->tv_usec) / 1000.0;
}

int main(int argc, char *argv[])
{
    float *Ad, *Bd, *Cd;
    float *Ah, *Bh, *Ch;
    struct timeval t1, t2;

    // prepare matrix A
    alloc_matrix(&Ah, &Ad, L, M);
    init_matrix(Ah, L, M);
    cudaMemcpy(Ad, Ah, sizeof(float) * L * M,
               cudaMemcpyHostToDevice);
    // do it again for matrix B
    alloc_matrix(&Bh, &Bd, M, N);
    init_matrix(Bh, M, N);
    cudaMemcpy(Bd, Bh, sizeof(float) * M * N,
               cudaMemcpyHostToDevice);
    // allocate spaces for matrix C
    alloc_matrix(&Ch, &Cd, L, N);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    // launch matmul kernel
    matmul<<<dim3(N / BS, L / BS),
            dim3(BS, BS)>>>(Ad, Bd, Cd, L, M, N);

    if (check_error("matmul")) {
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    printf("Elapsed time: %f msec\n", get_elapsed_time(&t1, &t2));
    
    // obtain the result
    cudaMemcpy(Ch, Cd, sizeof(float) * L * N, cudaMemcpyDeviceToHost);
    float *C_cpu = (float *)malloc(sizeof(float) * L * N);
    matmul_cpu(Ah, Bh, C_cpu, L, M, N);
    print_matrix(Ch, L, N);
    printf("\n");
    print_matrix(C_cpu, L, N);
    printf("\n");

    if(compare_matrix(Ch, C_cpu, L, N) >= 0)
        printf("OK\n");
    else
        printf("ERRRRR\n");

    /* Switch to native */
    cudaMalloc(NULL, 0);
    printf("Switched to native.....\n");
    printf("Press enter to continue...\n");
    getchar();

    thread_matrix<<<dim3(N / BS, L / BS),
            dim3(BS, BS)>>>(Cd, L, N);
    cudaMemcpy(Ch, Cd, sizeof(float) * L * N, cudaMemcpyDeviceToHost);
    print_matrix(Ch, L, N);
    printf("\n");

    matmul<<<dim3(N / BS, L / BS),
            dim3(BS, BS)>>>(Ad, Bd, Cd, L, M, N);
    cudaMemcpy(Ch, Cd, sizeof(float) * L * N, cudaMemcpyDeviceToHost);
    print_matrix(Ch, L, N);
    printf("\n");
    print_matrix(C_cpu, L, N);
    printf("\n");
    if(compare_matrix(Ch, C_cpu, L, N) >= 0)
        printf("OK\n");
    else
        printf("ERRRRR\n");

    free(C_cpu);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    return 0;
}



