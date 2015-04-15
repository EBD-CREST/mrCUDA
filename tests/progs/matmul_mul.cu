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
    float *Ad1, *Bd1, *Cd1;
    float *Ah1, *Bh1, *Ch1;
    float *Ad2, *Bd2, *Cd2;
    float *Ah2, *Bh2, *Ch2;
    struct timeval t1, t2;
    float *C_cpu;

    int num_device = 0;

    if (cudaGetDeviceCount(&num_device) != cudaSuccess || num_device < 2) {
        fprintf(stderr, "This program needs at least 2 devices.\n");
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(0);

    // prepare matrix A
    alloc_matrix(&Ah1, &Ad1, L, M);
    init_matrix(Ah1, L, M);
    cudaMemcpy(Ad1, Ah1, sizeof(float) * L * M,
               cudaMemcpyHostToDevice);
    // do it again for matrix B
    alloc_matrix(&Bh1, &Bd1, M, N);
    init_matrix(Bh1, M, N);
    cudaMemcpy(Bd1, Bh1, sizeof(float) * M * N,
               cudaMemcpyHostToDevice);
    // allocate spaces for matrix C
    alloc_matrix(&Ch1, &Cd1, L, N);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    // launch matmul kernel
    matmul<<<dim3(N / BS, L / BS),
            dim3(BS, BS)>>>(Ad1, Bd1, Cd1, L, M, N);

    if (check_error("matmul")) {
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    printf("Elapsed time: %f msec\n", get_elapsed_time(&t1, &t2));
    
    // obtain the result
    cudaMemcpy(Ch1, Cd1, sizeof(float) * L * N, cudaMemcpyDeviceToHost);
    C_cpu = (float *)malloc(sizeof(float) * L * N);
    matmul_cpu(Ah1, Bh1, C_cpu, L, M, N);
    print_matrix(Ch1, L, N);
    printf("\n");
    print_matrix(C_cpu, L, N);
    printf("\n");

    if(compare_matrix(Ch1, C_cpu, L, N) >= 0)
        printf("OK\n");
    else
        printf("ERRRRR\n");

    free(C_cpu);

    cudaSetDevice(1);

    // prepare matrix A
    alloc_matrix(&Ah2, &Ad2, L, M);
    init_matrix(Ah2, L, M);
    cudaMemcpy(Ad2, Ah2, sizeof(float) * L * M,
               cudaMemcpyHostToDevice);
    // do it again for matrix B
    alloc_matrix(&Bh2, &Bd2, M, N);
    init_matrix(Bh2, M, N);
    cudaMemcpy(Bd2, Bh2, sizeof(float) * M * N,
               cudaMemcpyHostToDevice);
    // allocate spaces for matrix C
    alloc_matrix(&Ch2, &Cd2, L, N);

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    // launch matmul kernel
    matmul<<<dim3(N / BS, L / BS),
            dim3(BS, BS)>>>(Ad2, Bd2, Cd2, L, M, N);

    if (check_error("matmul")) {
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    printf("Elapsed time: %f msec\n", get_elapsed_time(&t1, &t2));
    
    // obtain the result
    cudaMemcpy(Ch2, Cd2, sizeof(float) * L * N, cudaMemcpyDeviceToHost);
    C_cpu = (float *)malloc(sizeof(float) * L * N);
    matmul_cpu(Ah2, Bh2, C_cpu, L, M, N);
    print_matrix(Ch2, L, N);
    printf("\n");
    print_matrix(C_cpu, L, N);
    printf("\n");

    if(compare_matrix(Ch2, C_cpu, L, N) >= 0)
        printf("OK\n");
    else
        printf("ERRRRR\n");

    free(C_cpu);

    cudaFree(Ad1);
    cudaFree(Bd1);
    cudaFree(Cd1);

    cudaFree(Ad2);
    cudaFree(Bd2);
    cudaFree(Cd2);

    return 0;
}



