#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

void *thread_main2(void *opaque)
{
    int *devMem;
    int device;
    pid_t pid;
    pid = syscall(SYS_gettid);
    printf("Thread 2: thread %d\n", pid);
    cudaGetDevice(&device);
    printf("Thread 2: Device %d\n", device);
    cudaSetDevice(0);
    cudaMalloc(&devMem, sizeof(int) * 16);
    printf("Thread 2: Addr %p\n", devMem);
    cudaGetDevice(&device);
    printf("Thread 2: Device %d\n", device);
    return NULL;
}

void *thread_main1(void *opaque)
{
    int *devMem;
    int device;
    pthread_t t;
    pid_t pid;
    pid = syscall(SYS_gettid);
    printf("Thread 1: thread %d\n", pid);
    cudaGetDevice(&device);
    printf("Thread 1: Device %d\n", device);
    cudaSetDevice(1);
    cudaMalloc(&devMem, sizeof(int) * 16);
    printf("Thread 1: Addr %p\n", devMem);
    cudaGetDevice(&device);
    printf("Thread 1: Device %d\n", device);
    pthread_create(&t, NULL, thread_main2, NULL);
    cudaGetDevice(&device);
    printf("Thread 1: Device %d\n", device);
    pthread_join(t, NULL);
    cudaGetDevice(&device);
    printf("Thread 1: Device %d\n", device);
    return NULL;
}
 
int main()
{
    int *devMem;
    int device;
    pthread_t t;
    pid_t pid;
    pid = syscall(SYS_gettid);
    printf("Main: thread %d\n", pid);
    cudaSetDevice(0);
    cudaMalloc(&devMem, sizeof(int) * 32);
    printf("Main: Addr %p\n", devMem);
    pthread_create(&t, NULL, thread_main1, NULL);
    cudaGetDevice(&device);
    printf("Main: Device %d\n", device);
    pthread_join(t, NULL);
    cudaGetDevice(&device);
    printf("Main: Device %d\n", device);
    return 0;
}
