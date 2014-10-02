#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>

#define DEBUG 1

#if DEBUG
    #define DPRINTF(...) \
        {fprintf(stderr,  __VA_ARGS__);};
#else
    #define DPRINTF(fmt, ...) \
        {;;};
#endif

static void *nvida_cudart_handle;

__attribute__((constructor))
void init(void)
{
    DPRINTF("Enter init\n");
    nvida_cudart_handle = dlopen("/usr/local/cuda-5.0/lib64/libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if(!nvida_cudart_handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    dlerror();  /* Clear any existing error */
    DPRINTF("Exit init\n");
}

__attribute__((destructor))
void fini(void)
{
    DPRINTF("Enter fini\n");
    if(nvida_cudart_handle)
    {
        dlclose(nvida_cudart_handle);
    }
    DPRINTF("Exit fini\n");
}

void **__cudaRegisterFatBinary(void *fatCubin)
{
    void ** (*fn)(void *);
    char *error;

    DPRINTF("Enter __cudaRegisterFatBinary\n");
    *(void ***) (&fn) = dlsym(nvida_cudart_handle, "__cudaRegisterFatBinary");
    if((error = dlerror()) != NULL)
    {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }
    DPRINTF("Exit __cudaRegisterFatBinary\n");

    return (*fn)(fatCubin);
}
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*fn)(void **, size_t);
    char *error;

    DPRINTF("Enter cudaMalloc\n");
    fn = dlsym(nvida_cudart_handle, "cudaMalloc");
    if((error = dlerror()) != NULL)
    {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }
    DPRINTF("Exit cudaMalloc\n");

    return (*fn)(devPtr, size);
}

