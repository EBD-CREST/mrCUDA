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

static void *__nvidia_cudart_handle;
static void *__rcuda_cudart_handle;

static void ** (*__nvidia___cudaRegisterFatBinary)(void *);
static void CUDARTAPI (*__nvidia___cudaRegisterFunction)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__nvidia_cudaMalloc)(void **, size_t);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaLaunch)(const void *);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaSetupArgument)(const void *, size_t, size_t);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);

static void ** (*__rcuda___cudaRegisterFatBinary)(void *);
static void CUDARTAPI (*__rcuda___cudaRegisterFunction)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__rcuda_cudaMalloc)(void **, size_t);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaLaunch)(const void *);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaSetupArgument)(const void *, size_t, size_t);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);


static void **__nvidia_fatCubinHandle;
static void **__rcuda_fatCubinHandle;

static inline void *__safe_dlsym(void *handle, const char *symbol)
{
    char *error;
    void *ret_handle = dlsym(handle, symbol);
    
    if((error = dlerror()) != NULL)
    {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

    return ret_handle;
}

__attribute__((constructor))
void init(void)
{
    DPRINTF("Enter init\n");

    __rcuda_cudart_handle = dlopen("/home/pak/src/mrCUDA/libs/rCUDA/framework/rCUDAl/libcudart.so.5.0", RTLD_NOW | RTLD_GLOBAL);
    __nvidia_cudart_handle = dlopen("/usr/local/cuda-5.0/lib64/libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if(!__rcuda_cudart_handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    if(!__nvidia_cudart_handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    dlerror();  /* Clear any existing error */

    __nvidia___cudaRegisterFatBinary = __safe_dlsym(__nvidia_cudart_handle, "__cudaRegisterFatBinary");
    __nvidia___cudaRegisterFunction = __safe_dlsym(__nvidia_cudart_handle, "__cudaRegisterFunction");
    __nvidia_cudaMalloc = __safe_dlsym(__nvidia_cudart_handle, "cudaMalloc");
    __nvidia_cudaMemcpy = __safe_dlsym(__nvidia_cudart_handle, "cudaMemcpy");
    __nvidia_cudaLaunch = __safe_dlsym(__nvidia_cudart_handle, "cudaLaunch");
    __nvidia_cudaSetupArgument = __safe_dlsym(__nvidia_cudart_handle, "cudaSetupArgument");
    __nvidia_cudaConfigureCall = __safe_dlsym(__nvidia_cudart_handle, "cudaConfigureCall");

    __rcuda___cudaRegisterFatBinary = __safe_dlsym(__rcuda_cudart_handle, "__cudaRegisterFatBinary");
    __rcuda___cudaRegisterFunction = __safe_dlsym(__rcuda_cudart_handle, "__cudaRegisterFunction");
    __rcuda_cudaMalloc = __safe_dlsym(__rcuda_cudart_handle, "cudaMalloc");
    __rcuda_cudaMemcpy = __safe_dlsym(__rcuda_cudart_handle, "cudaMemcpy");
    __rcuda_cudaLaunch = __safe_dlsym(__rcuda_cudart_handle, "cudaLaunch");
    __rcuda_cudaSetupArgument = __safe_dlsym(__rcuda_cudart_handle, "cudaSetupArgument");
    __rcuda_cudaConfigureCall = __safe_dlsym(__rcuda_cudart_handle, "cudaConfigureCall");

    DPRINTF("Exit init\n");
}

__attribute__((destructor))
void fini(void)
{
    DPRINTF("Enter fini\n");
    if(__nvidia_cudart_handle)
    {
        dlclose(__nvidia_cudart_handle);
    }
    if(__rcuda_cudart_handle)
    {
        dlclose(__rcuda_cudart_handle);
    }
    DPRINTF("Exit fini\n");
}

void **__cudaRegisterFatBinary(void *fatCubin)
{
    DPRINTF("Enter __cudaRegisterFatBinary\n");
    __nvidia_fatCubinHandle = (*__nvidia___cudaRegisterFatBinary)(fatCubin);
    __rcuda_fatCubinHandle = (*__rcuda___cudaRegisterFatBinary)(fatCubin);
    DPRINTF("Exit __cudaRegisterFatBinary\n");

    return __rcuda_fatCubinHandle;
}

extern void CUDARTAPI __cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
)
{
    DPRINTF("Enter __cudaRegisterFunction\n");
    (*__nvidia___cudaRegisterFunction)(
        __nvidia_fatCubinHandle,
        hostFun,
        deviceFun,
        deviceName,
        thread_limit,
        tid,
        bid,
        bDim,
        gDim,
        wSize
    );
    (*__rcuda___cudaRegisterFunction)(
       fatCubinHandle,
        hostFun,
        deviceFun,
        deviceName,
        thread_limit,
        tid,
        bid,
        bDim,
        gDim,
        wSize
    );
    DPRINTF("Exit __cudaRegisterFunction\n");
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t ret;
    DPRINTF("Enter cudaMalloc\n");
    (*__nvidia_cudaMalloc)(devPtr, size);
    ret = (*__rcuda_cudaMalloc)(devPtr, size);
    DPRINTF("Exit cudaMalloc\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t ret;
    DPRINTF("Enter cudaMemcpy\n");
    (*__nvidia_cudaMemcpy)(dst, src, count, kind);
    if(kind == cudaMemcpyDeviceToHost)
    {
        printf("nvidia: %s\n", (char *)dst);
    }
    ret = (*__rcuda_cudaMemcpy)(dst, src, count, kind);
    if(kind == cudaMemcpyDeviceToHost)
    {
        printf("rcuda: %s\n", (char *)dst);
    }
    DPRINTF("Exit cudaMemcpy\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
    cudaError_t ret;
    DPRINTF("Enter cudaLaunch\n");
    (*__nvidia_cudaLaunch)(func);
    ret = (*__rcuda_cudaLaunch)(func);
    DPRINTF("Exit cudaLaunch\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    cudaError_t ret;
    DPRINTF("Enter cudaSetupArgument\n");
    (*__nvidia_cudaSetupArgument)(arg, size, offset);
    ret = (*__rcuda_cudaSetupArgument)(arg, size, offset);
    DPRINTF("Exit cudaSetupArgument\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    cudaError_t ret;
    DPRINTF("Enter cudaConfigureCall\n");
    (*__nvidia_cudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
    ret = (*__rcuda_cudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
    DPRINTF("Exit cudaConfigureCall\n");
    return ret;
}
