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

/* NVIDIA CUDA */
static void ** (*__nvidia___cudaRegisterFatBinary)(void *);
static void CUDARTAPI (*__nvidia___cudaRegisterFunction)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__nvidia_cudaMalloc)(void **, size_t);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaLaunch)(const void *);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaSetupArgument)(const void *, size_t, size_t);
static __host__ cudaError_t CUDARTAPI (*__nvidia_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__nvidia_cudaFree)(void *);

/* RCUDA */
static void ** (*__rcuda___cudaRegisterFatBinary)(void *);
static void CUDARTAPI (*__rcuda___cudaRegisterFunction)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__rcuda_cudaMalloc)(void **, size_t);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaLaunch)(const void *);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaSetupArgument)(const void *, size_t, size_t);
static __host__ cudaError_t CUDARTAPI (*__rcuda_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__rcuda_cudaFree)(void *);

/* Default */
static void ** (*__default___cudaRegisterFatBinary)(void *);
static void CUDARTAPI (*__default___cudaRegisterFunction)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__default_cudaMalloc)(void **, size_t);
static __host__ cudaError_t CUDARTAPI (*__default_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
static __host__ cudaError_t CUDARTAPI (*__default_cudaLaunch)(const void *);
static __host__ cudaError_t CUDARTAPI (*__default_cudaSetupArgument)(const void *, size_t, size_t);
static __host__ cudaError_t CUDARTAPI (*__default_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
static __host__ __cudart_builtin__ cudaError_t CUDARTAPI (*__default_cudaFree)(void *);


static void **__nvidia_fatCubinHandle;
static void **__rcuda_fatCubinHandle;

typedef struct __CacheMalloc
{
    void *devPtr;
    size_t size;
    struct __CacheMalloc *next;
} __CacheMalloc;

static __CacheMalloc *__first_cachemalloc = NULL;
static __CacheMalloc *__last_cachemalloc = NULL;

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
    __nvidia_cudaFree = __safe_dlsym(__nvidia_cudart_handle, "cudaFree");

    __rcuda___cudaRegisterFatBinary = __safe_dlsym(__rcuda_cudart_handle, "__cudaRegisterFatBinary");
    __rcuda___cudaRegisterFunction = __safe_dlsym(__rcuda_cudart_handle, "__cudaRegisterFunction");
    __rcuda_cudaMalloc = __safe_dlsym(__rcuda_cudart_handle, "cudaMalloc");
    __rcuda_cudaMemcpy = __safe_dlsym(__rcuda_cudart_handle, "cudaMemcpy");
    __rcuda_cudaLaunch = __safe_dlsym(__rcuda_cudart_handle, "cudaLaunch");
    __rcuda_cudaSetupArgument = __safe_dlsym(__rcuda_cudart_handle, "cudaSetupArgument");
    __rcuda_cudaConfigureCall = __safe_dlsym(__rcuda_cudart_handle, "cudaConfigureCall");
    __rcuda_cudaFree = __safe_dlsym(__rcuda_cudart_handle, "cudaFree");

    __default___cudaRegisterFatBinary = __rcuda___cudaRegisterFatBinary;
    __default___cudaRegisterFunction = __rcuda___cudaRegisterFunction;
    __default_cudaMalloc = __rcuda_cudaMalloc;
    __default_cudaMemcpy = __rcuda_cudaMemcpy;
    __default_cudaLaunch = __rcuda_cudaLaunch;
    __default_cudaSetupArgument = __rcuda_cudaSetupArgument;
    __default_cudaConfigureCall = __rcuda_cudaConfigureCall;
    __default_cudaFree = __rcuda_cudaFree;

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

extern void mrcudaSwitchToNative()
{
    __CacheMalloc *cachemalloc = __first_cachemalloc;
    void *devPtr;
    cudaError_t error;
    void *dataCache;

    /* Replay cudaMalloc */
    DPRINTF("Enter mrcudaSwitchToNative\n");
    while(cachemalloc != NULL)
    {
        error = (*__nvidia_cudaMalloc)(&devPtr, cachemalloc->size);
        if(error != cudaSuccess)
        {
            fprintf(stderr, "Cannot replay cudaMalloc.\n");
            exit(EXIT_FAILURE);
        }
        dataCache = malloc(cachemalloc->size);
        if(dataCache == NULL)
        {
            fprintf(stderr, "Cannot allocate dataCache.\n");
            exit(EXIT_FAILURE);
        }
        error = (*__rcuda_cudaMemcpy)(dataCache, cachemalloc->devPtr, cachemalloc->size, cudaMemcpyDeviceToHost);
        if(error != cudaSuccess)
        {
            fprintf(stderr, "Cannot perform __rcuda_cudaMemcpy.\n");
            exit(EXIT_FAILURE);
        }
        error = (*__nvidia_cudaMemcpy)(devPtr, dataCache, cachemalloc->size, cudaMemcpyHostToDevice);
        if(error != cudaSuccess)
        {
            fprintf(stderr, "Cannot perform __nvidia_cudaMemcpy.\n");
            exit(EXIT_FAILURE);
        }
        free(dataCache);
        __first_cachemalloc = cachemalloc->next;
        free(cachemalloc);
        cachemalloc = __first_cachemalloc;
    }
    __first_cachemalloc = NULL;
    __last_cachemalloc = NULL;

    /* Change default function handler */
    __default___cudaRegisterFatBinary = __nvidia___cudaRegisterFatBinary;
    __default___cudaRegisterFunction = __nvidia___cudaRegisterFunction;
    __default_cudaMalloc = __nvidia_cudaMalloc;
    __default_cudaMemcpy = __nvidia_cudaMemcpy;
    __default_cudaLaunch = __nvidia_cudaLaunch;
    __default_cudaSetupArgument = __nvidia_cudaSetupArgument;
    __default_cudaConfigureCall = __nvidia_cudaConfigureCall;
    __default_cudaFree = __nvidia_cudaFree;
    DPRINTF("Exit mrcudaSwitchToNative\n");
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
    cudaError_t ret = cudaSuccess;
    __CacheMalloc *cachemalloc;

    if(devPtr == NULL && size == 0)
        mrcudaSwitchToNative();
    else
    {
        DPRINTF("Enter cudaMalloc\n");
        ret = (*__default_cudaMalloc)(devPtr, size);
        if(__default_cudaMalloc == __rcuda_cudaMalloc)
        {
            cachemalloc = malloc(sizeof(__CacheMalloc));
            if(cachemalloc == NULL)
            {
                fprintf(stderr, "Cannot allocate a new __CacheMalloc.\n");
                exit(EXIT_FAILURE);
            }
            cachemalloc->devPtr = *devPtr;
            cachemalloc->size = size;
            cachemalloc->next = NULL;
            if(__first_cachemalloc == NULL)
                __first_cachemalloc = cachemalloc;
            if(__last_cachemalloc != NULL)
                __last_cachemalloc->next = cachemalloc;
            __last_cachemalloc = cachemalloc;
        }
        DPRINTF("Exit cudaMalloc\n");
    }
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t ret;
    DPRINTF("Enter cudaMemcpy\n");
    ret = (*__default_cudaMemcpy)(dst, src, count, kind);
    DPRINTF("Exit cudaMemcpy\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
    cudaError_t ret;
    DPRINTF("Enter cudaLaunch\n");
    ret = (*__default_cudaLaunch)(func);
    DPRINTF("Exit cudaLaunch\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    cudaError_t ret;
    DPRINTF("Enter cudaSetupArgument\n");
    ret = (*__default_cudaSetupArgument)(arg, size, offset);
    DPRINTF("Exit cudaSetupArgument\n");
    return ret;
}

extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    cudaError_t ret;
    DPRINTF("Enter cudaConfigureCall\n");
    ret = (*__default_cudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
    DPRINTF("Exit cudaConfigureCall\n");
    return ret;
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    cudaError_t ret;
    DPRINTF("Enter cudaFree\n");
    ret = (*__default_cudaFree)(devPtr);
    DPRINTF("Exit cudaFree\n");
    return ret;
}
