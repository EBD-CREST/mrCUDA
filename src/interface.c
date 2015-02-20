#include "common.h"
#include "mrcuda.h"
#include "record.h"

/**
 * Perform barrier checking and increase the number of executing functions by 1.
 * If this function finds a barrier, it blocks the execution until the barrier is left.
 */
static void inline __mrcuda_function_enter()
{
}

/**
 * Decrease the number of executing functions by 1.
 */
static void inline __mrcuda_function_exit()
{
}

/**
 * Interface of cudaThreadSynchronize.
 */
extern __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaThreadSynchronize();
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaLaunch.
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaLaunch(func);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaMemcpyToSymbol.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaMemcpyToSymbol(symbol, src, count, offset, kind);
    mrcuda_record_cudaMemcpyToSymbol(symbol, count, offset, kind);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaMemcpy.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaMemcpy(dst, src, count, kind);
    mrcuda_record_cudaMemcpy(dst, count, kind);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaHostAlloc.
 */
extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaHostAlloc(pHost, size, flags);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaMemset.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaMemset(devPtr, value, count);
    mrcuda_record_cudaMemset(devPtr, value, count);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaFreeHost.
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaFreeHost(ptr);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaSetupArgument.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaSetupArgument(arg, size, offset);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaMalloc.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaMalloc(devPtr, size);
    mrcuda_record_cudaMalloc(size);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaFree.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaFree(devPtr);
    mrcuda_record_cudaFree(devPtr);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaConfigureCall.
 */
extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaConfigureCall(girdDim, blockDim, sharedMem, stream);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaGetLastError.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaGetLastError();
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaBindTexture.
 */
extern __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaBindTexture(offset, texref, devPtr, desc, size);
    mrcuda_record_cudaBindTexture(textref, devPtr, desc, size);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaCreateChannelDesc.
 */
extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    struct cudaChannelFormatDesc ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaCreateChannelDesc(x, y, z, w, f);
    mrcuda_record_cudaCreateChannelDesc(x, y, z, w, f);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaGetDeviceProperties.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaGetDeviceProperties(prop, device);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaStreamCreate.
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->mrcudaStreamCreate(pStream);
    mrcuda_record_cudaStreamCreate();
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaMemGetInfo.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaMemGetInfo(free, total);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaSetDevice.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaSetDevice(device);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaSetDeviceFlags.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags )
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaSetDeviceFlags(flags);
    mrcuda_record_cudaSetDeviceFlags(flags);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaGetDevice.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaGetDevice(device);
    __mrcuda_function_exit();
    return ret;
}

/**
 * Interface of cudaGetDeviceCount.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    cudaError_t ret;
    __mrcuda_function_enter();
    ret = mrcudaSymDefault->cudaGetDeviceCount(count);
    __mrcuda_function_exit();
    return ret;
}

