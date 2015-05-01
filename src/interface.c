#include "common.h"
#include "mrcuda.h"
#include "datatypes.h"
#include "record.h"

static long int __cudaLaunchCount = 0;

/**
 * Interface of __cudaRegisterFatBinary.
 */
void** __cudaRegisterFatBinary(void* fatCubin)
{
    void **ret;
    MRCUDAGPU_t *gpu;
    mrcuda_init();
    gpu = mrcuda_get_current_gpu();
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->__mrcudaRegisterFatBinary(fatCubin);
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaRegisterFatBinary(gpu, fatCubin, ret);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of __cudaRegisterFunction.
 */
void __cudaRegisterFunction(
    void **fatCubinHandle,
    const char *hostFun,
    char *deviceFun,
    const char *deviceName,
    int thread_limit,
    uint3 *tid,
    uint3 *bid,
    dim3 *bDim,
    dim3 *gDim,
    int *wSize
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    mrcuda_function_call_lock(gpu);
    gpu->defaultHandler->__mrcudaRegisterFunction(
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
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaRegisterFunction(
            gpu,
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
    mrcuda_function_call_release(gpu);
}

/**
 * Interface of __cudaRegisterVar.
 */
void __cudaRegisterVar(
    void **fatCubinHandle,
    char *hostVar,
    char *deviceAddress,
    const char *deviceName,
    int ext,
    int size,
    int constant,
    int global
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    mrcuda_function_call_lock(gpu);
    gpu->defaultHandler->__mrcudaRegisterVar(
        fatCubinHandle,
        hostVar,
        deviceAddress,
        deviceName,
        ext,
        size,
        constant,
        global
    );
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaRegisterVar(
            gpu,
            fatCubinHandle,
            hostVar,
            deviceAddress,
            deviceName,
            ext,
            size,
            constant,
            global
        );
    mrcuda_function_call_release(gpu);
}

/**
 * Interface of __cudaRegisterTexture.
 */
void __cudaRegisterTexture(
    void **fatCubinHandle,
    const struct textureReference *hostVar,
    const void **deviceAddress,
    const char *deviceName,
    int dim,
    int norm,
    int ext
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    mrcuda_function_call_lock(gpu);
    gpu->defaultHandler->__mrcudaRegisterTexture(
        fatCubinHandle,
        hostVar,
        deviceAddress,
        deviceName,
        dim,
        norm,
        ext
    );
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaRegisterTexture(
            gpu,
            fatCubinHandle,
            hostVar,
            deviceAddress,
            deviceName,
            dim,
            norm,
            ext
        );
    mrcuda_function_call_release(gpu);
}

/**
 * Interface of __cudaUnregisterFatBinary.
 */
void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    /*mrcuda_function_call_lock();
    mrcudaSymDefault->__mrcudaUnregisterFatBinary(
        fatCubinHandle
    );
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaUnregisterFatBinary(fatCubinHandle);
    mrcuda_function_call_release();*/
}

/**
 * Interface of cudaThreadSynchronize.
 */
extern __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void)
{
    // cudaThreadSynchronize eventually calls cudaDeviceSynchronize.
    // Thus, locking cannot be done here since it will cause dead-lock.
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    ret = gpu->defaultHandler->mrcudaThreadSynchronize();
    return ret;
}

/**
 * Interface of cudaLaunch.
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaLaunch(func);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaMemcpyToSymbol.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(
    const void *symbol, 
    const void *src, 
    size_t count, 
    size_t offset, 
    enum cudaMemcpyKind kind
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaMemcpyToSymbol(symbol, src, count, offset, kind);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaMemcpy.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy(
    void *dst, 
    const void *src, 
    size_t count, 
    enum cudaMemcpyKind kind
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaMemcpy(dst, src, count, kind);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaHostAlloc.
 */
extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    void *pHost1;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaHostAlloc(pHost, size, flags);
    if (!gpu->nativeFromStart)
        // This function has to be recorded regardless we are using rCUDA or not.
        // This ensures that we calls cudaFreeHost using the right library (rCUDA or native).
        // However, GPUs that are running natively from the start don't need to be recorded.
        mrcuda_record_cudaHostAlloc(gpu, pHost, size, flags);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaMemset.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaMemset(devPtr, value, count);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaFreeHost.
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    if (!gpu->nativeFromStart)
        // Call the right library of cudaFreeHost according to the recorded cudaHostAlloc calls.
        mrcuda_replay_cudaFreeHost(gpu, ptr)->mrcudaFreeHost(ptr);
    else
        gpu->defaultHandler->mrcudaFreeHost(ptr);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaSetupArgument.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaSetupArgument(arg, size, offset);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaMalloc.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaMalloc(devPtr, size);
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaMalloc(gpu, devPtr, size);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaFree.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaFree(devPtr);
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaFree(gpu, devPtr);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaConfigureCall.
 */
extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(
    dim3 gridDim, 
    dim3 blockDim, 
    size_t sharedMem, 
    cudaStream_t stream
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaGetLastError.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaGetLastError();
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaBindTexture.
 */
extern __host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t *offset, 
    const struct textureReference *texref, 
    const void *devPtr, 
    const struct cudaChannelFormatDesc *desc, 
    size_t size
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaBindTexture(offset, texref, devPtr, desc, size);
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaBindTexture(gpu, offset, texref, devPtr, desc, size);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaCreateChannelDesc.
 */
extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, 
    int y, 
    int z, 
    int w, 
    enum cudaChannelFormatKind f
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    struct cudaChannelFormatDesc ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaCreateChannelDesc(x, y, z, w, f);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaGetDeviceProperties.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(
    struct cudaDeviceProp *prop, 
    int device
)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaGetDeviceProperties(prop, device);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaStreamCreate.
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaStreamCreate(pStream);
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaStreamCreate(gpu, pStream);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaMemGetInfo.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaMemGetInfo(free, total);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaSetDevice.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaSetDevice(device);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaSetDeviceFlags.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags )
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaSetDeviceFlags(flags);
    if (gpu->status == MRCUDA_GPU_STATUS_RCUDA)
        mrcuda_record_cudaSetDeviceFlags(gpu, flags);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaGetDevice.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaGetDevice(device);
    mrcuda_function_call_release(gpu);
    return ret;
}

/**
 * Interface of cudaGetDeviceCount.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaGetDeviceCount(count);
    mrcuda_function_call_release(gpu);
    return ret;
}

cudaError_t cudaDeviceSynchronize(void)
{
    MRCUDAGPU_t *gpu = mrcuda_get_current_gpu();
    cudaError_t ret;
    mrcuda_function_call_lock(gpu);
    ret = gpu->defaultHandler->mrcudaDeviceSynchronize();
    mrcuda_function_call_release(gpu);
    return ret;
}

