#include "common.h"
#include "mrcuda.h"
#include "record.h"

static long int __cudaLaunchCount = 0;

/**
 * Interface of __cudaRegisterFatBinary.
 */
void** __cudaRegisterFatBinary(void* fatCubin)
{
    void **ret;
    mrcuda_init();
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->__mrcudaRegisterFatBinary(fatCubin);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaRegisterFatBinary(fatCubin, ret);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of __cudaRegisterFunction.
 */
void __cudaRegisterFunction(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize)
{
    mrcuda_function_call_lock();
    mrcudaSymDefault->__mrcudaRegisterFunction(
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
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaRegisterFunction(
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
    mrcuda_function_call_release();
}

/**
 * Interface of __cudaRegisterVar.
 */
void __cudaRegisterVar(void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global)
{
    mrcuda_function_call_lock();
    mrcudaSymDefault->__mrcudaRegisterVar(
        fatCubinHandle,
        hostVar,
        deviceAddress,
        deviceName,
        ext,
        size,
        constant,
        global
    );
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaRegisterVar(
            fatCubinHandle,
            hostVar,
            deviceAddress,
            deviceName,
            ext,
            size,
            constant,
            global
        );
    mrcuda_function_call_release();
}

/**
 * Interface of __cudaRegisterTexture.
 */
void __cudaRegisterTexture(void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext)
{
    mrcuda_function_call_lock();
    mrcudaSymDefault->__mrcudaRegisterTexture(
        fatCubinHandle,
        hostVar,
        deviceAddress,
        deviceName,
        dim,
        norm,
        ext
    );
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaRegisterTexture(
            fatCubinHandle,
            hostVar,
            deviceAddress,
            deviceName,
            dim,
            norm,
            ext
        );
    mrcuda_function_call_release();
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
    cudaError_t ret;
    ret = mrcudaSymDefault->mrcudaThreadSynchronize();
    return ret;
}

/**
 * Interface of cudaLaunch.
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaLaunch(func);
    mrcuda_function_call_release();
    if(mrcudaState == MRCUDA_STATE_RUNNING_RCUDA)
    {
        __cudaLaunchCount++;
        if(mrcudaNumLaunchSwitchThreashold == __cudaLaunchCount)
            mrcuda_switch();
    }
    return ret;
}

/**
 * Interface of cudaMemcpyToSymbol.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaMemcpyToSymbol(symbol, src, count, offset, kind);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaMemcpyToSymbol(symbol, src, count, offset, kind);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaMemcpy.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaMemcpy(dst, src, count, kind);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaHostAlloc.
 */
extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    cudaError_t ret;
    void *pHost1;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaHostAlloc(pHost, size, flags);
    // This function has to be recorded regardless of rCUDA are being executed or not.
    // This ensures that we calls cudaFreeHost using the right library (rCUDA or native).
    mrcuda_record_cudaHostAlloc(pHost, size, flags);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaMemset.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaMemset(devPtr, value, count);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaFreeHost.
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    // Call the right library of cudaFreeHost according to the recorded cudaHostAlloc calls.
    mrcuda_replay_cudaFreeHost(ptr)->mrcudaFreeHost(ptr);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaSetupArgument.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaSetupArgument(arg, size, offset);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaMalloc.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaMalloc(devPtr, size);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaMalloc(devPtr, size);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaFree.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaFree(devPtr);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaFree(devPtr);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaConfigureCall.
 */
extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaGetLastError.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaGetLastError();
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaBindTexture.
 */
extern __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaBindTexture(offset, texref, devPtr, desc, size);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaBindTexture(offset, texref, devPtr, desc, size);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaCreateChannelDesc.
 */
extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    struct cudaChannelFormatDesc ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaCreateChannelDesc(x, y, z, w, f);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaGetDeviceProperties.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaGetDeviceProperties(prop, device);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaStreamCreate.
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaStreamCreate(pStream);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaStreamCreate(pStream);
    ret = cudaSuccess;
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaMemGetInfo.
 */
extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaMemGetInfo(free, total);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaSetDevice.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaSetDevice(device);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaSetDeviceFlags.
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags )
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaSetDeviceFlags(flags);
    if(mrcudaSymDefault == mrcudaSymRCUDA)
        mrcuda_record_cudaSetDeviceFlags(flags);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaGetDevice.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaGetDevice(device);
    mrcuda_function_call_release();
    return ret;
}

/**
 * Interface of cudaGetDeviceCount.
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaGetDeviceCount(count);
    mrcuda_function_call_release();
    return ret;
}

cudaError_t cudaDeviceSynchronize(void)
{
    cudaError_t ret;
    mrcuda_function_call_lock();
    ret = mrcudaSymDefault->mrcudaDeviceSynchronize();
    mrcuda_function_call_release();
    return ret;
}

