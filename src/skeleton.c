#include "mrcuda.h"

extern cudaError_t cudaDeviceReset(void)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceReset))();
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceSynchronize(void)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceSynchronize))();
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit,  size_t value)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceSetLimit))(limit, value);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceGetLimit(size_t *pValue,  enum cudaLimit limit)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceGetLimit))(pValue, limit);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceGetCacheConfig))(pCacheConfig);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceSetCacheConfig))(cacheConfig);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig)
{
    cudaError_t result;
    
    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceGetSharedMemConfig))(pConfig);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceSetSharedMemConfig))(config);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceGetByPCIBusId(int *device,  char *pciBusId)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceGetByPCIBusId))(device, pciBusId);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId,  int len,  int device)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaDeviceGetPCIBusId))(pciBusId, len, device);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,  cudaEvent_t event)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaIpcGetEventHandle))(handle, event);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,  cudaIpcEventHandle_t handle)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaIpcOpenEventHandle))(event, handle);
    mrcuda_function_call_release();
    
    return result;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,  void *devPtr)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaIpcGetMemHandle))(handle, devPtr);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr,  cudaIpcMemHandle_t handle,  unsigned int flags)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaIpcOpenMemHandle))(devPtr, handle, flags);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaIpcCloseMemHandle))(devPtr);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaThreadExit(void)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaThreadExit))();
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaThreadSynchronize(void)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaThreadSynchronize))();
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit,  size_t value)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaThreadSetLimit))(limit, value);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaThreadGetLimit(size_t *pValue,  enum cudaLimit limit)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaThreadGetLimit))(pValue, limit);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    cudaError_t result;

    mrcuda_function_call_lock();
    result = (*(mrcudaSymDefault->mrcudaThreadGetCacheConfig))(pCacheConfig);
    mrcuda_function_call_release();

    return result;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
}

cudaError_t (*cudaGetLastError)(void);
cudaError_t (*cudaPeekAtLastError)(void);
const char* (*cudaGetErrorString)(cudaError_t error);
cudaError_t (*cudaGetDeviceCount)(int *count);
cudaError_t (*cudaGetDeviceProperties)(struct cudaDeviceProp *prop,  int device);
cudaError_t (*cudaDeviceGetAttribute)(int *value,  enum cudaDeviceAttr attr,  int device);
cudaError_t (*cudaChooseDevice)(int *device,  const struct cudaDeviceProp *prop);
cudaError_t (*cudaSetDevice)(int device);
cudaError_t (*cudaGetDevice)(int *device);
cudaError_t (*cudaSetValidDevices)(int *device_arr,  int len);
cudaError_t (*cudaSetDeviceFlags)( unsigned int flags );
cudaError_t (*cudaStreamCreate)(cudaStream_t *pStream);
cudaError_t (*cudaStreamCreateWithFlags)(cudaStream_t *pStream,  unsigned int flags);
cudaError_t (*cudaStreamDestroy)(cudaStream_t stream);
cudaError_t (*cudaStreamWaitEvent)(cudaStream_t stream,  cudaEvent_t event,  unsigned int flags);
cudaError_t (*cudaStreamAddCallback)(cudaStream_t stream, cudaStreamCallback_t callback,  void *userData,  unsigned int flags);
cudaError_t (*cudaStreamSynchronize)(cudaStream_t stream);
cudaError_t (*cudaStreamQuery)(cudaStream_t stream);
cudaError_t (*cudaEventCreate)(cudaEvent_t *event);
cudaError_t (*cudaEventCreateWithFlags)(cudaEvent_t *event,  unsigned int flags);
cudaError_t (*cudaEventRecord)(cudaEvent_t event,  cudaStream_t stream );
cudaError_t (*cudaEventQuery)(cudaEvent_t event);
cudaError_t (*cudaEventSynchronize)(cudaEvent_t event);
cudaError_t (*cudaEventDestroy)(cudaEvent_t event);
cudaError_t (*cudaEventElapsedTime)(float *ms,  cudaEvent_t start,  cudaEvent_t end);
cudaError_t (*cudaConfigureCall)(dim3 gridDim,  dim3 blockDim,  size_t sharedMem ,  cudaStream_t stream );
cudaError_t (*cudaSetupArgument)(const void *arg,  size_t size,  size_t offset);
cudaError_t (*cudaFuncSetCacheConfig)(const void *func,  enum cudaFuncCache cacheConfig);
cudaError_t (*cudaFuncSetSharedMemConfig)(const void *func,  enum cudaSharedMemConfig config);
cudaError_t (*cudaLaunch)(const void *func);
cudaError_t (*cudaFuncGetAttributes)(struct cudaFuncAttributes *attr,  const void *func);
cudaError_t (*cudaSetDoubleForDevice)(double *d);
cudaError_t (*cudaSetDoubleForHost)(double *d);
cudaError_t (*cudaMalloc)(void **devPtr,  size_t size);
cudaError_t (*cudaMallocHost)(void **ptr,  size_t size);
cudaError_t (*cudaMallocPitch)(void **devPtr,  size_t *pitch,  size_t width,  size_t height);
cudaError_t (*cudaMallocArray)(cudaArray_t *array,  const struct cudaChannelFormatDesc *desc,  size_t width,  size_t height ,  unsigned int flags );
cudaError_t (*cudaFree)(void *devPtr);
cudaError_t (*cudaFreeHost)(void *ptr);
cudaError_t (*cudaFreeArray)(cudaArray_t array);
cudaError_t (*cudaFreeMipmappedArray)(cudaMipmappedArray_t mipmappedArray);
cudaError_t (*cudaHostAlloc)(void **pHost,  size_t size,  unsigned int flags);
cudaError_t (*cudaHostRegister)(void *ptr,  size_t size,  unsigned int flags);
cudaError_t (*cudaHostUnregister)(void *ptr);
cudaError_t (*cudaHostGetDevicePointer)(void **pDevice,  void *pHost,  unsigned int flags);
cudaError_t (*cudaHostGetFlags)(unsigned int *pFlags,  void *pHost);
cudaError_t (*cudaMalloc3D)(struct cudaPitchedPtr* pitchedDevPtr,  struct cudaExtent extent);
cudaError_t (*cudaMalloc3DArray)(cudaArray_t *array,  const struct cudaChannelFormatDesc* desc,  struct cudaExtent extent,  unsigned int flags );
cudaError_t (*cudaMallocMipmappedArray)(cudaMipmappedArray_t *mipmappedArray,  const struct cudaChannelFormatDesc* desc,  struct cudaExtent extent,  unsigned int numLevels,  unsigned int flags );
cudaError_t (*cudaGetMipmappedArrayLevel)(cudaArray_t *levelArray,  cudaMipmappedArray_const_t mipmappedArray,  unsigned int level);
cudaError_t (*cudaMemcpy3D)(const struct cudaMemcpy3DParms *p);
cudaError_t (*cudaMemcpy3DPeer)(const struct cudaMemcpy3DPeerParms *p);
cudaError_t (*cudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p,  cudaStream_t stream );
cudaError_t (*cudaMemcpy3DPeerAsync)(const struct cudaMemcpy3DPeerParms *p,  cudaStream_t stream );
cudaError_t (*cudaMemGetInfo)(size_t *free,  size_t *total);
cudaError_t (*cudaArrayGetInfo)(struct cudaChannelFormatDesc *desc,  struct cudaExtent *extent,  unsigned int *flags,  cudaArray_t array);
cudaError_t (*cudaMemcpy)(void *dst,  const void *src,  size_t count,  enum cudaMemcpyKind kind);
cudaError_t (*cudaMemcpyPeer)(void *dst,  int dstDevice,  const void *src,  int srcDevice,  size_t count);
cudaError_t (*cudaMemcpyToArray)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t count,  enum cudaMemcpyKind kind);
cudaError_t (*cudaMemcpyFromArray)(void *dst,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t count,  enum cudaMemcpyKind kind);
cudaError_t (*cudaMemcpyArrayToArray)(cudaArray_t dst,  size_t wOffsetDst,  size_t hOffsetDst,  cudaArray_const_t src,  size_t wOffsetSrc,  size_t hOffsetSrc,  size_t count,  enum cudaMemcpyKind kind );
cudaError_t (*cudaMemcpy2D)(void *dst,  size_t dpitch,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind);
cudaError_t (*cudaMemcpy2DToArray)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind);
cudaError_t (*cudaMemcpy2DFromArray)(void *dst,  size_t dpitch,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t width,  size_t height,  enum cudaMemcpyKind kind);
cudaError_t (*cudaMemcpy2DArrayToArray)(cudaArray_t dst,  size_t wOffsetDst,  size_t hOffsetDst,  cudaArray_const_t src,  size_t wOffsetSrc,  size_t hOffsetSrc,  size_t width,  size_t height,  enum cudaMemcpyKind kind );
cudaError_t (*cudaMemcpyToSymbol)(const void *symbol,  const void *src,  size_t count,  size_t offset ,  enum cudaMemcpyKind kind );
cudaError_t (*cudaMemcpyFromSymbol)(void *dst,  const void *symbol,  size_t count,  size_t offset ,  enum cudaMemcpyKind kind );
cudaError_t (*cudaMemcpyAsync)(void *dst,  const void *src,  size_t count,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpyPeerAsync)(void *dst,  int dstDevice,  const void *src,  int srcDevice,  size_t count,  cudaStream_t stream );
cudaError_t (*cudaMemcpyToArrayAsync)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t count,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpyFromArrayAsync)(void *dst,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t count,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpy2DAsync)(void *dst,  size_t dpitch,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpy2DToArrayAsync)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpy2DFromArrayAsync)(void *dst,  size_t dpitch,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t width,  size_t height,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpyToSymbolAsync)(const void *symbol,  const void *src,  size_t count,  size_t offset,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemcpyFromSymbolAsync)(void *dst,  const void *symbol,  size_t count,  size_t offset,  enum cudaMemcpyKind kind,  cudaStream_t stream );
cudaError_t (*cudaMemset)(void *devPtr,  int value,  size_t count);
cudaError_t (*cudaMemset2D)(void *devPtr,  size_t pitch,  int value,  size_t width,  size_t height);
cudaError_t (*cudaMemset3D)(struct cudaPitchedPtr pitchedDevPtr,  int value,  struct cudaExtent extent);
cudaError_t (*cudaMemsetAsync)(void *devPtr,  int value,  size_t count,  cudaStream_t stream );
cudaError_t (*cudaMemset2DAsync)(void *devPtr,  size_t pitch,  int value,  size_t width,  size_t height,  cudaStream_t stream );
cudaError_t (*cudaMemset3DAsync)(struct cudaPitchedPtr pitchedDevPtr,  int value,  struct cudaExtent extent,  cudaStream_t stream );
cudaError_t (*cudaGetSymbolAddress)(void **devPtr,  const void *symbol);
cudaError_t (*cudaGetSymbolSize)(size_t *size,  const void *symbol);
cudaError_t (*cudaPointerGetAttributes)(struct cudaPointerAttributes *attributes,  const void *ptr);
cudaError_t (*cudaDeviceCanAccessPeer)(int *canAccessPeer,  int device,  int peerDevice);
cudaError_t (*cudaDeviceEnablePeerAccess)(int peerDevice,  unsigned int flags);
cudaError_t (*cudaDeviceDisablePeerAccess)(int peerDevice);
cudaError_t (*cudaGraphicsUnregisterResource)(cudaGraphicsResource_t resource);
cudaError_t (*cudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource,  unsigned int flags);
cudaError_t (*cudaGraphicsMapResources)(int count,  cudaGraphicsResource_t *resources,  cudaStream_t stream );
cudaError_t (*cudaGraphicsUnmapResources)(int count,  cudaGraphicsResource_t *resources,  cudaStream_t stream );
cudaError_t (*cudaGraphicsResourceGetMappedPointer)(void **devPtr,  size_t *size,  cudaGraphicsResource_t resource);
cudaError_t (*cudaGraphicsSubResourceGetMappedArray)(cudaArray_t *array,  cudaGraphicsResource_t resource,  unsigned int arrayIndex,  unsigned int mipLevel);
cudaError_t (*cudaGraphicsResourceGetMappedMipmappedArray)(cudaMipmappedArray_t *mipmappedArray,  cudaGraphicsResource_t resource);
cudaError_t (*cudaGetChannelDesc)(struct cudaChannelFormatDesc *desc,  cudaArray_const_t array);
struct cudaChannelFormatDesc (*cudaCreateChannelDesc)(int x,  int y,  int z,  int w,  enum cudaChannelFormatKind f);
cudaError_t (*cudaBindTexture)(size_t *offset,  const struct textureReference *texref,  const void *devPtr,  const struct cudaChannelFormatDesc *desc,  size_t size );
cudaError_t (*cudaBindTexture2D)(size_t *offset,  const struct textureReference *texref,  const void *devPtr,  const struct cudaChannelFormatDesc *desc,  size_t width,  size_t height,  size_t pitch);
cudaError_t (*cudaBindTextureToArray)(const struct textureReference *texref,  cudaArray_const_t array,  const struct cudaChannelFormatDesc *desc);
cudaError_t (*cudaBindTextureToMipmappedArray)(const struct textureReference *texref,  cudaMipmappedArray_const_t mipmappedArray,  const struct cudaChannelFormatDesc *desc);
cudaError_t (*cudaUnbindTexture)(const struct textureReference *texref);
cudaError_t (*cudaGetTextureAlignmentOffset)(size_t *offset,  const struct textureReference *texref);
cudaError_t (*cudaGetTextureReference)(const struct textureReference **texref,  const void *symbol);
cudaError_t (*cudaBindSurfaceToArray)(const struct surfaceReference *surfref,  cudaArray_const_t array,  const struct cudaChannelFormatDesc *desc);
cudaError_t (*cudaGetSurfaceReference)(const struct surfaceReference **surfref,  const void *symbol);
cudaError_t (*cudaCreateTextureObject)(cudaTextureObject_t *pTexObject,  const struct cudaResourceDesc *pResDesc,  const struct cudaTextureDesc *pTexDesc,  const struct cudaResourceViewDesc *pResViewDesc);
cudaError_t (*cudaDestroyTextureObject)(cudaTextureObject_t texObject);
cudaError_t (*cudaGetTextureObjectResourceDesc)(struct cudaResourceDesc *pResDesc,  cudaTextureObject_t texObject);
cudaError_t (*cudaGetTextureObjectTextureDesc)(struct cudaTextureDesc *pTexDesc,  cudaTextureObject_t texObject);
cudaError_t (*cudaGetTextureObjectResourceViewDesc)(struct cudaResourceViewDesc *pResViewDesc,  cudaTextureObject_t texObject);
cudaError_t (*cudaCreateSurfaceObject)(cudaSurfaceObject_t *pSurfObject,  const struct cudaResourceDesc *pResDesc);
cudaError_t (*cudaDestroySurfaceObject)(cudaSurfaceObject_t surfObject);
cudaError_t (*cudaGetSurfaceObjectResourceDesc)(struct cudaResourceDesc *pResDesc,  cudaSurfaceObject_t surfObject);
cudaError_t (*cudaDriverGetVersion)(int *driverVersion);
cudaError_t (*cudaRuntimeGetVersion)(int *runtimeVersion);
cudaError_t (*cudaGetExportTable)(const void **ppExportTable,  const cudaUUID_t *pExportTableId);
void** (*__cudaRegisterFatBinary)(void* fatCubin);
void (*__cudaUnregisterFatBinary)(void **fatCubinHandle);
void (*__cudaRegisterVar)(void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global);
void (*__cudaRegisterTexture)(void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext);
void (*__cudaRegisterSurface)(void **fatCubinHandle,const struct surfaceReference  *hostVar,const void **deviceAddress,const char *deviceName,int dim,int ext);
void (*__cudaRegisterFunction)(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize);
void (*__cudaRegisterShared)(void **fatCubinHandle, void **devicePtr);
void (*__cudaRegisterSharedVar)(void **fatCubinHandle,void **devicePtr,size_t size,size_t alignment,int storage);
int (*__cudaSynchronizeThreads)(void** one,void* two);
void (*__cudaTextureFetch)(const void* tex,void* index,int integer,void* val);
void (*__cudaMutexOperation)(int lock);
cudaError_t (*__cudaRegisterDeviceFunction)();
