#ifndef __MRCUDA_DATATYPES__HEADER__
#define __MRCUDA_DATATYPES__HEADER__

#include <cuda_runtime.h>
#include <glib.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <pthread.h>

/* Pre-declared Structs */
typedef struct MHelperProcess_t MHelperProcess_t;
typedef struct MRCUDAGPU_t MRCUDAGPU_t;

/* Struct of CUDA symbolic pointers */
typedef struct MRCUDASym_t
{
    union {
        void *symHandler;
        MHelperProcess_t *processHandler;
    } handler;

    cudaError_t (*mrcudaDeviceReset)(void);
    cudaError_t (*mrcudaDeviceSynchronize)(void);
    cudaError_t (*mrcudaDeviceSetLimit)(enum cudaLimit limit,  size_t value);
    cudaError_t (*mrcudaDeviceGetLimit)(size_t *pValue,  enum cudaLimit limit);
    cudaError_t (*mrcudaDeviceGetCacheConfig)(enum cudaFuncCache *pCacheConfig);
    cudaError_t (*mrcudaDeviceSetCacheConfig)(enum cudaFuncCache cacheConfig);
    cudaError_t (*mrcudaDeviceGetSharedMemConfig)(enum cudaSharedMemConfig *pConfig);
    cudaError_t (*mrcudaDeviceSetSharedMemConfig)(enum cudaSharedMemConfig config);
    cudaError_t (*mrcudaDeviceGetByPCIBusId)(int *device,  char *pciBusId);
    cudaError_t (*mrcudaDeviceGetPCIBusId)(char *pciBusId,  int len,  int device);
    cudaError_t (*mrcudaIpcGetEventHandle)(cudaIpcEventHandle_t *handle,  cudaEvent_t event);
    cudaError_t (*mrcudaIpcOpenEventHandle)(cudaEvent_t *event,  cudaIpcEventHandle_t handle);
    cudaError_t (*mrcudaIpcGetMemHandle)(cudaIpcMemHandle_t *handle,  void *devPtr);
    cudaError_t (*mrcudaIpcOpenMemHandle)(void **devPtr,  cudaIpcMemHandle_t handle,  unsigned int flags);
    cudaError_t (*mrcudaIpcCloseMemHandle)(void *devPtr);
    cudaError_t (*mrcudaThreadExit)(void);
    cudaError_t (*mrcudaThreadSynchronize)(void);
    cudaError_t (*mrcudaThreadSetLimit)(enum cudaLimit limit,  size_t value);
    cudaError_t (*mrcudaThreadGetLimit)(size_t *pValue,  enum cudaLimit limit);
    cudaError_t (*mrcudaThreadGetCacheConfig)(enum cudaFuncCache *pCacheConfig);
    cudaError_t (*mrcudaThreadSetCacheConfig)(enum cudaFuncCache cacheConfig);
    cudaError_t (*mrcudaGetLastError)(void);
    cudaError_t (*mrcudaPeekAtLastError)(void);
    const char* (*mrcudaGetErrorString)(cudaError_t error);
    cudaError_t (*mrcudaGetDeviceCount)(int *count);
    cudaError_t (*mrcudaGetDeviceProperties)(struct cudaDeviceProp *prop,  int device);
    cudaError_t (*mrcudaDeviceGetAttribute)(int *value,  enum cudaDeviceAttr attr,  int device);
    cudaError_t (*mrcudaChooseDevice)(int *device,  const struct cudaDeviceProp *prop);
    cudaError_t (*mrcudaSetDevice)(int device);
    cudaError_t (*mrcudaGetDevice)(int *device);
    cudaError_t (*mrcudaSetValidDevices)(int *device_arr,  int len);
    cudaError_t (*mrcudaSetDeviceFlags)( unsigned int flags );
    cudaError_t (*mrcudaStreamCreate)(cudaStream_t *pStream);
    cudaError_t (*mrcudaStreamCreateWithFlags)(cudaStream_t *pStream,  unsigned int flags);
    cudaError_t (*mrcudaStreamDestroy)(cudaStream_t stream);
    cudaError_t (*mrcudaStreamWaitEvent)(cudaStream_t stream,  cudaEvent_t event,  unsigned int flags);
    cudaError_t (*mrcudaStreamAddCallback)(cudaStream_t stream, cudaStreamCallback_t callback,  void *userData,  unsigned int flags);
    cudaError_t (*mrcudaStreamSynchronize)(cudaStream_t stream);
    cudaError_t (*mrcudaStreamQuery)(cudaStream_t stream);
    cudaError_t (*mrcudaEventCreate)(cudaEvent_t *event);
    cudaError_t (*mrcudaEventCreateWithFlags)(cudaEvent_t *event,  unsigned int flags);
    cudaError_t (*mrcudaEventRecord)(cudaEvent_t event,  cudaStream_t stream );
    cudaError_t (*mrcudaEventQuery)(cudaEvent_t event);
    cudaError_t (*mrcudaEventSynchronize)(cudaEvent_t event);
    cudaError_t (*mrcudaEventDestroy)(cudaEvent_t event);
    cudaError_t (*mrcudaEventElapsedTime)(float *ms,  cudaEvent_t start,  cudaEvent_t end);
    cudaError_t (*mrcudaConfigureCall)(dim3 gridDim,  dim3 blockDim,  size_t sharedMem ,  cudaStream_t stream );
    cudaError_t (*mrcudaSetupArgument)(const void *arg,  size_t size,  size_t offset);
    cudaError_t (*mrcudaFuncSetCacheConfig)(const void *func,  enum cudaFuncCache cacheConfig);
    cudaError_t (*mrcudaFuncSetSharedMemConfig)(const void *func,  enum cudaSharedMemConfig config);
    cudaError_t (*mrcudaLaunch)(const void *func);
    cudaError_t (*mrcudaFuncGetAttributes)(struct cudaFuncAttributes *attr,  const void *func);
    cudaError_t (*mrcudaSetDoubleForDevice)(double *d);
    cudaError_t (*mrcudaSetDoubleForHost)(double *d);
    cudaError_t (*mrcudaMalloc)(void **devPtr,  size_t size);
    cudaError_t (*mrcudaMallocHost)(void **ptr,  size_t size);
    cudaError_t (*mrcudaMallocPitch)(void **devPtr,  size_t *pitch,  size_t width,  size_t height);
    cudaError_t (*mrcudaMallocArray)(cudaArray_t *array,  const struct cudaChannelFormatDesc *desc,  size_t width,  size_t height ,  unsigned int flags );
    cudaError_t (*mrcudaFree)(void *devPtr);
    cudaError_t (*mrcudaFreeHost)(void *ptr);
    cudaError_t (*mrcudaFreeArray)(cudaArray_t array);
    cudaError_t (*mrcudaFreeMipmappedArray)(cudaMipmappedArray_t mipmappedArray);
    cudaError_t (*mrcudaHostAlloc)(void **pHost,  size_t size,  unsigned int flags);
    cudaError_t (*mrcudaHostRegister)(void *ptr,  size_t size,  unsigned int flags);
    cudaError_t (*mrcudaHostUnregister)(void *ptr);
    cudaError_t (*mrcudaHostGetDevicePointer)(void **pDevice,  void *pHost,  unsigned int flags);
    cudaError_t (*mrcudaHostGetFlags)(unsigned int *pFlags,  void *pHost);
    cudaError_t (*mrcudaMalloc3D)(struct cudaPitchedPtr* pitchedDevPtr,  struct cudaExtent extent);
    cudaError_t (*mrcudaMalloc3DArray)(cudaArray_t *array,  const struct cudaChannelFormatDesc* desc,  struct cudaExtent extent,  unsigned int flags );
    cudaError_t (*mrcudaMallocMipmappedArray)(cudaMipmappedArray_t *mipmappedArray,  const struct cudaChannelFormatDesc* desc,  struct cudaExtent extent,  unsigned int numLevels,  unsigned int flags );
    cudaError_t (*mrcudaGetMipmappedArrayLevel)(cudaArray_t *levelArray,  cudaMipmappedArray_const_t mipmappedArray,  unsigned int level);
    cudaError_t (*mrcudaMemcpy3D)(const struct cudaMemcpy3DParms *p);
    cudaError_t (*mrcudaMemcpy3DPeer)(const struct cudaMemcpy3DPeerParms *p);
    cudaError_t (*mrcudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpy3DPeerAsync)(const struct cudaMemcpy3DPeerParms *p,  cudaStream_t stream );
    cudaError_t (*mrcudaMemGetInfo)(size_t *free,  size_t *total);
    cudaError_t (*mrcudaArrayGetInfo)(struct cudaChannelFormatDesc *desc,  struct cudaExtent *extent,  unsigned int *flags,  cudaArray_t array);
    cudaError_t (*mrcudaMemcpy)(void *dst,  const void *src,  size_t count,  enum cudaMemcpyKind kind);
    cudaError_t (*mrcudaMemcpyPeer)(void *dst,  int dstDevice,  const void *src,  int srcDevice,  size_t count);
    cudaError_t (*mrcudaMemcpyToArray)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t count,  enum cudaMemcpyKind kind);
    cudaError_t (*mrcudaMemcpyFromArray)(void *dst,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t count,  enum cudaMemcpyKind kind);
    cudaError_t (*mrcudaMemcpyArrayToArray)(cudaArray_t dst,  size_t wOffsetDst,  size_t hOffsetDst,  cudaArray_const_t src,  size_t wOffsetSrc,  size_t hOffsetSrc,  size_t count,  enum cudaMemcpyKind kind );
    cudaError_t (*mrcudaMemcpy2D)(void *dst,  size_t dpitch,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind);
    cudaError_t (*mrcudaMemcpy2DToArray)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind);
    cudaError_t (*mrcudaMemcpy2DFromArray)(void *dst,  size_t dpitch,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t width,  size_t height,  enum cudaMemcpyKind kind);
    cudaError_t (*mrcudaMemcpy2DArrayToArray)(cudaArray_t dst,  size_t wOffsetDst,  size_t hOffsetDst,  cudaArray_const_t src,  size_t wOffsetSrc,  size_t hOffsetSrc,  size_t width,  size_t height,  enum cudaMemcpyKind kind );
    cudaError_t (*mrcudaMemcpyToSymbol)(const void *symbol,  const void *src,  size_t count,  size_t offset ,  enum cudaMemcpyKind kind );
    cudaError_t (*mrcudaMemcpyFromSymbol)(void *dst,  const void *symbol,  size_t count,  size_t offset ,  enum cudaMemcpyKind kind );
    cudaError_t (*mrcudaMemcpyAsync)(void *dst,  const void *src,  size_t count,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpyPeerAsync)(void *dst,  int dstDevice,  const void *src,  int srcDevice,  size_t count,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpyToArrayAsync)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t count,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpyFromArrayAsync)(void *dst,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t count,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpy2DAsync)(void *dst,  size_t dpitch,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpy2DToArrayAsync)(cudaArray_t dst,  size_t wOffset,  size_t hOffset,  const void *src,  size_t spitch,  size_t width,  size_t height,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpy2DFromArrayAsync)(void *dst,  size_t dpitch,  cudaArray_const_t src,  size_t wOffset,  size_t hOffset,  size_t width,  size_t height,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpyToSymbolAsync)(const void *symbol,  const void *src,  size_t count,  size_t offset,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemcpyFromSymbolAsync)(void *dst,  const void *symbol,  size_t count,  size_t offset,  enum cudaMemcpyKind kind,  cudaStream_t stream );
    cudaError_t (*mrcudaMemset)(void *devPtr,  int value,  size_t count);
    cudaError_t (*mrcudaMemset2D)(void *devPtr,  size_t pitch,  int value,  size_t width,  size_t height);
    cudaError_t (*mrcudaMemset3D)(struct cudaPitchedPtr pitchedDevPtr,  int value,  struct cudaExtent extent);
    cudaError_t (*mrcudaMemsetAsync)(void *devPtr,  int value,  size_t count,  cudaStream_t stream );
    cudaError_t (*mrcudaMemset2DAsync)(void *devPtr,  size_t pitch,  int value,  size_t width,  size_t height,  cudaStream_t stream );
    cudaError_t (*mrcudaMemset3DAsync)(struct cudaPitchedPtr pitchedDevPtr,  int value,  struct cudaExtent extent,  cudaStream_t stream );
    cudaError_t (*mrcudaGetSymbolAddress)(void **devPtr,  const void *symbol);
    cudaError_t (*mrcudaGetSymbolSize)(size_t *size,  const void *symbol);
    cudaError_t (*mrcudaPointerGetAttributes)(struct cudaPointerAttributes *attributes,  const void *ptr);
    cudaError_t (*mrcudaDeviceCanAccessPeer)(int *canAccessPeer,  int device,  int peerDevice);
    cudaError_t (*mrcudaDeviceEnablePeerAccess)(int peerDevice,  unsigned int flags);
    cudaError_t (*mrcudaDeviceDisablePeerAccess)(int peerDevice);
    cudaError_t (*mrcudaGraphicsUnregisterResource)(cudaGraphicsResource_t resource);
    cudaError_t (*mrcudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource,  unsigned int flags);
    cudaError_t (*mrcudaGraphicsMapResources)(int count,  cudaGraphicsResource_t *resources,  cudaStream_t stream );
    cudaError_t (*mrcudaGraphicsUnmapResources)(int count,  cudaGraphicsResource_t *resources,  cudaStream_t stream );
    cudaError_t (*mrcudaGraphicsResourceGetMappedPointer)(void **devPtr,  size_t *size,  cudaGraphicsResource_t resource);
    cudaError_t (*mrcudaGraphicsSubResourceGetMappedArray)(cudaArray_t *array,  cudaGraphicsResource_t resource,  unsigned int arrayIndex,  unsigned int mipLevel);
    cudaError_t (*mrcudaGraphicsResourceGetMappedMipmappedArray)(cudaMipmappedArray_t *mipmappedArray,  cudaGraphicsResource_t resource);
    cudaError_t (*mrcudaGetChannelDesc)(struct cudaChannelFormatDesc *desc,  cudaArray_const_t array);
    struct cudaChannelFormatDesc (*mrcudaCreateChannelDesc)(int x,  int y,  int z,  int w,  enum cudaChannelFormatKind f);
    cudaError_t (*mrcudaBindTexture)(size_t *offset,  const struct textureReference *texref,  const void *devPtr,  const struct cudaChannelFormatDesc *desc,  size_t size );
    cudaError_t (*mrcudaBindTexture2D)(size_t *offset,  const struct textureReference *texref,  const void *devPtr,  const struct cudaChannelFormatDesc *desc,  size_t width,  size_t height,  size_t pitch);
    cudaError_t (*mrcudaBindTextureToArray)(const struct textureReference *texref,  cudaArray_const_t array,  const struct cudaChannelFormatDesc *desc);
    cudaError_t (*mrcudaBindTextureToMipmappedArray)(const struct textureReference *texref,  cudaMipmappedArray_const_t mipmappedArray,  const struct cudaChannelFormatDesc *desc);
    cudaError_t (*mrcudaUnbindTexture)(const struct textureReference *texref);
    cudaError_t (*mrcudaGetTextureAlignmentOffset)(size_t *offset,  const struct textureReference *texref);
    cudaError_t (*mrcudaGetTextureReference)(const struct textureReference **texref,  const void *symbol);
    cudaError_t (*mrcudaBindSurfaceToArray)(const struct surfaceReference *surfref,  cudaArray_const_t array,  const struct cudaChannelFormatDesc *desc);
    cudaError_t (*mrcudaGetSurfaceReference)(const struct surfaceReference **surfref,  const void *symbol);
    cudaError_t (*mrcudaCreateTextureObject)(cudaTextureObject_t *pTexObject,  const struct cudaResourceDesc *pResDesc,  const struct cudaTextureDesc *pTexDesc,  const struct cudaResourceViewDesc *pResViewDesc);
    cudaError_t (*mrcudaDestroyTextureObject)(cudaTextureObject_t texObject);
    cudaError_t (*mrcudaGetTextureObjectResourceDesc)(struct cudaResourceDesc *pResDesc,  cudaTextureObject_t texObject);
    cudaError_t (*mrcudaGetTextureObjectTextureDesc)(struct cudaTextureDesc *pTexDesc,  cudaTextureObject_t texObject);
    cudaError_t (*mrcudaGetTextureObjectResourceViewDesc)(struct cudaResourceViewDesc *pResViewDesc,  cudaTextureObject_t texObject);
    cudaError_t (*mrcudaCreateSurfaceObject)(cudaSurfaceObject_t *pSurfObject,  const struct cudaResourceDesc *pResDesc);
    cudaError_t (*mrcudaDestroySurfaceObject)(cudaSurfaceObject_t surfObject);
    cudaError_t (*mrcudaGetSurfaceObjectResourceDesc)(struct cudaResourceDesc *pResDesc,  cudaSurfaceObject_t surfObject);
    cudaError_t (*mrcudaDriverGetVersion)(int *driverVersion);
    cudaError_t (*mrcudaRuntimeGetVersion)(int *runtimeVersion);
    cudaError_t (*mrcudaGetExportTable)(const void **ppExportTable,  const cudaUUID_t *pExportTableId);
    void** (*__mrcudaRegisterFatBinary)(void* fatCubin);
    void (*__mrcudaUnregisterFatBinary)(void **fatCubinHandle);
    void (*__mrcudaRegisterVar)(void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global);
    void (*__mrcudaRegisterTexture)(void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext);
    void (*__mrcudaRegisterSurface)(void **fatCubinHandle,const struct surfaceReference  *hostVar,const void **deviceAddress,const char *deviceName,int dim,int ext);
    void (*__mrcudaRegisterFunction)(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize);
    void (*__mrcudaRegisterShared)(void **fatCubinHandle, void **devicePtr);
    void (*__mrcudaRegisterSharedVar)(void **fatCubinHandle,void **devicePtr,size_t size,size_t alignment,int storage);
    int (*__mrcudaSynchronizeThreads)(void** one,void* two);
    void (*__mrcudaTextureFetch)(const void* tex,void* index,int integer,void* val);
    void (*__mrcudaMutexOperation)(int lock);
    cudaError_t (*__mrcudaRegisterDeviceFunction)();
} MRCUDASym_t;

/* Shared-memory-related Structure */
typedef struct MRCUDASharedMem_t {
    key_t key;  /* shared-memory key */
    size_t size;    /* size of the shared memory region */
} MRCUDASharedMem_t;

typedef struct MRCUDASharedMemLocalInfo_t {
    MRCUDASharedMem_t sharedMem;
    int shmid;
    void *startAddr;
} MRCUDASharedMemLocalInfo_t;

/* Group of CUDA-related call parameter struct */
typedef struct cudaRegisterFatBinary_t {
    void *fatCubin;
    void **fatCubinHandle;
    MRCUDASharedMem_t sharedMem;
} cudaRegisterFatBinary_t;

typedef struct cudaRegisterFunction_t {
    void **fatCubinHandle;
    union {
        const char *ptr;
        size_t offset;    /* relative to the start of the specified shared memory. */
    } hostFun;
    union {
        char *ptr;
        size_t offset;    /* relative to the start of the specified shared memory. */
    } deviceFun;
    union {
        const char *ptr;
        size_t offset;    /* relative to the start of the specified shared memory. */
    } deviceName;
    int thread_limit;
    uint3* tid;
    uint3* bid;
    dim3* bDim;
    dim3* gDim;
    int* wSize;
    MRCUDASharedMem_t shminfo;
    /** 
     * pointer to cudaRegisterFatBinary_t.fatCubinHandle
     * However, we cannot use it in IPC.
     */
    void ***fatCubinHandlePtr;  
} cudaRegisterFunction_t;

typedef struct cudaRegisterVar_t {
    void **fatCubinHandle;
    union {
        char *ptr;
        size_t offset;
    } hostVar;
    union {
        char *ptr;
        size_t offset;
    } deviceAddress;
    union {
        const char *ptr;
        size_t offset;
    } deviceName;
    int ext;
    int size;
    int constant;
    int global;
    MRCUDASharedMem_t shminfo;
    /** 
     * pointer to cudaRegisterFatBinary_t.fatCubinHandle
     * However, we cannot use it in IPC.
     */
    void ***fatCubinHandlePtr;  
} cudaRegisterVar_t;

typedef struct cudaRegisterTexture_t {
    void **fatCubinHandle;
    union {
        const struct textureReference *ptr;
        size_t offset;
    } hostVar;
    const void **deviceAddress;
    union {
        const char *ptr;
        size_t offset;
    } deviceName;
    int dim;
    int norm;
    int ext;
    /** 
     * pointer to cudaRegisterFatBinary_t.fatCubinHandle
     * However, we cannot use it in IPC.
     */
    void ***fatCubinHandlePtr;  
} cudaRegisterTexture_t;

typedef struct cudaUnregisterFatBinary_t {
    void **fatCubinHandle;
    /** 
     * pointer to cudaRegisterFatBinary_t.fatCubinHandle
     * However, we cannot use it in IPC.
     */
    void ***fatCubinHandlePtr;
} cudaUnregisterFatBinary_t;

typedef struct cudaMalloc_t {
    void *devPtr;
    size_t size;
} cudaMalloc_t;

typedef struct cudaFree_t {
    void *devPtr;
} cudaFree_t;

typedef struct cudaBindTexture_t {
    size_t offset;
    const struct textureReference *texref;
    const void *devPtr;
    struct cudaChannelFormatDesc desc;
    size_t size;
} cudaBindTexture_t;

typedef struct cudaStreamCreate_t {
    cudaStream_t *pStream;
} cudaStreamCreate_t;

typedef struct cudaSetDeviceFlags_t {
    unsigned int flags;
} cudaSetDeviceFlags_t;

typedef struct cudaMemcpyToSymbol_t {
    size_t dataSize;
    enum cudaMemcpyKind kind;
    MRCUDASharedMem_t sharedMem;
} cudaMemcpyToSymbol_t;

typedef struct cudaMemcpy_t {
    void *dst;
    size_t size;
    enum cudaMemcpyKind kind;
    MRCUDASharedMem_t sharedMem;
} cudaMemcpy_t;

/* MRecord Struct */
typedef struct MRecord_t {
    char *functionName;
    int skipMockStream;
    union {
        cudaRegisterFatBinary_t cudaRegisterFatBinary;
        cudaRegisterFunction_t cudaRegisterFunction;
        cudaRegisterVar_t cudaRegisterVar;
        cudaRegisterTexture_t cudaRegisterTexture;
        cudaUnregisterFatBinary_t cudaUnregisterFatBinary;
        cudaMalloc_t cudaMalloc;
        cudaFree_t cudaFree;
        cudaBindTexture_t cudaBindTexture;
        cudaStreamCreate_t cudaStreamCreate;
        cudaSetDeviceFlags_t cudaSetDeviceFlags;
    } data;
    void (*replayFunc)(MRCUDAGPU_t *, struct MRecord_t*);
    struct MRecord_t *next;
} MRecord_t;

/* Communication-related Structs */
typedef enum MHelperCommandType_e {
    MRCOMMAND_TYPE_CUDAREGISTERFATBINARY = 0,
    MRCOMMAND_TYPE_CUDAREGISTERFUNCTION,
    MRCOMMAND_TYPE_CUDAREGISTERVAR,
    MRCOMMAND_TYPE_CUDAREGISTERTEXTURE,
    MRCOMMAND_TYPE_CUDAUNREGISTERFATBINARY,
    MRCOMMAND_TYPE_CUDAMALLOC,
    MRCOMMAND_TYPE_CUDAFREE,
    MRCOMMAND_TYPE_CUDABINDTEXTURE,
    MRCOMMAND_TYPE_CUDASTREAMCREATE,
    MRCOMMAND_TYPE_CUDASETDEVICEFLAGS,
    MRCOMMAND_TYPE_CUDAMEMCPYTOSYMBOL,
    MRCOMMAND_TYPE_CUDAMEMCPY,
    MRCOMMAND_TYPE_CUCTXCREATE
} MHelperCommandType_e;

typedef struct MHelperCommand_t {
    int id;
    MHelperCommandType_e type;
    union {
        cudaRegisterFatBinary_t cudaRegisterFatBinary;
        cudaRegisterFunction_t cudaRegisterFunction;
        cudaRegisterVar_t cudaRegisterVar;
        cudaRegisterTexture_t cudaRegisterTexture;
        cudaUnregisterFatBinary_t cudaUnregisterFatBinary;
        cudaMalloc_t cudaMalloc;
        cudaFree_t cudaFree;
        cudaBindTexture_t cudaBindTexture;
        cudaStreamCreate_t cudaStreamCreate;
        cudaSetDeviceFlags_t cudaSetDeviceFlags;
        cudaMemcpyToSymbol_t cudaMemcpyToSymbol;
        cudaMemcpy_t cudaMemcpy;
    } command;
} MHelperCommand_t;

typedef struct MHelperResult_t {
    int id;
    MHelperCommandType_e type;
    int internalError;
    cudaError_t cudaError;
    union {
        cudaRegisterFatBinary_t cudaRegisterFatBinary;
    } result;
} MHelperResult_t;

struct MHelperProcess_t {
    pid_t pid;
    int readPipe;
    int writePipe;
    MRCUDASym_t *handle;
};

/* MRecordGPU Struct */
typedef struct MRecordGPU_t {
    MRCUDAGPU_t *mrcudaGPU;
    MRecord_t *mrcudaRecordHeadPtr;
    MRecord_t *mrcudaRecordTailPtr;
    GHashTable *activeMemoryTable;
    GHashTable *activeSymbolTable;
    GHashTable *fatCubinHandleAddrTable;
    GHashTable *hostAllocTable;
} MRecordGPU_t;

/* MRCUDAGPU Struct */
typedef enum MRCUDAGPUStatus_e {
    MRCUDA_GPU_STATUS_RCUDA = 0,
    MRCUDA_GPU_STATUS_NATIVE,
    MRCUDA_GPU_STATUS_HELPER
} MRCUDAGPUStatus_e;

struct MRCUDAGPU_t {
    int virtualNumber;
    int realNumber;
    int nativeFromStart;
    int switchThreshold;
    int cudaLaunchCount;
    MRCUDAGPUStatus_e status;
    pthread_mutex_t mutex;
    MRCUDASym_t *defaultHandler;
    MRecordGPU_t *mrecordGPU;
    MHelperProcess_t *mhelperProcess;
};

typedef enum MRCUDAState_e {
    MRCUDA_STATE_UNINITIALIZED = 0,
    MRCUDA_STATE_RUNNING,
    MRCUDA_STATE_FINALIZED
} MRCUDAState_e;

#endif  /* __MRCUDA_DATATYPES__HEADER__ */

