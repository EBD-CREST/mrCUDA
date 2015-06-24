#define _GNU_SOURCE

#include "common.h"
#include "mrcuda.h"
#include "record.h"
#include "comm.h"
#include "datatypes.h"
#include "intercomm.h"

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>
#include <glib.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

// For manual profiling
#include <sys/time.h>

#define __RCUDA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_RCUDA_LIB_PATH"
#define __NVIDIA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_NVIDIA_LIB_PATH"
#define __SOCK_PATH_ENV_NAME__ "MRCUDA_SOCK_PATH"
#define __MHELPER_PATH_ENV_NAME__ "MHELPER_PATH"
#define __MRCUDA_SWITCH_THRESHOLD_HEADER_ENV_NAME__ "MRCUDA_SWITCH_THRESHOLD_"
#define __RCUDA_DEVICE_HEADER_ENV_NAME__    "RCUDA_DEVICE_"
#define __RCUDA_DEVICE_COUNT_ENV_NAME__ "RCUDA_DEVICE_COUNT"
#define __LOCALHOST__ "localhost"

MRCUDAState_e mrcudaState;

MRCUDASym_t *mrcudaSymNvidia;
MRCUDASym_t *mrcudaSymRCUDA;

int mrcudaNumGPUs = 0;
MRCUDAGPU_t *mrcudaGPUList;

static char *__rCUDALibPath;
static char *__nvidiaLibPath;
static char *__helperPath;

char *__sockPath;

static pthread_mutex_t __switch_mutex;
static int __hasNativeGPU = 0;

GHashTable *mrcudaGPUThreadMap = NULL;

/**
 * Initialize mrcudaGPUThreadMap.
 * @return 0 if success, other number otherwise.
 */
static inline int __init_mrcuda_gpu_thread_map()
{
    if (mrcudaGPUThreadMap == NULL)
        mrcudaGPUThreadMap = g_hash_table_new(g_direct_hash, g_direct_equal);
    return mrcudaGPUThreadMap == NULL ? -1 : 0;
}

/**
 * Get the ID of the calling thread.
 * @return thread ID.
 */
static inline pid_t gettid()
{
    return (pid_t)syscall(SYS_gettid);
}

/**
 * Get the GPU assigned to the calling thread.
 * @return a pointer to the assigned GPU.
 */
MRCUDAGPU_t *mrcuda_get_current_gpu()
{
    pid_t tid;
    MRCUDAGPU_t *mrcudaGPU;
    __init_mrcuda_gpu_thread_map();
    tid = gettid();
    if ((mrcudaGPU = g_hash_table_lookup(mrcudaGPUThreadMap, GINT_TO_POINTER(tid))) == NULL) {
        mrcudaGPU = mrcudaGPUList;
        g_hash_table_insert(mrcudaGPUThreadMap, GINT_TO_POINTER(tid), mrcudaGPU);
    }
    return mrcudaGPU;
}

/**
 * Set the GPU assigned to the calling thread.
 * @param device virtual device ID.
 */
void mrcuda_set_current_gpu(int device)
{
    pid_t tid;
    mrcuda_get_current_gpu();
    tid = gettid();
    g_hash_table_replace(mrcudaGPUThreadMap, GINT_TO_POINTER(tid), &(mrcudaGPUList[device]));
}


/**
 * Try to link the specified symbol to the handle.
 * Print error and terminate the program if an error occurs.
 * @param handle a handle returned from dlopen.
 * @param symbol a symbol to be linked to the handle.
 * @return a pointer to the linked symbol.
 */
static inline void *__safe_dlsym(void *handle, const char *symbol)
{
    char *error;
    void *ret_handle = dlsym(handle, symbol);
    
    if((error = dlerror()) != NULL)
        REPORT_ERROR_AND_EXIT("%s\n", error);

    return ret_handle;
}

/**
 * Symlink functions of the handle.
 * @param mrcudaSym handle to be sym-linked.
 */
static void __symlink_handle(MRCUDASym_t *mrcudaSym)
{
    mrcudaSym->mrcudaDeviceReset = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceReset");
    mrcudaSym->mrcudaDeviceSynchronize = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceSynchronize");
    mrcudaSym->mrcudaDeviceSetLimit = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceSetLimit");
    mrcudaSym->mrcudaDeviceGetLimit = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceGetLimit");
    mrcudaSym->mrcudaDeviceGetCacheConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceGetCacheConfig");
    mrcudaSym->mrcudaDeviceSetCacheConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceSetCacheConfig");
    mrcudaSym->mrcudaDeviceGetSharedMemConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceGetSharedMemConfig");
    mrcudaSym->mrcudaDeviceSetSharedMemConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceSetSharedMemConfig");
    mrcudaSym->mrcudaDeviceGetByPCIBusId = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceGetByPCIBusId");
    mrcudaSym->mrcudaDeviceGetPCIBusId = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceGetPCIBusId");
    mrcudaSym->mrcudaIpcGetEventHandle = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaIpcGetEventHandle");
    mrcudaSym->mrcudaIpcOpenEventHandle = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaIpcOpenEventHandle");
    mrcudaSym->mrcudaIpcGetMemHandle = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaIpcGetMemHandle");
    mrcudaSym->mrcudaIpcOpenMemHandle = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaIpcOpenMemHandle");
    mrcudaSym->mrcudaIpcCloseMemHandle = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaIpcCloseMemHandle");
    mrcudaSym->mrcudaThreadExit = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaThreadExit");
    mrcudaSym->mrcudaThreadSynchronize = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaThreadSynchronize");
    mrcudaSym->mrcudaThreadSetLimit = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaThreadSetLimit");
    mrcudaSym->mrcudaThreadGetLimit = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaThreadGetLimit");
    mrcudaSym->mrcudaThreadGetCacheConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaThreadGetCacheConfig");
    mrcudaSym->mrcudaThreadSetCacheConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaThreadSetCacheConfig");
    mrcudaSym->mrcudaGetLastError = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetLastError");
    mrcudaSym->mrcudaPeekAtLastError = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaPeekAtLastError");
    mrcudaSym->mrcudaGetErrorString = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetErrorString");
    mrcudaSym->mrcudaGetDeviceCount = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetDeviceCount");
    mrcudaSym->mrcudaGetDeviceProperties = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetDeviceProperties");
    mrcudaSym->mrcudaDeviceGetAttribute = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceGetAttribute");
    mrcudaSym->mrcudaChooseDevice = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaChooseDevice");
    mrcudaSym->mrcudaSetDevice = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaSetDevice");
    mrcudaSym->mrcudaGetDevice = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetDevice");
    mrcudaSym->mrcudaSetValidDevices = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaSetValidDevices");
    mrcudaSym->mrcudaSetDeviceFlags = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaSetDeviceFlags");
    mrcudaSym->mrcudaStreamCreate = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamCreate");
    mrcudaSym->mrcudaStreamCreateWithFlags = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamCreateWithFlags");
    mrcudaSym->mrcudaStreamDestroy = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamDestroy");
    mrcudaSym->mrcudaStreamWaitEvent = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamWaitEvent");
    mrcudaSym->mrcudaStreamAddCallback = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamAddCallback");
    mrcudaSym->mrcudaStreamSynchronize = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamSynchronize");
    mrcudaSym->mrcudaStreamQuery = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaStreamQuery");
    mrcudaSym->mrcudaEventCreate = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventCreate");
    mrcudaSym->mrcudaEventCreateWithFlags = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventCreateWithFlags");
    mrcudaSym->mrcudaEventRecord = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventRecord");
    mrcudaSym->mrcudaEventQuery = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventQuery");
    mrcudaSym->mrcudaEventSynchronize = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventSynchronize");
    mrcudaSym->mrcudaEventDestroy = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventDestroy");
    mrcudaSym->mrcudaEventElapsedTime = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaEventElapsedTime");
    mrcudaSym->mrcudaConfigureCall = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaConfigureCall");
    mrcudaSym->mrcudaSetupArgument = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaSetupArgument");
    mrcudaSym->mrcudaFuncSetCacheConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFuncSetCacheConfig");
    mrcudaSym->mrcudaFuncSetSharedMemConfig = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFuncSetSharedMemConfig");
    mrcudaSym->mrcudaLaunch = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaLaunch");
    mrcudaSym->mrcudaFuncGetAttributes = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFuncGetAttributes");
    mrcudaSym->mrcudaSetDoubleForDevice = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaSetDoubleForDevice");
    mrcudaSym->mrcudaSetDoubleForHost = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaSetDoubleForHost");
    mrcudaSym->mrcudaMalloc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMalloc");
    mrcudaSym->mrcudaMallocHost = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMallocHost");
    mrcudaSym->mrcudaMallocPitch = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMallocPitch");
    mrcudaSym->mrcudaMallocArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMallocArray");
    mrcudaSym->mrcudaFree = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFree");
    mrcudaSym->mrcudaFreeHost = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFreeHost");
    mrcudaSym->mrcudaFreeArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFreeArray");
    mrcudaSym->mrcudaFreeMipmappedArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaFreeMipmappedArray");
    mrcudaSym->mrcudaHostAlloc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaHostAlloc");
    mrcudaSym->mrcudaHostRegister = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaHostRegister");
    mrcudaSym->mrcudaHostUnregister = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaHostUnregister");
    mrcudaSym->mrcudaHostGetDevicePointer = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaHostGetDevicePointer");
    mrcudaSym->mrcudaHostGetFlags = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaHostGetFlags");
    mrcudaSym->mrcudaMalloc3D = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMalloc3D");
    mrcudaSym->mrcudaMalloc3DArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMalloc3DArray");
    mrcudaSym->mrcudaMallocMipmappedArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMallocMipmappedArray");
    mrcudaSym->mrcudaGetMipmappedArrayLevel = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetMipmappedArrayLevel");
    mrcudaSym->mrcudaMemcpy3D = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy3D");
    mrcudaSym->mrcudaMemcpy3DPeer = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy3DPeer");
    mrcudaSym->mrcudaMemcpy3DAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy3DAsync");
    mrcudaSym->mrcudaMemcpy3DPeerAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy3DPeerAsync");
    mrcudaSym->mrcudaMemGetInfo = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemGetInfo");
    mrcudaSym->mrcudaArrayGetInfo = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaArrayGetInfo");
    mrcudaSym->mrcudaMemcpy = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy");
    mrcudaSym->mrcudaMemcpyPeer = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyPeer");
    mrcudaSym->mrcudaMemcpyToArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyToArray");
    mrcudaSym->mrcudaMemcpyFromArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyFromArray");
    mrcudaSym->mrcudaMemcpyArrayToArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyArrayToArray");
    mrcudaSym->mrcudaMemcpy2D = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2D");
    mrcudaSym->mrcudaMemcpy2DToArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2DToArray");
    mrcudaSym->mrcudaMemcpy2DFromArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2DFromArray");
    mrcudaSym->mrcudaMemcpy2DArrayToArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2DArrayToArray");
    mrcudaSym->mrcudaMemcpyToSymbol = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyToSymbol");
    mrcudaSym->mrcudaMemcpyFromSymbol = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyFromSymbol");
    mrcudaSym->mrcudaMemcpyAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyAsync");
    mrcudaSym->mrcudaMemcpyPeerAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyPeerAsync");
    mrcudaSym->mrcudaMemcpyToArrayAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyToArrayAsync");
    mrcudaSym->mrcudaMemcpyFromArrayAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyFromArrayAsync");
    mrcudaSym->mrcudaMemcpy2DAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2DAsync");
    mrcudaSym->mrcudaMemcpy2DToArrayAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2DToArrayAsync");
    mrcudaSym->mrcudaMemcpy2DFromArrayAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpy2DFromArrayAsync");
    mrcudaSym->mrcudaMemcpyToSymbolAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyToSymbolAsync");
    mrcudaSym->mrcudaMemcpyFromSymbolAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemcpyFromSymbolAsync");
    mrcudaSym->mrcudaMemset = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemset");
    mrcudaSym->mrcudaMemset2D = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemset2D");
    mrcudaSym->mrcudaMemset3D = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemset3D");
    mrcudaSym->mrcudaMemsetAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemsetAsync");
    mrcudaSym->mrcudaMemset2DAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemset2DAsync");
    mrcudaSym->mrcudaMemset3DAsync = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaMemset3DAsync");
    mrcudaSym->mrcudaGetSymbolAddress = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetSymbolAddress");
    mrcudaSym->mrcudaGetSymbolSize = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetSymbolSize");
    mrcudaSym->mrcudaPointerGetAttributes = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaPointerGetAttributes");
    mrcudaSym->mrcudaDeviceCanAccessPeer = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceCanAccessPeer");
    mrcudaSym->mrcudaDeviceEnablePeerAccess = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceEnablePeerAccess");
    mrcudaSym->mrcudaDeviceDisablePeerAccess = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDeviceDisablePeerAccess");
    /*mrcudaSym->mrcudaGraphicsUnregisterResource = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsUnregisterResource");
    mrcudaSym->mrcudaGraphicsResourceSetMapFlags = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsResourceSetMapFlags");
    mrcudaSym->mrcudaGraphicsMapResources = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsMapResources");
    mrcudaSym->mrcudaGraphicsUnmapResources = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsUnmapResources");
    mrcudaSym->mrcudaGraphicsResourceGetMappedPointer = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsResourceGetMappedPointer");
    mrcudaSym->mrcudaGraphicsSubResourceGetMappedArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsSubResourceGetMappedArray");
    mrcudaSym->mrcudaGraphicsResourceGetMappedMipmappedArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGraphicsResourceGetMappedMipmappedArray");*/
    mrcudaSym->mrcudaGetChannelDesc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetChannelDesc");
    mrcudaSym->mrcudaCreateChannelDesc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaCreateChannelDesc");
    mrcudaSym->mrcudaBindTexture = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaBindTexture");
    mrcudaSym->mrcudaBindTexture2D = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaBindTexture2D");
    mrcudaSym->mrcudaBindTextureToArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaBindTextureToArray");
    mrcudaSym->mrcudaBindTextureToMipmappedArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaBindTextureToMipmappedArray");
    mrcudaSym->mrcudaUnbindTexture = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaUnbindTexture");
    mrcudaSym->mrcudaGetTextureAlignmentOffset = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetTextureAlignmentOffset");
    mrcudaSym->mrcudaGetTextureReference = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetTextureReference");
    mrcudaSym->mrcudaBindSurfaceToArray = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaBindSurfaceToArray");
    mrcudaSym->mrcudaGetSurfaceReference = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetSurfaceReference");
    mrcudaSym->mrcudaCreateTextureObject = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaCreateTextureObject");
    mrcudaSym->mrcudaDestroyTextureObject = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDestroyTextureObject");
    mrcudaSym->mrcudaGetTextureObjectResourceDesc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetTextureObjectResourceDesc");
    mrcudaSym->mrcudaGetTextureObjectTextureDesc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetTextureObjectTextureDesc");
    mrcudaSym->mrcudaGetTextureObjectResourceViewDesc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetTextureObjectResourceViewDesc");
    mrcudaSym->mrcudaCreateSurfaceObject = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaCreateSurfaceObject");
    mrcudaSym->mrcudaDestroySurfaceObject = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDestroySurfaceObject");
    mrcudaSym->mrcudaGetSurfaceObjectResourceDesc = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetSurfaceObjectResourceDesc");
    mrcudaSym->mrcudaDriverGetVersion = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaDriverGetVersion");
    mrcudaSym->mrcudaRuntimeGetVersion = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaRuntimeGetVersion");
    mrcudaSym->mrcudaGetExportTable = __safe_dlsym(mrcudaSym->handler.symHandler, "cudaGetExportTable");
    mrcudaSym->__mrcudaRegisterFatBinary = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterFatBinary");
    mrcudaSym->__mrcudaUnregisterFatBinary = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaUnregisterFatBinary");
    mrcudaSym->__mrcudaRegisterVar = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterVar");
    mrcudaSym->__mrcudaRegisterTexture = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterTexture");
    mrcudaSym->__mrcudaRegisterSurface = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterSurface");
    mrcudaSym->__mrcudaRegisterFunction = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterFunction");
    //mrcudaSym->__mrcudaRegisterShared = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterShared");
    //mrcudaSym->__mrcudaRegisterSharedVar = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterSharedVar");
    //mrcudaSym->__mrcudaSynchronizeThreads = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaSynchronizeThreads");
    //mrcudaSym->__mrcudaTextureFetch = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaTextureFetch");
    //mrcudaSym->__mrcudaMutexOperation = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaMutexOperation");
    //mrcudaSym->__mrcudaRegisterDeviceFunction = __safe_dlsym(mrcudaSym->handler.symHandler, "__cudaRegisterDeviceFunction");
}

/**
 * Initialize mrCUDA.
 * Print error and terminate the program if an error occurs.
 */
__attribute__((constructor))
void mrcuda_init()
{
    char *switchThreshold;
    char *rCUDANumGPUs;
    char *endptr;
    char *tmp;
    char envName[32];
    int i, j;

    if (mrcudaState == MRCUDA_STATE_UNINITIALIZED) {
        // Get configurations from environment variables.
        __rCUDALibPath = getenv(__RCUDA_LIBRARY_PATH_ENV_NAME__);
        if (__rCUDALibPath == NULL || strlen(__rCUDALibPath) == 0)
            REPORT_ERROR_AND_EXIT("%s is not specified.\n", __RCUDA_LIBRARY_PATH_ENV_NAME__);

        __nvidiaLibPath = getenv(__NVIDIA_LIBRARY_PATH_ENV_NAME__);
        if (__nvidiaLibPath == NULL || strlen(__nvidiaLibPath) == 0)
            REPORT_ERROR_AND_EXIT("%s is not specified.\n", __NVIDIA_LIBRARY_PATH_ENV_NAME__);

        __sockPath = getenv(__SOCK_PATH_ENV_NAME__);
        if (__sockPath == NULL || strlen(__sockPath) == 0)
            REPORT_ERROR_AND_EXIT("%s is not specified.\n", __SOCK_PATH_ENV_NAME__);

        __helperPath = getenv(__MHELPER_PATH_ENV_NAME__);
        if (__helperPath == NULL || strlen(__helperPath) == 0)
            REPORT_ERROR_AND_EXIT("%s is not specified.\n", __MHELPER_PATH_ENV_NAME__);

        // Initialize mrcudaNumGPUs and mrcudaGPUInfos
        if ((rCUDANumGPUs = getenv(__RCUDA_DEVICE_COUNT_ENV_NAME__)) == NULL)
            REPORT_ERROR_AND_EXIT(__RCUDA_DEVICE_COUNT_ENV_NAME__ " is not specified.\n");

        // Allocate space for global variables.
        if ((mrcudaSymRCUDA = malloc(sizeof(MRCUDASym_t))) == NULL)
            REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymRCUDA.\n");

        if ((mrcudaSymNvidia = malloc(sizeof(MRCUDASym_t))) == NULL)
            REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymNvidia.\n");
        
        // Create handles for CUDA libraries.
        mrcudaSymNvidia->handler.symHandler = dlopen(__nvidiaLibPath, RTLD_NOW | RTLD_GLOBAL);
        if (mrcudaSymNvidia->handler.symHandler == NULL)
            REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymNvidia.\n");

        mrcudaSymRCUDA->handler.symHandler = dlopen(__rCUDALibPath, RTLD_NOW | RTLD_GLOBAL);
        if (mrcudaSymRCUDA->handler.symHandler == NULL)
            REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymRCUDA.\n");
        
        // Symlink rCUDA and CUDA functions.
        __symlink_handle(mrcudaSymRCUDA);
        __symlink_handle(mrcudaSymNvidia);

        // Initialize each GPU information.
        mrcudaNumGPUs = (int)strtol(rCUDANumGPUs, &endptr, 10);
        if (*endptr != '\0')
            REPORT_ERROR_AND_EXIT(__RCUDA_DEVICE_COUNT_ENV_NAME__"'s value is not valid.\n");
        else if ((mrcudaGPUList = calloc(mrcudaNumGPUs, sizeof(MRCUDAGPU_t))) == NULL)
            REPORT_ERROR_AND_EXIT("Cannot allocate memmory for mrcudaGPUInfos.\n");

        for (i = 0; i < mrcudaNumGPUs; i++) {
            mrcudaGPUList[i].cudaLaunchCount = 0;
            mrcudaGPUList[i].virtualNumber = i;
            sprintf(envName, __RCUDA_DEVICE_HEADER_ENV_NAME__"%d", i);
            if ((tmp = getenv(envName)) == NULL)
                REPORT_ERROR_AND_EXIT("%s is not specified.\n", envName);
            if (strcasestr(tmp, __LOCALHOST__) != NULL) {   // native GPU
                mrcudaGPUList[i].nativeFromStart = 1;
                mrcudaGPUList[i].status = MRCUDA_GPU_STATUS_NATIVE;
                mrcudaGPUList[i].defaultHandler = mrcudaSymNvidia;
                __hasNativeGPU = 1;
            }
            else {    // rCUDA GPU
                mrcudaGPUList[i].nativeFromStart = 0;
                mrcudaGPUList[i].status = MRCUDA_GPU_STATUS_RCUDA;
                mrcudaGPUList[i].defaultHandler = mrcudaSymRCUDA;
            }

            j = 0;
            while (tmp[j] != '\0' && tmp[j] != ':')
                j++;
            if (tmp[j] == ':') {
                mrcudaGPUList[i].realNumber = (int)strtol(&(tmp[j + 1]), &endptr, 10);
                if (*endptr != '\0')
                    REPORT_ERROR_AND_EXIT("%s's value is not valid.\n", envName);
            }
            else
                mrcudaGPUList[i].realNumber = 0;

            // Get the threshold value.
            sprintf(envName, __MRCUDA_SWITCH_THRESHOLD_HEADER_ENV_NAME__"%d", i);
            if ((tmp = getenv(envName)) == NULL)
                REPORT_ERROR_AND_EXIT("%s is not specified.\n", envName);
            mrcudaGPUList[i].switchThreshold = (int)strtol(tmp, &endptr, 10);
            if (*endptr != '\0')
                REPORT_ERROR_AND_EXIT("%s's value is not valid.\n", envName);
        }

        // Initialize the record/replay module.
        mrcuda_record_init();

        // Initialize mutex for switching locking.
        pthread_mutex_init(&__switch_mutex, NULL);

        mrcudaState = MRCUDA_STATE_RUNNING;

        // Start listening on the specified socket path.
        /*if(mrcuda_comm_listen_for_signal(__sockPath, &mrcuda_switch) != 0)
            REPORT_ERROR_AND_EXIT("Encounter a problem with the specified socket path.\n");*/
    }
}


/**
 * Finalize mrCUDA.
 */
__attribute__((destructor))
int mrcuda_fini()
{
    if (mrcudaState == MRCUDA_STATE_RUNNING) {
        // Close and free mrcudaSymRCUDA resources.
        if (mrcudaSymRCUDA) {
            if (mrcudaSymRCUDA->handler.symHandler)
                dlclose(mrcudaSymRCUDA->handler.symHandler);
            free(mrcudaSymRCUDA);
        }

        // Close and free mrcudaSymNvidia resources.
        if (mrcudaSymNvidia) {
            if (mrcudaSymNvidia->handler.symHandler)
                dlclose(mrcudaSymNvidia->handler.symHandler);
            free(mrcudaSymNvidia);
        }

        mrcuda_record_fini();

        pthread_mutex_destroy(&__switch_mutex);

        mrcudaState = MRCUDA_STATE_FINALIZED;
    }
}

/**
 * Lock the switch mutex.
 */
static inline void __mrcuda_switch_lock()
{
    if (mrcudaState == MRCUDA_STATE_RUNNING && pthread_mutex_lock(&__switch_mutex) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __switch_mutex locking process.\n");
}

/**
 * Release the switch mutex.
 */
static inline void __mrcuda_switch_release()
{
    if (mrcudaState == MRCUDA_STATE_RUNNING && pthread_mutex_unlock(&__switch_mutex) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __switch_mutex unlocking process.\n");
}

/**
 * Switch the specified mrcudaGPU from rCUDA to native.
 * @param mrcudaGPU a ptr to the mrcudaGPU to be switched.
 * @param toGPUNumber the native GPU number to be moved to.
 */
void mrcuda_switch(MRCUDAGPU_t *mrcudaGPU, int toGPUNumber)
{
    MRecord_t *record = NULL;
    int already_mock_stream = 0;
    MRCUDAGPU_t *currentGPU;

    if (mrcudaState == MRCUDA_STATE_RUNNING && mrcudaGPU->status == MRCUDA_GPU_STATUS_RCUDA) {
        __mrcuda_switch_lock();
        mrcuda_function_call_lock(mrcudaGPU);

        // Save the current GPU of this thread.
        currentGPU = mrcuda_get_current_gpu();

        // Temporary switch to the GPU to be switched.
        mrcuda_set_current_gpu(mrcudaGPU->virtualNumber);

        // Set up migration.
        if (!__hasNativeGPU) {
            // Set up rCUDA-to-native migration.
            __hasNativeGPU = !__hasNativeGPU;
            mrcudaGPU->status = MRCUDA_GPU_STATUS_NATIVE;
            mrcudaGPU->defaultHandler = mrcudaSymNvidia;
        }
        else {
            // Set up rCUDA-to-helper migration.
            mrcudaGPU->status = MRCUDA_GPU_STATUS_HELPER;
            if (mhelper_create(mrcudaGPU, __helperPath, toGPUNumber) == NULL)
                REPORT_ERROR_AND_EXIT("Something went wrong in mhelper_create.\n");
            mrcudaGPU->defaultHandler = mrcudaGPU->mhelperProcess->handle;
        }

        // Create a context to emulate rCUDA address space.
        if (mrcudaGPU->realNumber != 0 && mrcuda_simulate_cuCtxCreate(mrcudaGPU, toGPUNumber) != 0)
            REPORT_ERROR_AND_EXIT("Cannot simulate cuCtxCreate.\n");
        
        mrcudaGPU->realNumber = toGPUNumber;
        
        // Waiting for everything to be stable on rCUDA side.
        mrcudaSymRCUDA->mrcudaSetDevice(mrcudaGPU->virtualNumber);
        sleep(1);
        mrcudaSymRCUDA->mrcudaDeviceSynchronize();

        // Replay recorded commands.
        record = mrcudaGPU->mrecordGPU->mrcudaRecordHeadPtr;
        while (record != NULL) {
            if (!already_mock_stream && !(record->skipMockStream)) {
                if (mrcudaGPU->status == MRCUDA_GPU_STATUS_HELPER && mhelper_int_cudaSetDevice_internal(mrcudaGPU, toGPUNumber) != cudaSuccess)
                    REPORT_ERROR_AND_EXIT("Cannot set device in the mhelper.\n");
                mrcuda_simulate_stream(mrcudaGPU);
                already_mock_stream = !already_mock_stream;
            }
            record->replayFunc(mrcudaGPU, record);
            record = record->next;
        }

        if (!already_mock_stream) {
            if (mrcudaGPU->status == MRCUDA_GPU_STATUS_HELPER && mhelper_int_cudaSetDevice_internal(mrcudaGPU, toGPUNumber) != cudaSuccess)
                REPORT_ERROR_AND_EXIT("Cannot set device in the mhelper.\n");
            mrcuda_simulate_stream(mrcudaGPU);
            already_mock_stream = !already_mock_stream;
        }

        mrcuda_sync_mem(mrcudaGPU);

        // Restore current device
        mrcudaSymRCUDA->mrcudaSetDevice(currentGPU->virtualNumber);

        mrcuda_function_call_release(mrcudaGPU);
        __mrcuda_switch_release();
    }
}

/**
 * Create a barrier such that subsequent calls are blocked until the barrier is released.
 * @param mrcudaGPU a ptr to the GPU a barrier will be created on.
 */
void mrcuda_function_call_lock(MRCUDAGPU_t *mrcudaGPU)
{
    if (mrcudaState == MRCUDA_STATE_RUNNING && pthread_mutex_lock(&(mrcudaGPU->mutex)) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __processing_func_mutex locking process.\n");
}

/**
 * Release the barrier; thus, allow subsequent calls to be processed normally.
 * @param mrcudaGPU a ptr to the GPU the barrier will be released.
 */
void mrcuda_function_call_release(MRCUDAGPU_t *mrcudaGPU)
{
    if (mrcudaState == MRCUDA_STATE_RUNNING && pthread_mutex_unlock(&(mrcudaGPU->mutex)) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __processing_func_mutex unlocking process.\n");
}

