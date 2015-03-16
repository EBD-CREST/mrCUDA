#include "common.h"
#include "mrcuda.h"
#include "record.h"
#include "comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>

#define __RCUDA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_RCUDA_LIB_PATH"
#define __NVIDIA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_NVIDIA_LIB_PATH"
#define __SOCK_PATH_ENV_NAME__ "MRCUDA_SOCK_PATH"

MRCUDASym *mrcudaSymNvidia;
MRCUDASym *mrcudaSymRCUDA;
MRCUDASym *mrcudaSymDefault;

static char *__rCUDALibPath;
static char *__nvidiaLibPath;

char *__sockPath;

static pthread_mutex_t __processing_func_mutex;

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
static void __symlink_handle(MRCUDASym *mrcudaSym)
{
    mrcudaSym->mrcudaDeviceReset = __safe_dlsym(mrcudaSym->handle, "cudaDeviceReset");
    mrcudaSym->mrcudaDeviceSynchronize = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceSynchronize");
    mrcudaSym->mrcudaDeviceSetLimit = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceSetLimit");
    mrcudaSym->mrcudaDeviceGetLimit = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceGetLimit");
    mrcudaSym->mrcudaDeviceGetCacheConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceGetCacheConfig");
    mrcudaSym->mrcudaDeviceSetCacheConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceSetCacheConfig");
    mrcudaSym->mrcudaDeviceGetSharedMemConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceGetSharedMemConfig");
    mrcudaSym->mrcudaDeviceSetSharedMemConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceSetSharedMemConfig");
    mrcudaSym->mrcudaDeviceGetByPCIBusId = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceGetByPCIBusId");
    mrcudaSym->mrcudaDeviceGetPCIBusId = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceGetPCIBusId");
    mrcudaSym->mrcudaIpcGetEventHandle = __safe_dlsym(mrcudaSym->handle, "mrcudaIpcGetEventHandle");
    mrcudaSym->mrcudaIpcOpenEventHandle = __safe_dlsym(mrcudaSym->handle, "mrcudaIpcOpenEventHandle");
    mrcudaSym->mrcudaIpcGetMemHandle = __safe_dlsym(mrcudaSym->handle, "mrcudaIpcGetMemHandle");
    mrcudaSym->mrcudaIpcOpenMemHandle = __safe_dlsym(mrcudaSym->handle, "mrcudaIpcOpenMemHandle");
    mrcudaSym->mrcudaIpcCloseMemHandle = __safe_dlsym(mrcudaSym->handle, "mrcudaIpcCloseMemHandle");
    mrcudaSym->mrcudaThreadExit = __safe_dlsym(mrcudaSym->handle, "mrcudaThreadExit");
    mrcudaSym->mrcudaThreadSynchronize = __safe_dlsym(mrcudaSym->handle, "mrcudaThreadSynchronize");
    mrcudaSym->mrcudaThreadSetLimit = __safe_dlsym(mrcudaSym->handle, "mrcudaThreadSetLimit");
    mrcudaSym->mrcudaThreadGetLimit = __safe_dlsym(mrcudaSym->handle, "mrcudaThreadGetLimit");
    mrcudaSym->mrcudaThreadGetCacheConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaThreadGetCacheConfig");
    mrcudaSym->mrcudaThreadSetCacheConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaThreadSetCacheConfig");
    mrcudaSym->mrcudaGetLastError = __safe_dlsym(mrcudaSym->handle, "mrcudaGetLastError");
    mrcudaSym->mrcudaPeekAtLastError = __safe_dlsym(mrcudaSym->handle, "mrcudaPeekAtLastError");
    mrcudaSym->mrcudaGetErrorString = __safe_dlsym(mrcudaSym->handle, "mrcudaGetErrorString");
    mrcudaSym->mrcudaGetDeviceCount = __safe_dlsym(mrcudaSym->handle, "mrcudaGetDeviceCount");
    mrcudaSym->mrcudaGetDeviceProperties = __safe_dlsym(mrcudaSym->handle, "mrcudaGetDeviceProperties");
    mrcudaSym->mrcudaDeviceGetAttribute = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceGetAttribute");
    mrcudaSym->mrcudaChooseDevice = __safe_dlsym(mrcudaSym->handle, "mrcudaChooseDevice");
    mrcudaSym->mrcudaSetDevice = __safe_dlsym(mrcudaSym->handle, "mrcudaSetDevice");
    mrcudaSym->mrcudaGetDevice = __safe_dlsym(mrcudaSym->handle, "mrcudaGetDevice");
    mrcudaSym->mrcudaSetValidDevices = __safe_dlsym(mrcudaSym->handle, "mrcudaSetValidDevices");
    mrcudaSym->mrcudaSetDeviceFlags = __safe_dlsym(mrcudaSym->handle, "mrcudaSetDeviceFlags");
    mrcudaSym->mrcudaStreamCreate = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamCreate");
    mrcudaSym->mrcudaStreamCreateWithFlags = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamCreateWithFlags");
    mrcudaSym->mrcudaStreamDestroy = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamDestroy");
    mrcudaSym->mrcudaStreamWaitEvent = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamWaitEvent");
    mrcudaSym->mrcudaStreamAddCallback = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamAddCallback");
    mrcudaSym->mrcudaStreamSynchronize = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamSynchronize");
    mrcudaSym->mrcudaStreamQuery = __safe_dlsym(mrcudaSym->handle, "mrcudaStreamQuery");
    mrcudaSym->mrcudaEventCreate = __safe_dlsym(mrcudaSym->handle, "mrcudaEventCreate");
    mrcudaSym->mrcudaEventCreateWithFlags = __safe_dlsym(mrcudaSym->handle, "mrcudaEventCreateWithFlags");
    mrcudaSym->mrcudaEventRecord = __safe_dlsym(mrcudaSym->handle, "mrcudaEventRecord");
    mrcudaSym->mrcudaEventQuery = __safe_dlsym(mrcudaSym->handle, "mrcudaEventQuery");
    mrcudaSym->mrcudaEventSynchronize = __safe_dlsym(mrcudaSym->handle, "mrcudaEventSynchronize");
    mrcudaSym->mrcudaEventDestroy = __safe_dlsym(mrcudaSym->handle, "mrcudaEventDestroy");
    mrcudaSym->mrcudaEventElapsedTime = __safe_dlsym(mrcudaSym->handle, "mrcudaEventElapsedTime");
    mrcudaSym->mrcudaConfigureCall = __safe_dlsym(mrcudaSym->handle, "mrcudaConfigureCall");
    mrcudaSym->mrcudaSetupArgument = __safe_dlsym(mrcudaSym->handle, "mrcudaSetupArgument");
    mrcudaSym->mrcudaFuncSetCacheConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaFuncSetCacheConfig");
    mrcudaSym->mrcudaFuncSetSharedMemConfig = __safe_dlsym(mrcudaSym->handle, "mrcudaFuncSetSharedMemConfig");
    mrcudaSym->mrcudaLaunch = __safe_dlsym(mrcudaSym->handle, "mrcudaLaunch");
    mrcudaSym->mrcudaFuncGetAttributes = __safe_dlsym(mrcudaSym->handle, "mrcudaFuncGetAttributes");
    mrcudaSym->mrcudaSetDoubleForDevice = __safe_dlsym(mrcudaSym->handle, "mrcudaSetDoubleForDevice");
    mrcudaSym->mrcudaSetDoubleForHost = __safe_dlsym(mrcudaSym->handle, "mrcudaSetDoubleForHost");
    mrcudaSym->mrcudaMalloc = __safe_dlsym(mrcudaSym->handle, "mrcudaMalloc");
    mrcudaSym->mrcudaMallocHost = __safe_dlsym(mrcudaSym->handle, "mrcudaMallocHost");
    mrcudaSym->mrcudaMallocPitch = __safe_dlsym(mrcudaSym->handle, "mrcudaMallocPitch");
    mrcudaSym->mrcudaMallocArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMallocArray");
    mrcudaSym->mrcudaFree = __safe_dlsym(mrcudaSym->handle, "mrcudaFree");
    mrcudaSym->mrcudaFreeHost = __safe_dlsym(mrcudaSym->handle, "mrcudaFreeHost");
    mrcudaSym->mrcudaFreeArray = __safe_dlsym(mrcudaSym->handle, "mrcudaFreeArray");
    mrcudaSym->mrcudaFreeMipmappedArray = __safe_dlsym(mrcudaSym->handle, "mrcudaFreeMipmappedArray");
    mrcudaSym->mrcudaHostAlloc = __safe_dlsym(mrcudaSym->handle, "mrcudaHostAlloc");
    mrcudaSym->mrcudaHostRegister = __safe_dlsym(mrcudaSym->handle, "mrcudaHostRegister");
    mrcudaSym->mrcudaHostUnregister = __safe_dlsym(mrcudaSym->handle, "mrcudaHostUnregister");
    mrcudaSym->mrcudaHostGetDevicePointer = __safe_dlsym(mrcudaSym->handle, "mrcudaHostGetDevicePointer");
    mrcudaSym->mrcudaHostGetFlags = __safe_dlsym(mrcudaSym->handle, "mrcudaHostGetFlags");
    mrcudaSym->mrcudaMalloc3D = __safe_dlsym(mrcudaSym->handle, "mrcudaMalloc3D");
    mrcudaSym->mrcudaMalloc3DArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMalloc3DArray");
    mrcudaSym->mrcudaMallocMipmappedArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMallocMipmappedArray");
    mrcudaSym->mrcudaGetMipmappedArrayLevel = __safe_dlsym(mrcudaSym->handle, "mrcudaGetMipmappedArrayLevel");
    mrcudaSym->mrcudaMemcpy3D = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy3D");
    mrcudaSym->mrcudaMemcpy3DPeer = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy3DPeer");
    mrcudaSym->mrcudaMemcpy3DAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy3DAsync");
    mrcudaSym->mrcudaMemcpy3DPeerAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy3DPeerAsync");
    mrcudaSym->mrcudaMemGetInfo = __safe_dlsym(mrcudaSym->handle, "mrcudaMemGetInfo");
    mrcudaSym->mrcudaArrayGetInfo = __safe_dlsym(mrcudaSym->handle, "mrcudaArrayGetInfo");
    mrcudaSym->mrcudaMemcpy = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy");
    mrcudaSym->mrcudaMemcpyPeer = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyPeer");
    mrcudaSym->mrcudaMemcpyToArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyToArray");
    mrcudaSym->mrcudaMemcpyFromArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyFromArray");
    mrcudaSym->mrcudaMemcpyArrayToArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyArrayToArray");
    mrcudaSym->mrcudaMemcpy2D = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2D");
    mrcudaSym->mrcudaMemcpy2DToArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2DToArray");
    mrcudaSym->mrcudaMemcpy2DFromArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2DFromArray");
    mrcudaSym->mrcudaMemcpy2DArrayToArray = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2DArrayToArray");
    mrcudaSym->mrcudaMemcpyToSymbol = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyToSymbol");
    mrcudaSym->mrcudaMemcpyFromSymbol = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyFromSymbol");
    mrcudaSym->mrcudaMemcpyAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyAsync");
    mrcudaSym->mrcudaMemcpyPeerAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyPeerAsync");
    mrcudaSym->mrcudaMemcpyToArrayAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyToArrayAsync");
    mrcudaSym->mrcudaMemcpyFromArrayAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyFromArrayAsync");
    mrcudaSym->mrcudaMemcpy2DAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2DAsync");
    mrcudaSym->mrcudaMemcpy2DToArrayAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2DToArrayAsync");
    mrcudaSym->mrcudaMemcpy2DFromArrayAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpy2DFromArrayAsync");
    mrcudaSym->mrcudaMemcpyToSymbolAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyToSymbolAsync");
    mrcudaSym->mrcudaMemcpyFromSymbolAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemcpyFromSymbolAsync");
    mrcudaSym->mrcudaMemset = __safe_dlsym(mrcudaSym->handle, "mrcudaMemset");
    mrcudaSym->mrcudaMemset2D = __safe_dlsym(mrcudaSym->handle, "mrcudaMemset2D");
    mrcudaSym->mrcudaMemset3D = __safe_dlsym(mrcudaSym->handle, "mrcudaMemset3D");
    mrcudaSym->mrcudaMemsetAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemsetAsync");
    mrcudaSym->mrcudaMemset2DAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemset2DAsync");
    mrcudaSym->mrcudaMemset3DAsync = __safe_dlsym(mrcudaSym->handle, "mrcudaMemset3DAsync");
    mrcudaSym->mrcudaGetSymbolAddress = __safe_dlsym(mrcudaSym->handle, "mrcudaGetSymbolAddress");
    mrcudaSym->mrcudaGetSymbolSize = __safe_dlsym(mrcudaSym->handle, "mrcudaGetSymbolSize");
    mrcudaSym->mrcudaPointerGetAttributes = __safe_dlsym(mrcudaSym->handle, "mrcudaPointerGetAttributes");
    mrcudaSym->mrcudaDeviceCanAccessPeer = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceCanAccessPeer");
    mrcudaSym->mrcudaDeviceEnablePeerAccess = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceEnablePeerAccess");
    mrcudaSym->mrcudaDeviceDisablePeerAccess = __safe_dlsym(mrcudaSym->handle, "mrcudaDeviceDisablePeerAccess");
    mrcudaSym->mrcudaGraphicsUnregisterResource = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsUnregisterResource");
    mrcudaSym->mrcudaGraphicsResourceSetMapFlags = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsResourceSetMapFlags");
    mrcudaSym->mrcudaGraphicsMapResources = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsMapResources");
    mrcudaSym->mrcudaGraphicsUnmapResources = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsUnmapResources");
    mrcudaSym->mrcudaGraphicsResourceGetMappedPointer = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsResourceGetMappedPointer");
    mrcudaSym->mrcudaGraphicsSubResourceGetMappedArray = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsSubResourceGetMappedArray");
    mrcudaSym->mrcudaGraphicsResourceGetMappedMipmappedArray = __safe_dlsym(mrcudaSym->handle, "mrcudaGraphicsResourceGetMappedMipmappedArray");
    mrcudaSym->mrcudaGetChannelDesc = __safe_dlsym(mrcudaSym->handle, "mrcudaGetChannelDesc");
    mrcudaSym->mrcudaCreateChannelDesc = __safe_dlsym(mrcudaSym->handle, "mrcudaCreateChannelDesc");
    mrcudaSym->mrcudaBindTexture = __safe_dlsym(mrcudaSym->handle, "mrcudaBindTexture");
    mrcudaSym->mrcudaBindTexture2D = __safe_dlsym(mrcudaSym->handle, "mrcudaBindTexture2D");
    mrcudaSym->mrcudaBindTextureToArray = __safe_dlsym(mrcudaSym->handle, "mrcudaBindTextureToArray");
    mrcudaSym->mrcudaBindTextureToMipmappedArray = __safe_dlsym(mrcudaSym->handle, "mrcudaBindTextureToMipmappedArray");
    mrcudaSym->mrcudaUnbindTexture = __safe_dlsym(mrcudaSym->handle, "mrcudaUnbindTexture");
    mrcudaSym->mrcudaGetTextureAlignmentOffset = __safe_dlsym(mrcudaSym->handle, "mrcudaGetTextureAlignmentOffset");
    mrcudaSym->mrcudaGetTextureReference = __safe_dlsym(mrcudaSym->handle, "mrcudaGetTextureReference");
    mrcudaSym->mrcudaBindSurfaceToArray = __safe_dlsym(mrcudaSym->handle, "mrcudaBindSurfaceToArray");
    mrcudaSym->mrcudaGetSurfaceReference = __safe_dlsym(mrcudaSym->handle, "mrcudaGetSurfaceReference");
    mrcudaSym->mrcudaCreateTextureObject = __safe_dlsym(mrcudaSym->handle, "mrcudaCreateTextureObject");
    mrcudaSym->mrcudaDestroyTextureObject = __safe_dlsym(mrcudaSym->handle, "mrcudaDestroyTextureObject");
    mrcudaSym->mrcudaGetTextureObjectResourceDesc = __safe_dlsym(mrcudaSym->handle, "mrcudaGetTextureObjectResourceDesc");
    mrcudaSym->mrcudaGetTextureObjectTextureDesc = __safe_dlsym(mrcudaSym->handle, "mrcudaGetTextureObjectTextureDesc");
    mrcudaSym->mrcudaGetTextureObjectResourceViewDesc = __safe_dlsym(mrcudaSym->handle, "mrcudaGetTextureObjectResourceViewDesc");
    mrcudaSym->mrcudaCreateSurfaceObject = __safe_dlsym(mrcudaSym->handle, "mrcudaCreateSurfaceObject");
    mrcudaSym->mrcudaDestroySurfaceObject = __safe_dlsym(mrcudaSym->handle, "mrcudaDestroySurfaceObject");
    mrcudaSym->mrcudaGetSurfaceObjectResourceDesc = __safe_dlsym(mrcudaSym->handle, "mrcudaGetSurfaceObjectResourceDesc");
    mrcudaSym->mrcudaDriverGetVersion = __safe_dlsym(mrcudaSym->handle, "mrcudaDriverGetVersion");
    mrcudaSym->mrcudaRuntimeGetVersion = __safe_dlsym(mrcudaSym->handle, "mrcudaRuntimeGetVersion");
    mrcudaSym->mrcudaGetExportTable = __safe_dlsym(mrcudaSym->handle, "mrcudaGetExportTable");
    mrcudaSym->__mrcudaRegisterFatBinary = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterFatBinary");
    mrcudaSym->__mrcudaUnregisterFatBinary = __safe_dlsym(mrcudaSym->handle, "__mrcudaUnregisterFatBinary");
    mrcudaSym->__mrcudaRegisterVar = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterVar");
    mrcudaSym->__mrcudaRegisterTexture = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterTexture");
    mrcudaSym->__mrcudaRegisterSurface = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterSurface");
    mrcudaSym->__mrcudaRegisterFunction = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterFunction");
    mrcudaSym->__mrcudaRegisterShared = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterShared");
    mrcudaSym->__mrcudaRegisterSharedVar = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterSharedVar");
    mrcudaSym->__mrcudaSynchronizeThreads = __safe_dlsym(mrcudaSym->handle, "__mrcudaSynchronizeThreads");
    mrcudaSym->__mrcudaTextureFetch = __safe_dlsym(mrcudaSym->handle, "__mrcudaTextureFetch");
    mrcudaSym->__mrcudaMutexOperation = __safe_dlsym(mrcudaSym->handle, "__mrcudaMutexOperation");
    mrcudaSym->__mrcudaRegisterDeviceFunction = __safe_dlsym(mrcudaSym->handle, "__mrcudaRegisterDeviceFunction");
}

/**
 * Initialize mrCUDA.
 * Print error and terminate the program if an error occurs.
 */
__attribute__((constructor))
void mrcuda_init()
{
    // Get configurations from environment variables.
    __rCUDALibPath = getenv(__RCUDA_LIBRARY_PATH_ENV_NAME__);
    if(__rCUDALibPath == NULL || strlen(__rCUDALibPath) == 0)
        REPORT_ERROR_AND_EXIT("%s is not specified.\n", __RCUDA_LIBRARY_PATH_ENV_NAME__);

    __nvidiaLibPath = getenv(__NVIDIA_LIBRARY_PATH_ENV_NAME__);
    if(__nvidiaLibPath == NULL || strlen(__nvidiaLibPath) == 0)
        REPORT_ERROR_AND_EXIT("%s is not specified.\n", __NVIDIA_LIBRARY_PATH_ENV_NAME__);

    __sockPath = getenv(__SOCK_PATH_ENV_NAME__);
    if(__sockPath == NULL || strlen(__sockPath) == 0)
        REPORT_ERROR_AND_EXIT("%s is not specified.\n", __SOCK_PATH_ENV_NAME__);

    // Allocate space for global variables.
    mrcudaSymRCUDA = malloc(sizeof(MRCUDASym));
    if(mrcudaSymRCUDA == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymRCUDA.\n");

    mrcudaSymNvidia = malloc(sizeof(MRCUDASym));
    if(mrcudaSymNvidia == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymNvidia.\n");

    // Assign mrcudaSymDefault to mrcudaSymRCUDA.
    mrcudaSymDefault = mrcudaSymRCUDA;


    // Create handles for CUDA libraries.
    mrcudaSymRCUDA->handle = dlopen(__rCUDALibPath, RTLD_NOW | RTLD_GLOBAL);
    if(mrcudaSymRCUDA->handle == NULL)
        REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymRCUDA.\n");

    mrcudaSymNvidia->handle = dlopen(__nvidiaLibPath, RTLD_NOW | RTLD_GLOBAL);
        REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymNvidia.\n");

    // Symlink rCUDA and CUDA functions.
    __symlink_handle(mrcudaSymRCUDA);
    __symlink_handle(mrcudaSymNvidia);

    // Initialize the record/replay module.
    mrcuda_record_init();

    // Initialize mutex for switching locking.
    pthread_mutex_init(&__processing_func_mutex, NULL);

    // Start listening on the specified socket path.
    if(mrcuda_comm_listen_for_signal(__sockPath, &mrcuda_switch) != 0)
        REPORT_ERROR_AND_EXIT("Encounter a problem with the specified socket path.\n");
}


/**
 * Finalize mrCUDA.
 */
__attribute__((destructor))
int mrcuda_fini()
{
    // Close and free mrcudaSymRCUDA resources.
    if(mrcudaSymRCUDA)
    {
        if(mrcudaSymRCUDA->handle)
            dlclose(mrcudaSymRCUDA->handle);
        free(mrcudaSymRCUDA);
    }

    // Close and free mrcudaSymNvidia resources.
    if(mrcudaSymNvidia)
    {
        if(mrcudaSymNvidia->handle)
            dlclose(mrcudaSymNvidia->handle);
        free(mrcudaSymNvidia);
    }

    mrcuda_record_fini();

	pthread_mutex_destroy(&__processing_func_mutex);
}

/**
 * Switch from rCUDA to native.
 */
void mrcuda_switch()
{
    MRecord *record = NULL;
    mrcuda_function_call_lock();
    mrcudaSymRCUDA->mrcudaDeviceSynchronize();
    record = mrcudaRecordHeadPtr;
    while(record != NULL)
    {
        record->replayFunc(record);
        record = record->next;
    }
    mrcuda_sync_mem();
    mrcudaSymDefault = mrcudaSymNvidia;
    mrcuda_function_call_release();
}

/**
 * Create a barrier such that subsequent calls are blocked until the barrier is released.
 */
void mrcuda_function_call_lock()
{
    if(pthread_mutex_lock(&__processing_func_mutex) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __processing_func_mutex locking process.\n");
}

/**
 * Release the barrier; thus, allow subsequent calls to be processed normally.
 */
void mrcuda_function_call_release()
{
    if(pthread_mutex_unlock(&__processing_func_mutex) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __processing_func_mutex unlocking process.\n");
}

