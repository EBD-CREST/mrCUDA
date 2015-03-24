#include "common.h"
#include "mrcuda.h"
#include "record.h"
#include "comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>

// For manual profiling
#include <sys/time.h>

#define __RCUDA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_RCUDA_LIB_PATH"
#define __NVIDIA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_NVIDIA_LIB_PATH"
#define __SOCK_PATH_ENV_NAME__ "MRCUDA_SOCK_PATH"
#define __SWITCH_THRESHOLD_ENV_NAME__ "MRCUDA_SWITCH_THRESHOLD"

MRCUDASym *mrcudaSymNvidia;
MRCUDASym *mrcudaSymRCUDA;
MRCUDASym *mrcudaSymDefault;

static char *__rCUDALibPath;
static char *__nvidiaLibPath;

char *__sockPath;

static pthread_mutex_t __processing_func_mutex;

enum mrcudaStateEnum mrcudaState = MRCUDA_STATE_UNINITIALIZED;

long int mrcudaNumLaunchSwitchThreashold;

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
    mrcudaSym->mrcudaDeviceSynchronize = __safe_dlsym(mrcudaSym->handle, "cudaDeviceSynchronize");
    mrcudaSym->mrcudaDeviceSetLimit = __safe_dlsym(mrcudaSym->handle, "cudaDeviceSetLimit");
    mrcudaSym->mrcudaDeviceGetLimit = __safe_dlsym(mrcudaSym->handle, "cudaDeviceGetLimit");
    mrcudaSym->mrcudaDeviceGetCacheConfig = __safe_dlsym(mrcudaSym->handle, "cudaDeviceGetCacheConfig");
    mrcudaSym->mrcudaDeviceSetCacheConfig = __safe_dlsym(mrcudaSym->handle, "cudaDeviceSetCacheConfig");
    mrcudaSym->mrcudaDeviceGetSharedMemConfig = __safe_dlsym(mrcudaSym->handle, "cudaDeviceGetSharedMemConfig");
    mrcudaSym->mrcudaDeviceSetSharedMemConfig = __safe_dlsym(mrcudaSym->handle, "cudaDeviceSetSharedMemConfig");
    mrcudaSym->mrcudaDeviceGetByPCIBusId = __safe_dlsym(mrcudaSym->handle, "cudaDeviceGetByPCIBusId");
    mrcudaSym->mrcudaDeviceGetPCIBusId = __safe_dlsym(mrcudaSym->handle, "cudaDeviceGetPCIBusId");
    mrcudaSym->mrcudaIpcGetEventHandle = __safe_dlsym(mrcudaSym->handle, "cudaIpcGetEventHandle");
    mrcudaSym->mrcudaIpcOpenEventHandle = __safe_dlsym(mrcudaSym->handle, "cudaIpcOpenEventHandle");
    mrcudaSym->mrcudaIpcGetMemHandle = __safe_dlsym(mrcudaSym->handle, "cudaIpcGetMemHandle");
    mrcudaSym->mrcudaIpcOpenMemHandle = __safe_dlsym(mrcudaSym->handle, "cudaIpcOpenMemHandle");
    mrcudaSym->mrcudaIpcCloseMemHandle = __safe_dlsym(mrcudaSym->handle, "cudaIpcCloseMemHandle");
    mrcudaSym->mrcudaThreadExit = __safe_dlsym(mrcudaSym->handle, "cudaThreadExit");
    mrcudaSym->mrcudaThreadSynchronize = __safe_dlsym(mrcudaSym->handle, "cudaThreadSynchronize");
    mrcudaSym->mrcudaThreadSetLimit = __safe_dlsym(mrcudaSym->handle, "cudaThreadSetLimit");
    mrcudaSym->mrcudaThreadGetLimit = __safe_dlsym(mrcudaSym->handle, "cudaThreadGetLimit");
    mrcudaSym->mrcudaThreadGetCacheConfig = __safe_dlsym(mrcudaSym->handle, "cudaThreadGetCacheConfig");
    mrcudaSym->mrcudaThreadSetCacheConfig = __safe_dlsym(mrcudaSym->handle, "cudaThreadSetCacheConfig");
    mrcudaSym->mrcudaGetLastError = __safe_dlsym(mrcudaSym->handle, "cudaGetLastError");
    mrcudaSym->mrcudaPeekAtLastError = __safe_dlsym(mrcudaSym->handle, "cudaPeekAtLastError");
    mrcudaSym->mrcudaGetErrorString = __safe_dlsym(mrcudaSym->handle, "cudaGetErrorString");
    mrcudaSym->mrcudaGetDeviceCount = __safe_dlsym(mrcudaSym->handle, "cudaGetDeviceCount");
    mrcudaSym->mrcudaGetDeviceProperties = __safe_dlsym(mrcudaSym->handle, "cudaGetDeviceProperties");
    mrcudaSym->mrcudaDeviceGetAttribute = __safe_dlsym(mrcudaSym->handle, "cudaDeviceGetAttribute");
    mrcudaSym->mrcudaChooseDevice = __safe_dlsym(mrcudaSym->handle, "cudaChooseDevice");
    mrcudaSym->mrcudaSetDevice = __safe_dlsym(mrcudaSym->handle, "cudaSetDevice");
    mrcudaSym->mrcudaGetDevice = __safe_dlsym(mrcudaSym->handle, "cudaGetDevice");
    mrcudaSym->mrcudaSetValidDevices = __safe_dlsym(mrcudaSym->handle, "cudaSetValidDevices");
    mrcudaSym->mrcudaSetDeviceFlags = __safe_dlsym(mrcudaSym->handle, "cudaSetDeviceFlags");
    mrcudaSym->mrcudaStreamCreate = __safe_dlsym(mrcudaSym->handle, "cudaStreamCreate");
    mrcudaSym->mrcudaStreamCreateWithFlags = __safe_dlsym(mrcudaSym->handle, "cudaStreamCreateWithFlags");
    mrcudaSym->mrcudaStreamDestroy = __safe_dlsym(mrcudaSym->handle, "cudaStreamDestroy");
    mrcudaSym->mrcudaStreamWaitEvent = __safe_dlsym(mrcudaSym->handle, "cudaStreamWaitEvent");
    mrcudaSym->mrcudaStreamAddCallback = __safe_dlsym(mrcudaSym->handle, "cudaStreamAddCallback");
    mrcudaSym->mrcudaStreamSynchronize = __safe_dlsym(mrcudaSym->handle, "cudaStreamSynchronize");
    mrcudaSym->mrcudaStreamQuery = __safe_dlsym(mrcudaSym->handle, "cudaStreamQuery");
    mrcudaSym->mrcudaEventCreate = __safe_dlsym(mrcudaSym->handle, "cudaEventCreate");
    mrcudaSym->mrcudaEventCreateWithFlags = __safe_dlsym(mrcudaSym->handle, "cudaEventCreateWithFlags");
    mrcudaSym->mrcudaEventRecord = __safe_dlsym(mrcudaSym->handle, "cudaEventRecord");
    mrcudaSym->mrcudaEventQuery = __safe_dlsym(mrcudaSym->handle, "cudaEventQuery");
    mrcudaSym->mrcudaEventSynchronize = __safe_dlsym(mrcudaSym->handle, "cudaEventSynchronize");
    mrcudaSym->mrcudaEventDestroy = __safe_dlsym(mrcudaSym->handle, "cudaEventDestroy");
    mrcudaSym->mrcudaEventElapsedTime = __safe_dlsym(mrcudaSym->handle, "cudaEventElapsedTime");
    mrcudaSym->mrcudaConfigureCall = __safe_dlsym(mrcudaSym->handle, "cudaConfigureCall");
    mrcudaSym->mrcudaSetupArgument = __safe_dlsym(mrcudaSym->handle, "cudaSetupArgument");
    mrcudaSym->mrcudaFuncSetCacheConfig = __safe_dlsym(mrcudaSym->handle, "cudaFuncSetCacheConfig");
    mrcudaSym->mrcudaFuncSetSharedMemConfig = __safe_dlsym(mrcudaSym->handle, "cudaFuncSetSharedMemConfig");
    mrcudaSym->mrcudaLaunch = __safe_dlsym(mrcudaSym->handle, "cudaLaunch");
    mrcudaSym->mrcudaFuncGetAttributes = __safe_dlsym(mrcudaSym->handle, "cudaFuncGetAttributes");
    mrcudaSym->mrcudaSetDoubleForDevice = __safe_dlsym(mrcudaSym->handle, "cudaSetDoubleForDevice");
    mrcudaSym->mrcudaSetDoubleForHost = __safe_dlsym(mrcudaSym->handle, "cudaSetDoubleForHost");
    mrcudaSym->mrcudaMalloc = __safe_dlsym(mrcudaSym->handle, "cudaMalloc");
    mrcudaSym->mrcudaMallocHost = __safe_dlsym(mrcudaSym->handle, "cudaMallocHost");
    mrcudaSym->mrcudaMallocPitch = __safe_dlsym(mrcudaSym->handle, "cudaMallocPitch");
    mrcudaSym->mrcudaMallocArray = __safe_dlsym(mrcudaSym->handle, "cudaMallocArray");
    mrcudaSym->mrcudaFree = __safe_dlsym(mrcudaSym->handle, "cudaFree");
    mrcudaSym->mrcudaFreeHost = __safe_dlsym(mrcudaSym->handle, "cudaFreeHost");
    mrcudaSym->mrcudaFreeArray = __safe_dlsym(mrcudaSym->handle, "cudaFreeArray");
    mrcudaSym->mrcudaFreeMipmappedArray = __safe_dlsym(mrcudaSym->handle, "cudaFreeMipmappedArray");
    mrcudaSym->mrcudaHostAlloc = __safe_dlsym(mrcudaSym->handle, "cudaHostAlloc");
    mrcudaSym->mrcudaHostRegister = __safe_dlsym(mrcudaSym->handle, "cudaHostRegister");
    mrcudaSym->mrcudaHostUnregister = __safe_dlsym(mrcudaSym->handle, "cudaHostUnregister");
    mrcudaSym->mrcudaHostGetDevicePointer = __safe_dlsym(mrcudaSym->handle, "cudaHostGetDevicePointer");
    mrcudaSym->mrcudaHostGetFlags = __safe_dlsym(mrcudaSym->handle, "cudaHostGetFlags");
    mrcudaSym->mrcudaMalloc3D = __safe_dlsym(mrcudaSym->handle, "cudaMalloc3D");
    mrcudaSym->mrcudaMalloc3DArray = __safe_dlsym(mrcudaSym->handle, "cudaMalloc3DArray");
    mrcudaSym->mrcudaMallocMipmappedArray = __safe_dlsym(mrcudaSym->handle, "cudaMallocMipmappedArray");
    mrcudaSym->mrcudaGetMipmappedArrayLevel = __safe_dlsym(mrcudaSym->handle, "cudaGetMipmappedArrayLevel");
    mrcudaSym->mrcudaMemcpy3D = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy3D");
    mrcudaSym->mrcudaMemcpy3DPeer = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy3DPeer");
    mrcudaSym->mrcudaMemcpy3DAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy3DAsync");
    mrcudaSym->mrcudaMemcpy3DPeerAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy3DPeerAsync");
    mrcudaSym->mrcudaMemGetInfo = __safe_dlsym(mrcudaSym->handle, "cudaMemGetInfo");
    mrcudaSym->mrcudaArrayGetInfo = __safe_dlsym(mrcudaSym->handle, "cudaArrayGetInfo");
    mrcudaSym->mrcudaMemcpy = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy");
    mrcudaSym->mrcudaMemcpyPeer = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyPeer");
    mrcudaSym->mrcudaMemcpyToArray = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyToArray");
    mrcudaSym->mrcudaMemcpyFromArray = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyFromArray");
    mrcudaSym->mrcudaMemcpyArrayToArray = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyArrayToArray");
    mrcudaSym->mrcudaMemcpy2D = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2D");
    mrcudaSym->mrcudaMemcpy2DToArray = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2DToArray");
    mrcudaSym->mrcudaMemcpy2DFromArray = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2DFromArray");
    mrcudaSym->mrcudaMemcpy2DArrayToArray = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2DArrayToArray");
    mrcudaSym->mrcudaMemcpyToSymbol = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyToSymbol");
    mrcudaSym->mrcudaMemcpyFromSymbol = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyFromSymbol");
    mrcudaSym->mrcudaMemcpyAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyAsync");
    mrcudaSym->mrcudaMemcpyPeerAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyPeerAsync");
    mrcudaSym->mrcudaMemcpyToArrayAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyToArrayAsync");
    mrcudaSym->mrcudaMemcpyFromArrayAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyFromArrayAsync");
    mrcudaSym->mrcudaMemcpy2DAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2DAsync");
    mrcudaSym->mrcudaMemcpy2DToArrayAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2DToArrayAsync");
    mrcudaSym->mrcudaMemcpy2DFromArrayAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpy2DFromArrayAsync");
    mrcudaSym->mrcudaMemcpyToSymbolAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyToSymbolAsync");
    mrcudaSym->mrcudaMemcpyFromSymbolAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemcpyFromSymbolAsync");
    mrcudaSym->mrcudaMemset = __safe_dlsym(mrcudaSym->handle, "cudaMemset");
    mrcudaSym->mrcudaMemset2D = __safe_dlsym(mrcudaSym->handle, "cudaMemset2D");
    mrcudaSym->mrcudaMemset3D = __safe_dlsym(mrcudaSym->handle, "cudaMemset3D");
    mrcudaSym->mrcudaMemsetAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemsetAsync");
    mrcudaSym->mrcudaMemset2DAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemset2DAsync");
    mrcudaSym->mrcudaMemset3DAsync = __safe_dlsym(mrcudaSym->handle, "cudaMemset3DAsync");
    mrcudaSym->mrcudaGetSymbolAddress = __safe_dlsym(mrcudaSym->handle, "cudaGetSymbolAddress");
    mrcudaSym->mrcudaGetSymbolSize = __safe_dlsym(mrcudaSym->handle, "cudaGetSymbolSize");
    mrcudaSym->mrcudaPointerGetAttributes = __safe_dlsym(mrcudaSym->handle, "cudaPointerGetAttributes");
    mrcudaSym->mrcudaDeviceCanAccessPeer = __safe_dlsym(mrcudaSym->handle, "cudaDeviceCanAccessPeer");
    mrcudaSym->mrcudaDeviceEnablePeerAccess = __safe_dlsym(mrcudaSym->handle, "cudaDeviceEnablePeerAccess");
    mrcudaSym->mrcudaDeviceDisablePeerAccess = __safe_dlsym(mrcudaSym->handle, "cudaDeviceDisablePeerAccess");
    /*mrcudaSym->mrcudaGraphicsUnregisterResource = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsUnregisterResource");
    mrcudaSym->mrcudaGraphicsResourceSetMapFlags = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsResourceSetMapFlags");
    mrcudaSym->mrcudaGraphicsMapResources = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsMapResources");
    mrcudaSym->mrcudaGraphicsUnmapResources = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsUnmapResources");
    mrcudaSym->mrcudaGraphicsResourceGetMappedPointer = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsResourceGetMappedPointer");
    mrcudaSym->mrcudaGraphicsSubResourceGetMappedArray = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsSubResourceGetMappedArray");
    mrcudaSym->mrcudaGraphicsResourceGetMappedMipmappedArray = __safe_dlsym(mrcudaSym->handle, "cudaGraphicsResourceGetMappedMipmappedArray");*/
    mrcudaSym->mrcudaGetChannelDesc = __safe_dlsym(mrcudaSym->handle, "cudaGetChannelDesc");
    mrcudaSym->mrcudaCreateChannelDesc = __safe_dlsym(mrcudaSym->handle, "cudaCreateChannelDesc");
    mrcudaSym->mrcudaBindTexture = __safe_dlsym(mrcudaSym->handle, "cudaBindTexture");
    mrcudaSym->mrcudaBindTexture2D = __safe_dlsym(mrcudaSym->handle, "cudaBindTexture2D");
    mrcudaSym->mrcudaBindTextureToArray = __safe_dlsym(mrcudaSym->handle, "cudaBindTextureToArray");
    mrcudaSym->mrcudaBindTextureToMipmappedArray = __safe_dlsym(mrcudaSym->handle, "cudaBindTextureToMipmappedArray");
    mrcudaSym->mrcudaUnbindTexture = __safe_dlsym(mrcudaSym->handle, "cudaUnbindTexture");
    mrcudaSym->mrcudaGetTextureAlignmentOffset = __safe_dlsym(mrcudaSym->handle, "cudaGetTextureAlignmentOffset");
    mrcudaSym->mrcudaGetTextureReference = __safe_dlsym(mrcudaSym->handle, "cudaGetTextureReference");
    mrcudaSym->mrcudaBindSurfaceToArray = __safe_dlsym(mrcudaSym->handle, "cudaBindSurfaceToArray");
    mrcudaSym->mrcudaGetSurfaceReference = __safe_dlsym(mrcudaSym->handle, "cudaGetSurfaceReference");
    mrcudaSym->mrcudaCreateTextureObject = __safe_dlsym(mrcudaSym->handle, "cudaCreateTextureObject");
    mrcudaSym->mrcudaDestroyTextureObject = __safe_dlsym(mrcudaSym->handle, "cudaDestroyTextureObject");
    mrcudaSym->mrcudaGetTextureObjectResourceDesc = __safe_dlsym(mrcudaSym->handle, "cudaGetTextureObjectResourceDesc");
    mrcudaSym->mrcudaGetTextureObjectTextureDesc = __safe_dlsym(mrcudaSym->handle, "cudaGetTextureObjectTextureDesc");
    mrcudaSym->mrcudaGetTextureObjectResourceViewDesc = __safe_dlsym(mrcudaSym->handle, "cudaGetTextureObjectResourceViewDesc");
    mrcudaSym->mrcudaCreateSurfaceObject = __safe_dlsym(mrcudaSym->handle, "cudaCreateSurfaceObject");
    mrcudaSym->mrcudaDestroySurfaceObject = __safe_dlsym(mrcudaSym->handle, "cudaDestroySurfaceObject");
    mrcudaSym->mrcudaGetSurfaceObjectResourceDesc = __safe_dlsym(mrcudaSym->handle, "cudaGetSurfaceObjectResourceDesc");
    mrcudaSym->mrcudaDriverGetVersion = __safe_dlsym(mrcudaSym->handle, "cudaDriverGetVersion");
    mrcudaSym->mrcudaRuntimeGetVersion = __safe_dlsym(mrcudaSym->handle, "cudaRuntimeGetVersion");
    mrcudaSym->mrcudaGetExportTable = __safe_dlsym(mrcudaSym->handle, "cudaGetExportTable");
    mrcudaSym->__mrcudaRegisterFatBinary = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterFatBinary");
    mrcudaSym->__mrcudaUnregisterFatBinary = __safe_dlsym(mrcudaSym->handle, "__cudaUnregisterFatBinary");
    mrcudaSym->__mrcudaRegisterVar = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterVar");
    mrcudaSym->__mrcudaRegisterTexture = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterTexture");
    mrcudaSym->__mrcudaRegisterSurface = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterSurface");
    mrcudaSym->__mrcudaRegisterFunction = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterFunction");
    //mrcudaSym->__mrcudaRegisterShared = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterShared");
    //mrcudaSym->__mrcudaRegisterSharedVar = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterSharedVar");
    //mrcudaSym->__mrcudaSynchronizeThreads = __safe_dlsym(mrcudaSym->handle, "__cudaSynchronizeThreads");
    //mrcudaSym->__mrcudaTextureFetch = __safe_dlsym(mrcudaSym->handle, "__cudaTextureFetch");
    //mrcudaSym->__mrcudaMutexOperation = __safe_dlsym(mrcudaSym->handle, "__cudaMutexOperation");
    //mrcudaSym->__mrcudaRegisterDeviceFunction = __safe_dlsym(mrcudaSym->handle, "__cudaRegisterDeviceFunction");
}

/**
 * Initialize mrCUDA.
 * Print error and terminate the program if an error occurs.
 */
__attribute__((constructor))
void mrcuda_init()
{
    char *switch_threshold;
    char *endptr;
    if(mrcudaState == MRCUDA_STATE_UNINITIALIZED)
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

        if((switch_threshold = getenv(__SWITCH_THRESHOLD_ENV_NAME__)) == NULL)
            switch_threshold = "RCUDA";

        if(strcmp(switch_threshold, "RCUDA") == 0)
        {
            mrcudaNumLaunchSwitchThreashold = -1;
            mrcudaState = MRCUDA_STATE_RUNNING_RCUDA;
        }
        else if(strcmp(switch_threshold, "NVIDIA") == 0)
        {
            mrcudaNumLaunchSwitchThreashold = -1;
            mrcudaState = MRCUDA_STATE_RUNNING_NVIDIA;
        }
        else
        {
            mrcudaState = MRCUDA_STATE_RUNNING_RCUDA;
            mrcudaNumLaunchSwitchThreashold = strtol(switch_threshold, &endptr, 10);
            if(*endptr != '\0')
                REPORT_ERROR_AND_EXIT("%s's value is not valid.\n", __SWITCH_THRESHOLD_ENV_NAME__);
        }


        // Allocate space for global variables.
        mrcudaSymRCUDA = malloc(sizeof(MRCUDASym));
        if(mrcudaSymRCUDA == NULL)
            REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymRCUDA.\n");

        mrcudaSymNvidia = malloc(sizeof(MRCUDASym));
        if(mrcudaSymNvidia == NULL)
            REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymNvidia.\n");

        // Assign the appropriate library to mrcudaSymDefault.
        if(mrcudaState == MRCUDA_STATE_RUNNING_RCUDA)
            mrcudaSymDefault = mrcudaSymRCUDA;
        else
            mrcudaSymDefault = mrcudaSymNvidia;


        // Create handles for CUDA libraries.
        mrcudaSymNvidia->handle = dlopen(__nvidiaLibPath, RTLD_NOW | RTLD_GLOBAL);
        if(mrcudaSymNvidia->handle == NULL)
            REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymNvidia.\n");

        mrcudaSymRCUDA->handle = dlopen(__rCUDALibPath, RTLD_NOW | RTLD_GLOBAL);
        if(mrcudaSymRCUDA->handle == NULL)
            REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymRCUDA.\n");


        // Symlink rCUDA and CUDA functions.
        __symlink_handle(mrcudaSymRCUDA);
        __symlink_handle(mrcudaSymNvidia);

        // Initialize the record/replay module.
        mrcuda_record_init();

        // Initialize mutex for switching locking.
        pthread_mutex_init(&__processing_func_mutex, NULL);

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
    if(mrcudaState > MRCUDA_STATE_UNINITIALIZED && mrcudaState < MRCUDA_STATE_FINALIZED)
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

        mrcudaState = MRCUDA_STATE_FINALIZED;
    }
}

/**
 * Switch from rCUDA to native.
 */
void mrcuda_switch()
{
    struct timeval start_switching_time, stop_switching_time;
    int num_replay = 0;

    MRecord *record = NULL;
    int already_mock_stream = 0;
    if(mrcudaState == MRCUDA_STATE_RUNNING_RCUDA)
    {
        DPRINTF("ENTER mrcuda_switch.\n");

        gettimeofday(&start_switching_time, NULL);
        mrcuda_function_call_lock();
        
        mrcudaSymRCUDA->mrcudaDeviceSynchronize();
        record = mrcudaRecordHeadPtr;
        while(record != NULL)
        {
            if(!already_mock_stream && !(record->skip_mock_stream))
            {
                mrcuda_simulate_stream();
                already_mock_stream = !already_mock_stream;
            }
            record->replayFunc(record);
            record = record->next;
            num_replay += 1;
        }
        mrcuda_sync_mem();
        mrcudaSymDefault = mrcudaSymNvidia;
        mrcudaState = MRCUDA_STATE_RUNNING_NVIDIA;

        mrcuda_function_call_release();
        gettimeofday(&stop_switching_time, NULL);
        DPRINTF("EXIT mrcuda_switch.\n");

        fprintf(stderr, "mrcuda_switch: num_replay: %d\n", num_replay);
        fprintf(stderr, "mrcuda_switch: time: %.6f\n",
            (stop_switching_time.tv_sec + (double)stop_switching_time.tv_usec / 1000000.0f) - (start_switching_time.tv_sec + (double)start_switching_time.tv_usec / 1000000.0f)
        );
    }
}

/**
 * Create a barrier such that subsequent calls are blocked until the barrier is released.
 */
void mrcuda_function_call_lock()
{
    if(mrcudaState < MRCUDA_STATE_FINALIZED && pthread_mutex_lock(&__processing_func_mutex) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __processing_func_mutex locking process.\n");
}

/**
 * Release the barrier; thus, allow subsequent calls to be processed normally.
 */
void mrcuda_function_call_release()
{
    if(mrcudaState < MRCUDA_STATE_FINALIZED && pthread_mutex_unlock(&__processing_func_mutex) != 0)
        REPORT_ERROR_AND_EXIT("Encounter an error during __processing_func_mutex unlocking process.\n");
}

