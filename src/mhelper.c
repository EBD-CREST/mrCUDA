#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fatBinaryCtl.h>
#include <dlfcn.h>
#include <string.h>

#include "common.h"
#include "datatypes.h"
#include "intercomm_mem.h"

#define __NVIDIA_LIBRARY_PATH_ENV_NAME__ "MRCUDA_NVIDIA_LIB_PATH"

MRCUDASym_t *mrcudaSymNvidia;
static int gpuID;

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

extern void** CUDARTAPI __cudaRegisterFatBinary(
  void *fatCubin
);

/**
 * Execute cuCtxCreate command.
 * @param command command information.
 * @param result output result.
 * @return 0 always.
 */
static int exec_cuCtxCreate(MHelperCommand_t command, MHelperResult_t *result)
{
    CUcontext pctx;

    result->cudaError = cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO, gpuID) == CUDA_SUCCESS ? cudaSuccess : cudaErrorApiFailureBase;
    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaSetDevice command.
 * @param command command information.
 * @param result output result.
 * @return 0 always.
 */
static int exec_cudaSetDevice(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaSetDevice(command.args.cudaSetDevice.device);
    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute __cudaRegisterFatBinary command.
 * @param command command information.
 * @param result output result.
 * @return 0 on success; -1 otherwise.
 */
static int exec_cudaRegisterFatBinary(MHelperCommand_t command, MHelperResult_t *result)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;
    void **fatCubinHandle;
    __fatBinC_Wrapper_t *fatCubinWrapper;

    if ((sharedMemInfo = mhelper_mem_get(command.args.cudaRegisterFatBinary.sharedMem)) == NULL)
        goto __exec_cudaRegisterFatBinary_err_0;
    fatCubinWrapper = sharedMemInfo->startAddr;
    fatCubinWrapper->data = sharedMemInfo->startAddr + sizeof(__fatBinC_Wrapper_t);
    fatCubinHandle = mrcudaSymNvidia->__mrcudaRegisterFatBinary(fatCubinWrapper);
    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    result->cudaError = cudaSuccess;
    result->args.cudaRegisterFatBinary.fatCubinHandle = fatCubinHandle;
    mhelper_mem_free(sharedMemInfo);
    return 0;

__exec_cudaRegisterFatBinary_err_0:
    return -1;
}

/**
 * Execute __cudaUnregisterFatBinary command.
 * @param command command information.
 * @param result output result.
 * @return 0 always.
 */
static int exec_cudaUnregisterFatBinary(MHelperCommand_t command, MHelperResult_t *result)
{
    mrcudaSymNvidia->__mrcudaUnregisterFatBinary(command.args.cudaUnregisterFatBinary.fatCubinHandle);
    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    result->cudaError = cudaSuccess;
    return 0;
}

/**
 * Execute __cudaRegisterFunction command.
 * @param command command information.
 * @param result output result.
 * @return 0 on success; -1 otherwise.
 */
static int exec_cudaRegisterFunction(MHelperCommand_t command, MHelperResult_t *result)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;

    if ((sharedMemInfo = mhelper_mem_get(command.args.cudaRegisterFunction.sharedMem)) == NULL)
        goto __exec_cudaRegisterFunction_err_0;
    mrcudaSymNvidia->__mrcudaRegisterFunction(
        command.args.cudaRegisterFunction.fatCubinHandle,
        (char *)(sharedMemInfo->startAddr + command.args.cudaRegisterFunction.hostFun.offset),
        (char *)(sharedMemInfo->startAddr + command.args.cudaRegisterFunction.deviceFun.offset),
        (char *)(sharedMemInfo->startAddr + command.args.cudaRegisterFunction.deviceName.offset),
        command.args.cudaRegisterFunction.thread_limit,
        command.args.cudaRegisterFunction.tid,
        command.args.cudaRegisterFunction.bid,
        command.args.cudaRegisterFunction.bDim,
        command.args.cudaRegisterFunction.gDim,
        command.args.cudaRegisterFunction.wSize
    );
    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    result->cudaError = cudaSuccess;
    mhelper_mem_free(sharedMemInfo);
    return 0;

__exec_cudaRegisterFunction_err_0:
    return -1;
}

/**
 * Execute cudaLaunch command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaLaunch(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaLaunch(command.args.cudaLaunch.func);

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaDeviceSynchronize command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaDeviceSynchronize(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaDeviceSynchronize();

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaMalloc command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaMalloc(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaMalloc(&(result->args.cudaMalloc.devPtr), command.args.cudaMalloc.size);

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaFree command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaFree(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaFree(command.args.cudaFree.devPtr);

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaSetupArgument command.
 * @param command command information.
 * @param result output result.
 * @return 0 on success; -1 otherwise.
 */
static int exec_cudaSetupArgument(MHelperCommand_t command, MHelperResult_t *result)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;

    if ((sharedMemInfo = mhelper_mem_get(command.args.cudaSetupArgument.sharedMem)) == NULL)
        goto __exec_cudaSetupArgument_err_0;
    result->cudaError = mrcudaSymNvidia->mrcudaSetupArgument(
        sharedMemInfo->startAddr,
        command.args.cudaSetupArgument.size,
        command.args.cudaSetupArgument.offset
    );
    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    mhelper_mem_free(sharedMemInfo);
    return 0;

__exec_cudaSetupArgument_err_0:
    return -1;
}

/**
 * Execute cudaConfigureCall command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaConfigureCall(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaConfigureCall(
        command.args.cudaConfigureCall.gridDim,
        command.args.cudaConfigureCall.blockDim,
        command.args.cudaConfigureCall.sharedMem,
        command.args.cudaConfigureCall.stream
    );

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaGetLastError command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaGetLastError(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaGetLastError();

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Execute cudaMemcpy command.
 * @param command command information.
 * @param result output result.
 * @return 0 on success; -1 otherwise.
 */
static int exec_cudaMemcpy(MHelperCommand_t command, MHelperResult_t *result)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;

    if (command.args.cudaMemcpy.kind == cudaMemcpyHostToDevice) {
        if ((sharedMemInfo = mhelper_mem_get(command.args.cudaMemcpy.sharedMem)) == NULL)
            goto __exec_cudaMemcpy_err_0;
        result->cudaError = mrcudaSymNvidia->mrcudaMemcpy(
            command.args.cudaMemcpy.dst,
            sharedMemInfo->startAddr,
            command.args.cudaMemcpy.count,
            command.args.cudaMemcpy.kind
        );
        mhelper_mem_detach(sharedMemInfo);
        mhelper_mem_free(sharedMemInfo);
        result->internalError = 0;
    }
    else if (command.args.cudaMemcpy.kind == cudaMemcpyDeviceToHost) {
        if ((sharedMemInfo = mhelper_mem_malloc(command.args.cudaMemcpy.count)) == NULL)
            goto __exec_cudaMemcpy_err_1;
        result->cudaError = mrcudaSymNvidia->mrcudaMemcpy(
            sharedMemInfo->startAddr,
            command.args.cudaMemcpy.src,
            command.args.cudaMemcpy.count,
            command.args.cudaMemcpy.kind
        );
        result->args.cudaMemcpy.sharedMem = sharedMemInfo->sharedMem;
        result->internalError = 0;
        mhelper_mem_detach(sharedMemInfo);
    }
    else
        result->internalError = -3;

    result->id = command.id;
    result->type = command.type;
    return 0;

__exec_cudaMemcpy_err_1:
    return -2;
__exec_cudaMemcpy_err_0:
    return -1;
}

/**
 * Execute cudaStreamCreate command.
 * @param command command information.
 * @param result output result.
 * @return 0 always
 */
static int exec_cudaStreamCreate(MHelperCommand_t command, MHelperResult_t *result)
{
    result->cudaError = mrcudaSymNvidia->mrcudaStreamCreate(&(result->args.cudaStreamCreate.stream));

    result->id = command.id;
    result->type = command.type;
    result->internalError = 0;
    return 0;
}

/**
 * Handle system signal.
 * @param signum the signal number caught.
 */
void sig_handler(int signum)
{
    exit(EXIT_SUCCESS);
}

/**
 * Waiting for a command.
 * @param command a ptr to a MHelperCommand_t that a successfully received command will be written to.
 * @return 0 on success; another number otherwise.
 */
static int receive_command(MHelperCommand_t *command)
{
    size_t n;
    char *buf = (char *)command;

    n = fread(buf, sizeof(MHelperCommand_t), 1, stdin);
    if (n != 1)
        goto __receive_command_err_0;
    return 0;

__receive_command_err_0:
    return -1;
}

/**
 * Execute the specified command.
 * Output the result of the execution through the result variable.
 * @param command the command to be executed.
 * @param result a ptr to a MHelperResult_t to be outputted to.
 * @return 0 on success; another number otherwise.
 */
static int execute_command(MHelperCommand_t command, MHelperResult_t *result)
{
    switch (command.type) {
        case MRCOMMAND_TYPE_CUCTXCREATE:
            return exec_cuCtxCreate(command, result);
        case MRCOMMAND_TYPE_CUDASETDEVICE:
            return exec_cudaSetDevice(command, result);
        case MRCOMMAND_TYPE_CUDAREGISTERFATBINARY:
            return exec_cudaRegisterFatBinary(command, result);
        case MRCOMMAND_TYPE_CUDAREGISTERFUNCTION:
            return exec_cudaRegisterFunction(command, result);
        case MRCOMMAND_TYPE_CUDALAUNCH:
            return exec_cudaLaunch(command, result);
        case MRCOMMAND_TYPE_CUDADEVICESYNCHRONIZE:
            return exec_cudaDeviceSynchronize(command, result);
        case MRCOMMAND_TYPE_CUDAMALLOC:
            return exec_cudaMalloc(command, result);
        case MRCOMMAND_TYPE_CUDAFREE:
            return exec_cudaFree(command, result);
        case MRCOMMAND_TYPE_CUDASETUPARGUMENT:
            return exec_cudaSetupArgument(command, result);
        case MRCOMMAND_TYPE_CUDACONFIGURECALL:
            return exec_cudaConfigureCall(command, result);
        case MRCOMMAND_TYPE_CUDAGETLASTERROR:
            return exec_cudaGetLastError(command, result);
        case MRCOMMAND_TYPE_CUDAMEMCPY:
            return exec_cudaMemcpy(command, result);
        case MRCOMMAND_TYPE_CUDASTREAMCREATE:
            return exec_cudaStreamCreate(command, result);
    }
    return -1;
}

/**
 * Return the result to the caller.
 * @param result the result to be returned.
 * @return 0 on success; another number otherwise.
 */
static int sendback_result(MHelperResult_t result)
{
    size_t n;
    char *buf = (char *)&result;

    n = fwrite(buf, sizeof(MHelperResult_t), 1, stdout);
    if (n != 1)
        goto __sendback_result_err_0;
    fflush(stdout);
    return 0;

__sendback_result_err_0:
    return -1;
}

/**
 * Start mhelper's command listening server.
 */
static void run_forever(void)
{
    MHelperCommand_t command;
    MHelperResult_t result;

    while (1) {
        if (receive_command(&command) != 0)
            continue;
        if (execute_command(command, &result) != 0) {
            result.id = command.id;
            result.type = command.type;
            result.internalError = -1;
            result.cudaError = cudaSuccess;
        }
        sendback_result(result);
    }
}

/**
 * Main function
 */
int main(int argc, char **argv)
{
    char *endptr;
    char *__nvidiaLibPath;

    if (argc < 2)
        REPORT_ERROR_AND_EXIT("The number of arguments should be two.\n");
    gpuID = (int)strtol(argv[1], &endptr, 10);
    if (*endptr != '\0')
        REPORT_ERROR_AND_EXIT("The GPU ID argument is invalid.\n");

    __nvidiaLibPath = getenv(__NVIDIA_LIBRARY_PATH_ENV_NAME__);
    if (__nvidiaLibPath == NULL || strlen(__nvidiaLibPath) == 0)
        REPORT_ERROR_AND_EXIT("%s is not specified.\n", __NVIDIA_LIBRARY_PATH_ENV_NAME__);

    if ((mrcudaSymNvidia = malloc(sizeof(MRCUDASym_t))) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate space for mrcudaSymNvidia.\n");
    
    // Create handles for CUDA libraries.
    mrcudaSymNvidia->handler.symHandler = dlopen(__nvidiaLibPath, RTLD_NOW | RTLD_GLOBAL);
    if (mrcudaSymNvidia->handler.symHandler == NULL)
        REPORT_ERROR_AND_EXIT("Cannot sym-link mrcudaSymNvidia.\n");

    __symlink_handle(mrcudaSymNvidia);

    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        REPORT_ERROR_AND_EXIT("Cannot register the signal handler.\n");

    run_forever();
    return 0;
}

