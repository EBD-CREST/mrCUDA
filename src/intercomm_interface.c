#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <fatBinaryCtl.h>
#include <fatbinary.h>

#include "datatypes.h"
#include "intercomm_interface.h"
#include "intercomm.h"
#include "mrcuda.h"

/**
 * Call mhelper_mem_malloc.
 * Exit the program if the return is NULL.
 */
static inline MRCUDASharedMemLocalInfo_t *__mhelper_mem_malloc_safe(size_t size)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;
    if ((sharedMemInfo = mhelper_mem_malloc(size)) == NULL)
        REPORT_ERROR_AND_EXIT("Something went wrong in mhelper_mem_malloc.\n");
    return sharedMemInfo;
}

/**
 * Initialize a handler with a helper process.
 * @param handler output of initialized handler.
 * @param process a ptr to a helper process.
 * @return 0 on success; -1 otherwise.
 */
int mhelper_int_init(MRCUDASym_t **handler, MHelperProcess_t *process)
{
    if ((*handler = calloc(1, sizeof(MRCUDASym_t))) == NULL)
        goto __mhelper_int_init_err_0;
    (*handler)->handler.processHandler = process;
    (*handler)->__mrcudaRegisterFatBinary = mhelper_int_cudaRegisterFatBinary;
    return 0;

__mhelper_int_init_err_0:
    return -1;
}

/* Interfaces */

/**
 * Create a context on the helper process.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t a context will be created on.
 * @return 0 on success; -1 otherwise.
 */
int mhelper_int_cuCtxCreate_internal(MRCUDAGPU_t *mrcudaGPU)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUCTXCREATE;
    result = mhelper_call(mhelperProcess, command);
    return result.internalError == 0 && result.cudaError == cudaSuccess ? 0 : -1;
}

void **mhelper_int_cudaRegisterFatBinary(void *fatCubin)
{
    return mhelper_int_cudaRegisterFatBinary_internal(mrcuda_get_current_gpu(), fatCubin);
}

void **mhelper_int_cudaRegisterFatBinary_internal(MRCUDAGPU_t *mrcudaGPU, void *fatCubin)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    void *addr;
    void **ret;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    __fatBinC_Wrapper_t *fatCubinWrapper = (__fatBinC_Wrapper_t *)(fatCubin);
    computeFatBinaryFormat_t fatCubinHeader = (computeFatBinaryFormat_t)(fatCubinWrapper->data);
    size_t fatCubinSize = (size_t)(sizeof(__fatBinC_Wrapper_t) + fatCubinHeader->headerSize + fatCubinHeader->fatSize);

    MRCUDASharedMemLocalInfo_t *sharedMemInfo = __mhelper_mem_malloc_safe(fatCubinSize);
    addr = sharedMemInfo->startAddr;
    addr = mempcpy(addr, fatCubin, sizeof(__fatBinC_Wrapper_t));
    addr = mempcpy(addr, fatCubinWrapper->data, fatCubinSize - sizeof(__fatBinC_Wrapper_t));
    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAREGISTERFATBINARY;
    command.args.cudaRegisterFatBinary.sharedMem = sharedMemInfo->sharedMem;
    mhelper_mem_detach(sharedMemInfo);
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError == 0)
        ret = result.args.cudaRegisterFatBinary.fatCubinHandle;
    else
        ret = NULL;
    free(sharedMemInfo);
    return ret;
}

void mhelper_int_cudaUnregisterFatBinary(void **fatCubinHandle)
{
    mhelper_int_cudaUnregisterFatBinary_internal(mrcuda_get_current_gpu(), fatCubinHandle);
}

void mhelper_int_cudaUnregisterFatBinary_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAUNREGISTERFATBINARY;
    command.args.cudaUnregisterFatBinary.fatCubinHandle = fatCubinHandle;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during __cudaUnregisterFatBinary execution.\n");
}

void mhelper_int_cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global)
{
    mhelper_int_cudaRegisterVar_internal(
        mrcuda_get_current_gpu(),
        fatCubinHandle,
        hostVar,
        deviceAddress,
        deviceName,
        ext,
        size,
        constant,
        global
    );
}

void mhelper_int_cudaRegisterVar_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;
    size_t hostVarSize = strlen(hostVar) + 1;
    size_t deviceAddressSize = strlen(deviceAddress) + 1;
    size_t deviceNameSize = strlen(deviceName) + 1
    MRCUDASharedMemLocalInfo_t *sharedMemInfo = __mhelper_mem_malloc_safe(hostVarSize + deviceAddressSize + deviceNameSize);
    void *addr = sharedMemInfo->startAddr;

    addr = mempcpy(addr, hostVar, hostVarSize);
    addr = mempcpy(addr, deviceAddress, deviceAddressSize);
    addr = mempcpy(addr, deviceName, deviceNameSize);

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAREGISTERVAR;
    command.args.cudaRegisterVar.fatCubinHandle = fatCubinHandle;
    command.args.cudaRegisterVar.hostVar.offset = 0;
    command.args.cudaRegisterVar.deviceAddress.offset = hostVarSize;
    command.args.cudaRegisterVar.deviceName.offset = hostVarSize + deviceAddressSize;
    command.args.cudaRegisterVar.ext = ext;
    command.args.cudaRegisterVar.size = size;
    command.args.cudaRegisterVar.constant = constant
    command.args.cudaRegisterVar.global = global;
    command.args.cudaRegisterVar.shminfo = sharedMemInfo->sharedMem;
    mhelper_mem_detach(sharedMemInfo);
    result = mhelper_call(mhelperProcess, command);
    free(sharedMemInfo);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during __cudaRegisterVar execution.\n");
}

void mhelper_int_cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext)
{
    mhelper_int_cudaRegisterTexture_internal(
        mrcuda_get_current_gpu(),
        fatCubinHandle,
        hostVar,
        deviceAddress,
        deviceName,
        dim,
        norm,
        ext
    );
}

void mhelper_int_cudaRegisterTexture_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;
    size_t hostVarSize = sizeof(struct textureReference);
    size_t deviceNameSize = strlen(deviceName) + 1;
    MRCUDASharedMemLocalInfo_t *sharedMemInfo = __mhelper_mem_malloc_safe(hostVarSize + deviceNameSize);
    void *addr = sharedMemInfo->startAddr;

    addr = mempcpy(addr, hostVar, hostVarSize);
    addr = mempcpy(addr, deviceName, deviceNameSize);

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAREGISTERTEXTURE;
    command.args.cudaRegisterTexture.fatCubinHandle = fatCubinHandle;
    command.args.cudaRegisterTexture.hostVar.offset = 0;
    command.args.cudaRegisterTexture.deviceAddress = deviceAddress;
    command.args.cudaRegisterTexture.deviceName.offset = hostVarSize;
    command.args.cudaRegisterTexture.dim = dim;
    command.args.cudaRegisterTexture.norm = norm;
    command.args.cudaRegisterTexture.ext = ext;
    command.args.cudaRegisterTexture.sharedMem = sharedMemInfo->sharedMem;
    mhelper_mem_detach(sharedMemInfo);
    result = mhelper_call(mhelperProcess, command);
    free(sharedMemInfo);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during __cudaRegisterTexture execution.\n");
}

void mhelper_int_cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    mhelper_int_cudaRegisterFunction_internal(
        mrcuda_get_current_gpu(),
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
}

void mhelper_int_cudaRegisterFunction_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;
    size_t hostFunSize = strlen(hostFun) + 1;
    size_t deviceFunSize = strlen(deviceFun) + 1;
    size_t deviceNameSize = strlen(deviceName) + 1;
    MRCUDASharedMemLocalInfo_t *sharedMemInfo = __mhelper_mem_malloc_safe(hostFunSize + deviceFunSize + deviceNameSize);
    void *addr = sharedMemInfo->startAddr;

    addr = mempcpy(addr, hostFun, hostFunSize);
    addr = mempcpy(addr, deviceFun, deviceFunSize);
    addr = mempcpy(addr, deviceName, deviceNameSize);

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAREGISTERFUNCTION;
    command.args.cudaRegisterFunction.fatCubinHandle = fatCubinHandle;
    command.args.cudaRegisterFunction.hostFun.offset = 0;
    command.args.cudaRegisterFunction.deviceFun.offset = hostFunSize;
    command.args.cudaRegisterFunction.deviceName.offset = hostFunSize + deviceFunSize;
    command.args.cudaRegisterFunction.thread_limit = thread_limit;
    command.args.cudaRegisterFunction.tid = tid;
    command.args.cudaRegisterFunction.bid = bid;
    command.args.cudaRegisterFunction.bDim = bDim;
    command.args.cudaRegisterFunction.gDim = gDim;
    command.args.cudaRegisterFunction.wSize = wSize;
    command.args.cudaRegisterFunction.sharedMem = sharedMemInfo->sharedMem;
    mhelper_mem_detach(sharedMemInfo);
    result = mhelper_call(mhelperProcess, command);
    free(sharedMemInfo);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during __cudaRegisterFunction execution.\n");
}

cudaError_t mhelper_int_cudaLaunch(const void *func)
{
    return mhelper_int_cudaLaunch_internal(mrcuda_get_current_gpu(), func);
}

cudaError_t mhelper_int_cudaLaunch_internal(MRCUDAGPU_t *mrcudaGPU, const void *func)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDALAUNCH;
    command.args.cudaLaunch.func = func;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during cudaLaunch execution.\n");
    return result.cudaError;
}

/**
 * Internally calls malloc.
 * There is no point in calling the actual cudaHostAlloc on the mhelper since the malloced memory cannot be accessed across processes.
 */
cudaError_t mhelper_int_cudaHostAlloc(void **pHost,  size_t size,  unsigned int flags)
{
    if (flags != cudaHostAllocDefault)
        return cudaErrorMemoryAllocation;
    *pHost = malloc(size);
    return cudaSuccess;
}

cudaError_t mhelper_int_cudaDeviceReset(void)
{
    return mhelper_int_cudaDeviceReset_internal(mrcuda_get_current_gpu());
}

cudaError_t mhelper_int_cudaDeviceReset_internal(MRCUDAGPU_t *mrcudaGPU)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDADEVICERESET;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during cudaDeviceReset execution.\n");
    return result.cudaError;
}

cudaError_t mhelper_int_cudaDeviceSynchronize(void)
{
    return mhelper_int_cudaDeviceSynchronize_internal(mrcuda_get_current_gpu());
}

cudaError_t mhelper_int_cudaDeviceSynchronize_internal(MRCUDAGPU_t *mrcudaGPU)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDADEVICESYNCHRONIZE;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during cudaDeviceSynchronize execution.\n");
    return result.cudaError;
}

cudaError_t mhelper_int_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    return mhelper_int_cudaGetDeviceProperties_internal(
        mrcuda_get_current_gpu(),
        prop,
        device
    );
}

cudaError_t mhelper_int_cudaGetDeviceProperties_internal(MRCUDAGPU_t *mrcudaGPU, struct cudaDeviceProp *prop, int device)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAGETDEVICEPROPERTIES;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during cudaDeviceSynchronize execution.\n");
    memcpy(prop, &result.args.cudaGetDeviceProperties.prop, sizeof(struct cudaDeviceProp));
    return result.cudaError;
}

cudaError_t mhelper_int_cudaMalloc(void **devPtr, size_t size)
{
    return mhelper_int_cudaMalloc_internal(
        mrcuda_get_current_gpu(),
        devPtr,
        size
    );
}

cudaError_t mhelper_int_cudaMalloc_internal(MRCUDAGPU_t *mrcudaGPU, void **devPtr, size_t size)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAMALLOC;
    command.args.cudaMalloc.size = size;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during cudaMalloc execution.\n");
    *devPtr = result.args.cudaMalloc.devPtr;
    return result.cudaError;
}

/**
 * Internally calls free since cudaMallocHost uses malloc.
 */
cudaError_t mhelper_int_cudaFreeHost(void *ptr)
{
    free(ptr);
    return cudaSuccess;
}

cudaError_t mhelper_int_cudaFree(void *devPtr)
{
    return mhelper_int_cudaFree_internal(
        mrcuda_get_current_gpu(),
        devPtr
    );
}

cudaError_t mhelper_int_cudaFree_internal(MRCUDAGPU_t *mrcudaGPU, void *devPtr)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    MHelperProcess_t *mhelperProcess = mrcudaGPU->mhelperProcess;

    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAFREE;
    command.args.cudaFree.devPtr = devPtr;
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError != 0)
        REPORT_ERROR_AND_EXIT("mhelper encountered an error during cudaFree execution.\n");
    return result.cudaError;
}

cudaError_t mhelper_int_cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t mhelper_int_cudaMemcpyToSymbolAsync_internal(MRCUDAGPU_t *mrcudaGPU, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t mhelper_int_cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t mhelper_int_cudaMemcpyFromSymbolAsync_internal(MRCUDAGPU_t *mrcudaGPU, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t mhelper_int_cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t mhelper_int_cudaSetupArgument_internal(MRCUDAGPU_t *mrcudaGPU, const void *arg, size_t size, size_t offset);

cudaError_t mhelper_int_cudaStreamSynchronize(cudaStream_t stream);
cudaError_t mhelper_int_cudaStreamSynchronize_internal(MRCUDAGPU_t *mrcudaGPU, cudaStream_t stream);

cudaError_t mhelper_int_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t mhelper_int_cudaConfigureCall_internal(MRCUDAGPU_t *mrcudaGPU, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);

cudaError_t mhelper_int_cudaGetLastError(void);
cudaError_t mhelper_int_cudaGetLastError_internal(MRCUDAGPU_t *mrcudaGPU);

cudaError_t mhelper_int_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t mhelper_int_cudaMemcpy_internal(MRCUDAGPU_t *mrcudaGPU, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
