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

    MRCUDASharedMemLocalInfo_t *sharedMemInfo = mhelper_mem_malloc(fatCubinSize);
    addr = sharedMemInfo->startAddr;
    addr = mempcpy(addr, fatCubin, sizeof(__fatBinC_Wrapper_t));
    addr = mempcpy(addr, fatCubinWrapper->data, fatCubinSize - sizeof(__fatBinC_Wrapper_t));
    command.id = mhelper_generate_command_id(mrcudaGPU);
    command.type = MRCOMMAND_TYPE_CUDAREGISTERFATBINARY;
    command.command.cudaRegisterFatBinary.sharedMem = sharedMemInfo->sharedMem;
    mhelper_mem_detach(sharedMemInfo);
    result = mhelper_call(mhelperProcess, command);
    if (result.internalError == 0)
        ret = result.result.cudaRegisterFatBinary.fatCubinHandle;
    else
        ret = NULL;
    free(sharedMemInfo);
    return ret;
}

