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
    (*handler)->__mrcudaRegisterFatBinary = mhelper_init_cudaRegisterFatBinary;
    return 0;

__mhelper_int_init_err_0:
    return -1;
}

/* Interfaces */

void **mhelper_init_cudaRegisterFatBinary(void *fatCubin)
{
    return mrcuda_get_current_gpu(mrcuda_get_current_gpu()->mhelperProcess, fatCubin);
}

void **mhelper_init_cudaRegisterFatBinary_internal(MHelperProcess_t *mhelperProcess, void *fatCubin)
{
    MHelperCommand_t command;
    MHelperResult_t result;
    void *addr;
    void **ret;

    __fatBinC_Wrapper_t *fatCubinWrapper = (__fatBinC_Wrapper_t *)(fatCubin);
    computeFatBinaryFormat_t fatCubinHeader = (computeFatBinaryFormat_t)(fatCubinWrapper->data);
    size_t fatCubinSize = (size_t)(sizeof(__fatBinC_Wrapper_t) + fatCubinHeader->headerSize + fatCubinHeader->fatSize);

    MRCUDASharedMemLocalInfo_t *sharedMemInfo = mhelper_mem_malloc(fatCubinSize);
    addr = sharedMemInfo->startAddr;
    addr = mempcpy(addr, fatCubin, sizeof(__fatBinC_Wraper_t));
    addr = mempcpy(addr, fatCubinWrapper->data, fatCubinSize - sizeof(__fatBinC_Wrapper_t));
    command.id = rand();
    command.type = MRCOMMAND_TYPE_CUDAREGISTERFATBINARY;
    command.command.cudaRegisterFatBinary.sharedMem = sharedMemInfo->sharedMem;
    mhelper_mem_detach(sharedMemInfo);
    result = mhelper_call(mhelperProcess, command);
    if (result.id == command.id && result.internalError == 0)
        ret = result.result.cudaRegisterFatBinary.fatCubinHandle;
    else
        ret = NULL;
    free(sharedMemInfo);
    return ret;
}

