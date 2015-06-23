#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fatBinaryCtl.h>

#include "common.h"
#include "datatypes.h"
#include "intercomm_mem.h"

extern void** CUDARTAPI __cudaRegisterFatBinary(
  void *fatCubin
);

static int gpuID;

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
    fatCubinHandle = __cudaRegisterFatBinary(fatCubinWrapper);
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
    __cudaUnregisterFatBinary(command.args.cudaUnregisterFatBinary.fatCubinHandle);
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
    __cudaRegisterFunction(
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
    result->cudaError = cudaLaunch(command.args.cudaLaunch.func);

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
    result->cudaError = cudaDeviceSynchronize();

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
    result->cudaError = cudaMalloc(&(result->args.cudaMalloc.devPtr), command.args.cudaMalloc.size);

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
    result->cudaError = cudaFree(command.args.cudaFree.devPtr);

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
    result->cudaError = cudaSetupArgument(
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
    result->cudaError = cudaConfigureCall(
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
    result->cudaError = cudaGetLastError();

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
        if ((sharedMemInfo = mhelper_mem_get(command.args.cudaSetupArgument.sharedMem)) == NULL)
            goto __exec_cudaMemcpy_err_0;
        result->cudaError = cudaMemcpy(
            command.args.cudaMemcpy.dst,
            sharedMemInfo->startAddr,
            command.args.cudaMemcpy.count,
            command.args.cudaMemcpy.kind
        );
        mhelper_mem_free(sharedMemInfo);
        result->internalError = 0;
    }
    else if (command.args.cudaMemcpy.kind == cudaMemcpyDeviceToHost) {
        if ((sharedMemInfo = mhelper_mem_malloc(command.args.cudaMemcpy.count)) == NULL)
            goto __exec_cudaMemcpy_err_1;
        result->cudaError = cudaMemcpy(
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
    size_t remainingSize = sizeof(MHelperCommand_t);
    char *buf = (char *)command;

    while (remainingSize > 0) {
        n = fread(buf, remainingSize, 1, stdin);
        if (n < 0)
            goto __receive_command_err_0;
        remainingSize -= n;
        buf += n;
    }
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
    size_t remainingSize = sizeof(MHelperResult_t);
    char *buf = (char *)&result;

    while (remainingSize > 0) {
        n = fwrite(buf, remainingSize, 1, stdout);
        if (n < 0)
            goto __sendback_result_err_0;
        remainingSize -= n;
        buf += n;
    }

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

    if (argc != 2)
        REPORT_ERROR_AND_EXIT("The number of arguments should be exactly two.\n");
    gpuID = (int)strtol(argv[1], &endptr, 10);
    if (*endptr != '\0')
        REPORT_ERROR_AND_EXIT("The GPU ID argument is invalid.\n");

    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        REPORT_ERROR_AND_EXIT("Cannot register the signal handler.\n");

    run_forever();
    return 0;
}

