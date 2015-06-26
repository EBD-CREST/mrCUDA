#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "datatypes.h"
#include "record.h"
#include "intercomm_mem.h"
#include "intercomm.h"
#include "mrcuda.h"

double recordAccTime = 0;
double memsyncAccTime = 0;
int memsyncNumCalls = 0;
double memsyncSize = 0;

MRecordGPU_t *mrecordGPUList = NULL;

/**
 * Allocate a new MRecord_t and appropriately link the new one with the previous.
 * The pointer to the new MRecord_t is recorded into recordPtr.
 * @param recordPtr the variable that the new MRecord_t will be recorded into.
 * @param mrcudaGPU a ptr to the called GPU.
 * @return 0 if success, otherwise error number.
 */
static int __mrcuda_record_new(MRecord_t **recordPtr, MRCUDAGPU_t *mrcudaGPU)
{
    if ((*recordPtr = calloc(1, sizeof(MRecord_t))) == NULL)
        goto __mrcuda_record_new_err_0;

    if (mrcudaGPU->mrecordGPU->mrcudaRecordHeadPtr == NULL)
        mrcudaGPU->mrecordGPU->mrcudaRecordHeadPtr = *recordPtr;
    else
        mrcudaGPU->mrecordGPU->mrcudaRecordTailPtr->next = *recordPtr;

    mrcudaGPU->mrecordGPU->mrcudaRecordTailPtr = *recordPtr;

    return 0;

__mrcuda_record_new_err_0:
    return -1;
}

/**
 * Call __mrcuda_record_new.
 * Exit and report error if found.
 * @param recordPtr the variable that the new MRecord_t will be recorded into.
 * @param mrcudaGPU a ptr to the called GPU.
 */
static inline void __mrcuda_record_new_safe(MRecord_t **recordPtr, MRCUDAGPU_t *mrcudaGPU)
{
    if (__mrcuda_record_new(recordPtr, mrcudaGPU) != 0)
        REPORT_ERROR_AND_EXIT("Cannot create a new MRecord.\n");
}

/**
 * Initialize the record/replay module.
 * Exit and report error if found.
 */
void mrcuda_record_init()
{
    int i;

    mrecordGPUList = calloc(mrcudaNumGPUs, sizeof(MRecordGPU_t));

    for (i = 0; i < mrcudaNumGPUs; i++) {
        if (mrcudaGPUList[i].status == MRCUDA_GPU_STATUS_RCUDA) {
            mrecordGPUList[i].mrcudaGPU = &(mrcudaGPUList[i]);
            mrcudaGPUList[i].mrecordGPU = &(mrecordGPUList[i]);
            mrecordGPUList[i].activeMemoryTable = g_hash_table_new(g_direct_hash, g_direct_equal);
            if(mrecordGPUList[i].activeMemoryTable == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate memory for activeMemoryTable of GPU #%d.\n", i);
            mrecordGPUList[i].activeSymbolTable = g_hash_table_new(g_direct_hash, g_direct_equal);
            if(mrecordGPUList[i].activeSymbolTable == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate memory for activeSymbolTable of GPU #%d.\n", i);
            mrecordGPUList[i].fatCubinHandleAddrTable = g_hash_table_new(g_direct_hash, g_direct_equal);
            if(mrecordGPUList[i].fatCubinHandleAddrTable == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate memory for fatCubinHandleAddrTable of GPU #%d.\n", i);
            mrecordGPUList[i].hostAllocTable = g_hash_table_new(g_direct_hash, g_direct_equal);
            if(mrecordGPUList[i].hostAllocTable == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate memory for hostAllocTable of GPU #%d.\n", i);
        }
    }
}

/**
 * Finalize the record/replay module.
 */
void mrcuda_record_fini()
{
    int i;
    if (mrecordGPUList != NULL) {
        for (i = 0; i < mrcudaNumGPUs; i++) {
            if (mrecordGPUList[i].activeMemoryTable)
                g_hash_table_destroy(mrecordGPUList[i].activeMemoryTable);
            if (mrecordGPUList[i].activeSymbolTable)
                g_hash_table_destroy(mrecordGPUList[i].activeSymbolTable);
            if (mrecordGPUList[i].fatCubinHandleAddrTable)
                g_hash_table_destroy(mrecordGPUList[i].fatCubinHandleAddrTable);
            if (mrecordGPUList[i].hostAllocTable)
                g_hash_table_destroy(mrecordGPUList[i].hostAllocTable);
        }
        free(mrecordGPUList);
    }
}

/*******************************************
 * Below this are record functions.        *
 *******************************************/

/**
 * Record a cudaRegisterFatBinary call.
 */
void mrcuda_record_cudaRegisterFatBinary(MRCUDAGPU_t *mrcudaGPU, void* fatCubin, void **fatCubinHandle)
{
    MRecord_t *recordPtr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    recordPtr->functionName = "__cudaRegisterFatBinary";
    recordPtr->skipMockStream = 1;
    recordPtr->data.cudaRegisterFatBinary.fatCubin = fatCubin;
    recordPtr->data.cudaRegisterFatBinary.fatCubinHandle = fatCubinHandle;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterFatBinary;

    g_hash_table_insert(mrcudaGPU->mrecordGPU->fatCubinHandleAddrTable, fatCubinHandle, &(recordPtr->data.cudaRegisterFatBinary.fatCubinHandle));
    
    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaRegisterFunction call.
 */
void mrcuda_record_cudaRegisterFunction(
    MRCUDAGPU_t *mrcudaGPU,
    void **fatCubinHandle, 
    const char *hostFun, 
    char *deviceFun, 
    const char *deviceName, 
    int thread_limit, 
    uint3 *tid, 
    uint3 *bid, 
    dim3 *bDim, 
    dim3 *gDim, 
    int *wSize
)
{
    MRecord_t *recordPtr;
    void ***fatCubinHandleAddr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    if ((fatCubinHandleAddr = g_hash_table_lookup(mrcudaGPU->mrecordGPU->fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaRegisterFunction";
    recordPtr->skipMockStream = 1;
    recordPtr->data.cudaRegisterFunction.fatCubinHandlePtr = fatCubinHandleAddr;
    recordPtr->data.cudaRegisterFunction.hostFun = hostFun;
    recordPtr->data.cudaRegisterFunction.deviceFun.ptr = deviceFun;
    recordPtr->data.cudaRegisterFunction.deviceName.ptr = deviceName;
    recordPtr->data.cudaRegisterFunction.thread_limit = thread_limit;
    recordPtr->data.cudaRegisterFunction.tid = tid;
    recordPtr->data.cudaRegisterFunction.bid = bid;
    recordPtr->data.cudaRegisterFunction.bDim = bDim;
    recordPtr->data.cudaRegisterFunction.gDim = gDim;
    recordPtr->data.cudaRegisterFunction.wSize = wSize;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterFunction;

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaRegisterVar call.
 */
void mrcuda_record_cudaRegisterVar(
    MRCUDAGPU_t *mrcudaGPU, 
    void **fatCubinHandle, 
    char *hostVar, 
    char *deviceAddress,
    const char *deviceName,
    int ext,
    int size,
    int constant,
    int global
)
{
    MRecord_t *recordPtr;
    void ***fatCubinHandleAddr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    if ((fatCubinHandleAddr = g_hash_table_lookup(mrcudaGPU->mrecordGPU->fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaRegisterVar";
    recordPtr->skipMockStream = 1;
    recordPtr->data.cudaRegisterVar.fatCubinHandlePtr = fatCubinHandleAddr;
    recordPtr->data.cudaRegisterVar.hostVar.ptr = hostVar;
    recordPtr->data.cudaRegisterVar.deviceAddress.ptr = deviceAddress;
    recordPtr->data.cudaRegisterVar.deviceName.ptr = deviceName;
    recordPtr->data.cudaRegisterVar.ext = ext;
    recordPtr->data.cudaRegisterVar.size = size;
    recordPtr->data.cudaRegisterVar.constant = constant;
    recordPtr->data.cudaRegisterVar.global = global;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterVar;

    g_hash_table_insert(mrcudaGPU->mrecordGPU->activeSymbolTable, hostVar, recordPtr);

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaRegisterTexture call.
 */
void mrcuda_record_cudaRegisterTexture(
    MRCUDAGPU_t *mrcudaGPU,
    void **fatCubinHandle,
    const struct textureReference *hostVar,
    const void **deviceAddress,
    const char *deviceName,
    int dim,
    int norm,
    int ext
)
{
    MRecord_t *recordPtr;
    void ***fatCubinHandleAddr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    if ((fatCubinHandleAddr = g_hash_table_lookup(mrcudaGPU->mrecordGPU->fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaRegisterTexture";
    recordPtr->skipMockStream = 1;
    recordPtr->data.cudaRegisterTexture.fatCubinHandlePtr = fatCubinHandleAddr;
    recordPtr->data.cudaRegisterTexture.hostVar.ptr = hostVar;
    recordPtr->data.cudaRegisterTexture.deviceAddress = deviceAddress;
    recordPtr->data.cudaRegisterTexture.deviceName.ptr = deviceName;
    recordPtr->data.cudaRegisterTexture.dim = dim;
    recordPtr->data.cudaRegisterTexture.norm = norm;
    recordPtr->data.cudaRegisterTexture.ext = ext;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterTexture;

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaUnregisterFatBinary call.
 */
void mrcuda_record_cudaUnregisterFatBinary(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle)
{
    /*MRecord_t *recordPtr;
    void ***fatCubinHandleAddr;
    
    __mrcuda_record_new_safe(&recordPtr);

    if((fatCubinHandleAddr = g_hash_table_lookup(__fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaUnregisterFatBinary";
    recordPtr->data.cudaUnregisterFatBinary.fatCubinHandle = fatCubinHandleAddr;
    recordPtr->replayFunc = &mrcuda_record_cudaUnregisterFatBinary;*/
}

/**
 * Record a cudaMalloc call.
 */
void mrcuda_record_cudaMalloc(MRCUDAGPU_t *mrcudaGPU, void **devPtr, size_t size)
{
    MRecord_t *recordPtr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);
    
    recordPtr->functionName = "cudaMalloc";
    recordPtr->skipMockStream = 0;
    recordPtr->data.cudaMalloc.devPtr = *devPtr;
    recordPtr->data.cudaMalloc.size = size;
    recordPtr->replayFunc = &mrcuda_replay_cudaMalloc;

    g_hash_table_insert(mrcudaGPU->mrecordGPU->activeMemoryTable, *devPtr, recordPtr);

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaFree call.
 */
void mrcuda_record_cudaFree(MRCUDAGPU_t *mrcudaGPU, void *devPtr)
{
    MRecord_t *recordPtr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    recordPtr->functionName = "cudaFree";
    recordPtr->skipMockStream = 0;
    recordPtr->data.cudaFree.devPtr = devPtr;
    recordPtr->replayFunc = &mrcuda_replay_cudaFree;

    g_hash_table_remove(mrcudaGPU->mrecordGPU->activeMemoryTable, devPtr);

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaBindTexture call.
 */
void mrcuda_record_cudaBindTexture(
    MRCUDAGPU_t *mrcudaGPU, 
    size_t *offset, 
    const struct textureReference *texref, 
    const void *devPtr, 
    const struct cudaChannelFormatDesc *desc, 
    size_t size
)
{
    MRecord_t *recordPtr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    recordPtr->functionName = "cudaBindTexture";
    recordPtr->skipMockStream = 0;
    if(offset == NULL)
        recordPtr->data.cudaBindTexture.offset = 0;
    else
        recordPtr->data.cudaBindTexture.offset = *offset;
    recordPtr->data.cudaBindTexture.texref = texref;
    recordPtr->data.cudaBindTexture.devPtr = devPtr;
    memcpy(&(recordPtr->data.cudaBindTexture.desc), desc, sizeof(struct cudaChannelFormatDesc));
    recordPtr->data.cudaBindTexture.size = size;
    recordPtr->replayFunc = &mrcuda_replay_cudaBindTexture;

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaStreamCreate call.
 */
void mrcuda_record_cudaStreamCreate(MRCUDAGPU_t *mrcudaGPU, cudaStream_t *pStream)
{
    MRecord_t *recordPtr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    recordPtr->functionName = "cudaStreamCreate";
    recordPtr->skipMockStream = 0;
    recordPtr->data.cudaStreamCreate.pStream = pStream;
    recordPtr->replayFunc = &mrcuda_replay_cudaStreamCreate;

    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaHostAlloc call.
 * The dual function of this call is mrcuda_replay_cudaFreeHost.
 */
void mrcuda_record_cudaHostAlloc(MRCUDAGPU_t *mrcudaGPU, void **pHost, size_t size, unsigned int flags)
{
    STARTTIMMER();
    g_hash_table_insert(mrcudaGPU->mrecordGPU->hostAllocTable, *pHost, mrcudaGPU->defaultHandler);
    ENDTIMMER(recordAccTime);
}

/**
 * Record a cudaSetDeviceFlags call.
 */
void mrcuda_record_cudaSetDeviceFlags(MRCUDAGPU_t *mrcudaGPU, unsigned int flags)
{
    MRecord_t *recordPtr;

    STARTTIMMER();

    __mrcuda_record_new_safe(&recordPtr, mrcudaGPU);

    recordPtr->functionName = "cudaSetDeviceFlags";
    recordPtr->skipMockStream = 1;
    recordPtr->data.cudaSetDeviceFlags.flags = flags;
    recordPtr->replayFunc = &mrcuda_replay_cudaSetDeviceFlags;

    ENDTIMMER(recordAccTime);
}

/*******************************************
 * Below this are replay functions.        *
 *******************************************/

/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    void **fatCubinHandle;

    fatCubinHandle = mrcudaGPU->defaultHandler->__mrcudaRegisterFatBinary(
        record->data.cudaRegisterFatBinary.fatCubin
    );

    record->data.cudaRegisterFatBinary.fatCubinHandle = fatCubinHandle;
}

/**
 * Replay a cudaRegisterFunction call.
 */
void mrcuda_replay_cudaRegisterFunction(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->__mrcudaRegisterFunction(
        *(record->data.cudaRegisterFunction.fatCubinHandlePtr),
        record->data.cudaRegisterFunction.hostFun,
        record->data.cudaRegisterFunction.deviceFun.ptr,
        record->data.cudaRegisterFunction.deviceName.ptr,
        record->data.cudaRegisterFunction.thread_limit,
        record->data.cudaRegisterFunction.tid,
        record->data.cudaRegisterFunction.bid,
        record->data.cudaRegisterFunction.bDim,
        record->data.cudaRegisterFunction.gDim,
        record->data.cudaRegisterFunction.wSize
    );
}

/**
 * Replay a cudaRegisterVar call.
 */
void mrcuda_replay_cudaRegisterVar(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->__mrcudaRegisterVar(
        *(record->data.cudaRegisterVar.fatCubinHandlePtr),
        record->data.cudaRegisterVar.hostVar.ptr,
        record->data.cudaRegisterVar.deviceAddress.ptr,
        record->data.cudaRegisterVar.deviceName.ptr,
        record->data.cudaRegisterVar.ext,
        record->data.cudaRegisterVar.size,
        record->data.cudaRegisterVar.constant,
        record->data.cudaRegisterVar.global
    );
}

/**
 * Replay a cudaRegisterTexture call.
 */
void mrcuda_replay_cudaRegisterTexture(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->__mrcudaRegisterTexture(
        *(record->data.cudaRegisterTexture.fatCubinHandlePtr),
        record->data.cudaRegisterTexture.hostVar.ptr,
        record->data.cudaRegisterTexture.deviceAddress,
        record->data.cudaRegisterTexture.deviceName.ptr,
        record->data.cudaRegisterTexture.dim,
        record->data.cudaRegisterTexture.norm,
        record->data.cudaRegisterTexture.ext
    );
}

/**
 * Replay a cudaUnregisterFatBinary call.
 */
void mrcuda_replay_cudaUnregisterFatBinary(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->__mrcudaUnregisterFatBinary(
        *(record->data.cudaUnregisterFatBinary.fatCubinHandle)
    );
}

/**
 * Replay a cudaMalloc call.
 */
void mrcuda_replay_cudaMalloc(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    void *devPtr;
    mrcudaGPU->defaultHandler->mrcudaMalloc(
        &devPtr,
        record->data.cudaMalloc.size
    );
    // Test whether we get the same ptr back.
    assert(devPtr == record->data.cudaMalloc.devPtr);
}

/**
 * Replay a cudaFree call.
 */
void mrcuda_replay_cudaFree(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->mrcudaFree(
        record->data.cudaFree.devPtr
    );
}


/**
 * Replay a cudaBindTexture call.
 */
void mrcuda_replay_cudaBindTexture(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    size_t offset;
    mrcudaGPU->defaultHandler->mrcudaBindTexture(
        &offset,
        record->data.cudaBindTexture.texref,
        record->data.cudaBindTexture.devPtr,
        &(record->data.cudaBindTexture.desc),
        record->data.cudaBindTexture.size
    );
}

/**
 * Replay a cudaStreamCreate call.
 */
void mrcuda_replay_cudaStreamCreate(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->mrcudaStreamCreate(
        record->data.cudaStreamCreate.pStream
    );
}

/**
 * Replay a cudaFreeHost call.
 * This function looks for the library used for allocating the ptr.
 * The dual function of this call is mrcuda_record_cudaHostAlloc.
 */
MRCUDASym_t *mrcuda_replay_cudaFreeHost(MRCUDAGPU_t *mrcudaGPU, void *ptr)
{
    MRCUDASym_t *calledLib = NULL;
    if ((calledLib = g_hash_table_lookup(mrcudaGPU->mrecordGPU->hostAllocTable, ptr)) != NULL)
        g_hash_table_remove(mrcudaGPU->mrecordGPU->hostAllocTable, ptr);
    else
        calledLib = mrcudaGPU->defaultHandler;
    return calledLib;
}

/**
 * Replay a cudaSetDeviceFlags call.
 */
void mrcuda_replay_cudaSetDeviceFlags(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record)
{
    mrcudaGPU->defaultHandler->mrcudaSetDeviceFlags(
        record->data.cudaSetDeviceFlags.flags
    );
}

/*
 * This function downloads the content of an active memory region to the native device.
 * The structure of this function is as of GHRFunc for compatibility with GHashTable.
 * @param key is a void *devPtr.
 * @param value is the MRecord_t *record associated with the key.
 * @param user_data is always NULL.
 * @return TRUE always.
 * @exception exit and report the error.
 */
gboolean __sync_mem_instance(gpointer key, gpointer value, gpointer user_data)
{
    MRecord_t *record = (MRecord_t *)value;
    MRCUDAGPU_t *mrcudaGPU = (MRCUDAGPU_t *)user_data;
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;
    MHelperCommand_t command;
    MHelperResult_t result;
    void *cache;

    memsyncSize += record->data.cudaMalloc.size;
    memsyncNumCalls++;

    switch (mrcudaGPU->status) {
        case MRCUDA_GPU_STATUS_NATIVE:
            cache = malloc(record->data.cudaMalloc.size);
            if (cache == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache for __sync_mem_instance.\n");
            break;
        case MRCUDA_GPU_STATUS_HELPER:
            if ((sharedMemInfo = mhelper_mem_malloc(record->data.cudaMalloc.size)) == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache on shared mem for __sync_mem_instance.\n");
            cache = sharedMemInfo->startAddr;
            break;
        default:
            REPORT_ERROR_AND_EXIT("Cannot perform __sync_mem_instance back to rCUDA.\n");
            break;
    }

    if(mrcudaSymRCUDA->mrcudaMemcpy(
        cache,
        record->data.cudaMalloc.devPtr,
        record->data.cudaMalloc.size,
        cudaMemcpyDeviceToHost
    ) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy memory from rCUDA to host for caching in __sync_mem_instance.\n");

    switch (mrcudaGPU->status) {
        case MRCUDA_GPU_STATUS_NATIVE:
            if(mrcudaSymNvidia->mrcudaMemcpy(
                record->data.cudaMalloc.devPtr,
                cache,
                record->data.cudaMalloc.size,
                cudaMemcpyHostToDevice
            ) != cudaSuccess)
                REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache to the native device in __sync_mem_instance.\n");
            free(cache);
            break;
        case MRCUDA_GPU_STATUS_HELPER:
            command.id = mhelper_generate_command_id(mrcudaGPU);
            command.type = MRCOMMAND_TYPE_CUDAMEMCPY;
            command.args.cudaMemcpy.dst = record->data.cudaMalloc.devPtr;
            command.args.cudaMemcpy.count = record->data.cudaMalloc.size;
            command.args.cudaMemcpy.kind = cudaMemcpyHostToDevice;
            command.args.cudaMemcpy.sharedMem = sharedMemInfo->sharedMem;
            mhelper_mem_detach(sharedMemInfo);
            result = mhelper_call(mrcudaGPU->mhelperProcess, command);
            free(sharedMemInfo);
            if (result.internalError != 0 || result.cudaError != cudaSuccess)
                REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache on shared mem to the native device in __sync_mem_instance.\n");
            break;
        default:
            REPORT_ERROR_AND_EXIT("Cannot perform __sync_mem_instance back to rCUDA.\n");
            break;
    }

    return TRUE;
}

/*
 * This function downloads the content of an active symbol to the native device.
 * The structure of this function is as of GHRFunc for compatibility with GHashTable.
 * @param key is a void *devPtr.
 * @param value is the MRecord_t *record associated with the key.
 * @param user_data is always NULL.
 * @return TRUE always.
 * @exception exit and report the error.
 */
gboolean __sync_symbol_instance(gpointer key, gpointer value, gpointer user_data)
{
    MRecord_t *record = (MRecord_t *)value;
    MRCUDAGPU_t *mrcudaGPU = (MRCUDAGPU_t *)user_data;
    MRCUDASharedMemLocalInfo_t *sharedMemInfo;
    MHelperCommand_t command;
    MHelperResult_t result;
    void *dst;
    cudaError_t error;

    memsyncSize += record->data.cudaRegisterVar.size;
    memsyncNumCalls++;

    switch (mrcudaGPU->status) {
        case MRCUDA_GPU_STATUS_NATIVE:
            if ((dst = malloc(record->data.cudaRegisterVar.size)) == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache for __sync_symbol_instance.\n");
            break;
        case MRCUDA_GPU_STATUS_HELPER:
            if ((sharedMemInfo = mhelper_mem_malloc(record->data.cudaRegisterVar.size + strlen(record->data.cudaRegisterVar.hostVar.ptr) + 1)) == NULL)
                REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache on shared mem for __sync_symbol_instance.\n");
            dst = sharedMemInfo->startAddr;
            strcpy(dst + record->data.cudaRegisterVar.size, record->data.cudaRegisterVar.hostVar.ptr);
            break;
        default:
            REPORT_ERROR_AND_EXIT("Cannot perform __sync_symbol_instance back to rCUDA.\n");
    }
    
    if ((error = mrcudaSymRCUDA->mrcudaMemcpyFromSymbol(
        dst, 
        record->data.cudaRegisterVar.hostVar.ptr, 
        record->data.cudaRegisterVar.size,
        0,
        cudaMemcpyDeviceToHost
    )) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy from rCUDA to host for caching in __sync_symbol_instance.\n");

    switch (mrcudaGPU->status) {
        case MRCUDA_GPU_STATUS_NATIVE:
            if (mrcudaGPU->defaultHandler->mrcudaMemcpyToSymbol(
                record->data.cudaRegisterVar.hostVar.ptr,
                dst,
                record->data.cudaRegisterVar.size,
                0,
                cudaMemcpyHostToDevice
            ) != cudaSuccess)
                REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache to the native device in __sync_symbol_instance.\n");
            free(dst);
            break;
        case MRCUDA_GPU_STATUS_HELPER:
            command.id = mhelper_generate_command_id(mrcudaGPU);
            command.type = MRCOMMAND_TYPE_CUDAMEMCPYTOSYMBOL;
            command.args.cudaMemcpyToSymbol.count = record->data.cudaRegisterVar.size;
            command.args.cudaMemcpyToSymbol.offset = 0;
            command.args.cudaMemcpyToSymbol.kind = cudaMemcpyHostToDevice;
            command.args.cudaMemcpyToSymbol.stream = NULL;
            command.args.cudaMemcpyToSymbol.sharedMem = sharedMemInfo->sharedMem;
            mhelper_mem_detach(sharedMemInfo);
            result = mhelper_call(mrcudaGPU->mhelperProcess, command);
            free(sharedMemInfo);
            if (result.internalError != 0 || result.cudaError != cudaSuccess)
                REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache on shared mem to the native device in __sync_symbol_instance.\n");
            break;
        default:
            REPORT_ERROR_AND_EXIT("Cannot perform __sync_symbol_instance back to rCUDA.\n");
    }

    return TRUE;
}

/**
 * Download the content of active memory regions to the native device.
 * Exit and report error if an error is found.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t that the sync mem will be performed on.
 */
void mrcuda_sync_mem(MRCUDAGPU_t *mrcudaGPU)
{
    STARTTIMMER();
    g_hash_table_foreach_remove(
        mrcudaGPU->mrecordGPU->activeMemoryTable,
        &__sync_mem_instance,
        mrcudaGPU
    );

    g_hash_table_foreach_remove(
        mrcudaGPU->mrecordGPU->activeSymbolTable,
        &__sync_symbol_instance,
        mrcudaGPU
    );
    ENDTIMMER(memsyncAccTime);
}

/**
 * Simulate cuda streams on the native CUDA so that the number of streams are equaled to that of rCUDA.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t that the simulate stream will be performed on.
 */
void mrcuda_simulate_stream(MRCUDAGPU_t *mrcudaGPU)
{
    cudaStream_t stream;

    if (mrcudaGPU->defaultHandler->mrcudaStreamCreate(&stream) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot simulate a cuda stream on the native CUDA.\n");
    if (mrcudaGPU->defaultHandler->mrcudaStreamCreate(&stream) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot simulate a cuda stream on the native CUDA.\n");
}

/**
 * Simulate cuCtxCreate on the specified gpuID.
 * If mrcudaGPU->status == MRCUDA_GPU_STATUS_HELPER, ask the helper to handle the command.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t.
 * @param gpuID the ID of the GPU a context will be created on.
 * @return 0 on success; -1 otherwise.
 */
int mrcuda_simulate_cuCtxCreate(MRCUDAGPU_t *mrcudaGPU, int gpuID)
{
    CUcontext pctx;

    if (mrcudaGPU->status == MRCUDA_GPU_STATUS_HELPER)
        return mhelper_int_cuCtxCreate_internal(mrcudaGPU);
    return cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO, gpuID) == CUDA_SUCCESS ? 0 : -1;
}

