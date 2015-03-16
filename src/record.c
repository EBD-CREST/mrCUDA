#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <assert.h>

#include "record.h"
#include "mrcuda.h"

MRecord *mrcudaRecordHeadPtr = NULL;
MRecord *mrcudaRecordTailPtr = NULL;

static GHashTable *__active_memory_table;

/**
 * Allocate a new MRecord and appropriately link the new one with the previous.
 * The pointer to the new MRecord is recorded into recordPtr.
 * @param recordPtr the variable that the new MRecord will be recorded into.
 * @return 0 if success, otherwise error number.
 */
static int __mrcuda_record_new(MRecord **recordPtr)
{
    if((*recordPtr = calloc(1, sizeof(MRecord))) == NULL)
        goto __mrcuda_record_new_err_1;

    if(mrcudaRecordHeadPtr == NULL)
        mrcudaRecordHeadPtr = *recordPtr;
    else
        mrcudaRecordTailPtr->next = *recordPtr;

    mrcudaRecordTailPtr = *recordPtr;
    return 0;

__mrcuda_record_new_err_1:
    return -1;
}

/**
 * Call __mrcuda_record_new.
 * Exit and report error if found.
 * @param recordPtr the variable that the new MRecord will be recorded into.
 */
static inline void __mrcuda_record_new_safe(MRecord **recordPtr)
{
    if(__mrcuda_record_new(recordPtr) != 0)
        REPORT_ERROR_AND_EXIT("Cannot create a new MRecord.\n");
}

/**
 * Initialize the record/replay module.
 * Exit and report error if found.
 */
void mrcuda_record_init()
{
    __active_memory_table = g_hash_table_new(g_int_hash, g_int_equal);
    if(__active_memory_table == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate memory for __active_memory_table.\n");
}

/**
 * Finalize the record/replay module.
 */
void mrcuda_record_fini()
{
}

/*******************************************
 * Below this are record functions.        *
 *******************************************/

/**
 * Record a cudaRegisterFatBinary call.
 */
void mrcuda_record_cudaRegisterFatBinary(void* fatCubin)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaRegisterFatBinary";
    recordPtr->data.cudaRegisterFatBinary.fatCubin = fatCubin;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterFatBinary;
}

/**
 * Record a cudaRegisterFunction call.
 */
void mrcuda_record_cudaRegisterFunction(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaRegisterFunction";
    recordPtr->data.cudaRegisterFunction.fatCubinHandle = fatCubinHandle;
    recordPtr->data.cudaRegisterFunction.hostFun = hostFun;
    recordPtr->data.cudaRegisterFunction.deviceFun = deviceFun;
    recordPtr->data.cudaRegisterFunction.deviceName = deviceName;
    recordPtr->data.cudaRegisterFunction.thread_limit = thread_limit;
    recordPtr->data.cudaRegisterFunction.tid = tid;
    recordPtr->data.cudaRegisterFunction.bid = bid;
    recordPtr->data.cudaRegisterFunction.bDim = bDim;
    recordPtr->data.cudaRegisterFunction.gDim = gDim;
    recordPtr->data.cudaRegisterFunction.wSize = wSize;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterFunction;
}

/**
 * Record a cudaMalloc call.
 */
void mrcuda_record_cudaMalloc(void **devPtr, size_t size)
{
    MRecord *recordPtr;

	DPRINTF("ENTER mrcuda_record_cudaMalloc.\n");
    __mrcuda_record_new_safe(&recordPtr);
    
    recordPtr->functionName = "cudaMalloc";
    recordPtr->data.cudaMalloc.devPtr = *devPtr;
    recordPtr->data.cudaMalloc.size = size;
    recordPtr->replayFunc = &mrcuda_replay_cudaMalloc;

	DPRINTF("mrcuda_record_cudaMalloc before g_hash_table_insert.\n");
    g_hash_table_insert(__active_memory_table, devPtr, recordPtr);

	DPRINTF("EXIT mrcuda_record_cudaMalloc.\n");
}

/**
 * Record a cudaFree call.
 */
void mrcuda_record_cudaFree(void *devPtr)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaFree";
    recordPtr->data.cudaFree.devPtr = devPtr;
    recordPtr->replayFunc = &mrcuda_replay_cudaFree;

    g_hash_table_remove(__active_memory_table, &devPtr);
}

/*******************************************
 * Below this are replay functions.        *
 *******************************************/

/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRecord* record)
{
    mrcudaSymNvidia->__mrcudaRegisterFatBinary(
        record->data.cudaRegisterFatBinary.fatCubin
    );
}

/**
 * Replay a cudaRegisterFunction call.
 */
void mrcuda_replay_cudaRegisterFunction(MRecord* record)
{
    mrcudaSymNvidia->__mrcudaRegisterFunction(
        record->data.cudaRegisterFunction.fatCubinHandle,
        record->data.cudaRegisterFunction.hostFun,
        record->data.cudaRegisterFunction.deviceFun,
        record->data.cudaRegisterFunction.deviceName,
        record->data.cudaRegisterFunction.thread_limit,
        record->data.cudaRegisterFunction.tid,
        record->data.cudaRegisterFunction.bid,
        record->data.cudaRegisterFunction.bDim,
        record->data.cudaRegisterFunction.gDim,
        record->data.cudaRegisterFunction.wSize
    );
}

/**
 * Replay a cudaMalloc call.
 */
void mrcuda_replay_cudaMalloc(MRecord* record)
{
    void *devPtr;
    mrcudaSymNvidia->mrcudaMalloc(
        &devPtr,
        record->data.cudaMalloc.size
    );
    // Test whether we get the same ptr back.
    assert(devPtr == record->data.cudaMalloc.devPtr);
}

/**
 * Replay a cudaFree call.
 */
void mrcuda_replay_cudaFree(MRecord* record)
{
    mrcudaSymNvidia->mrcudaFree(
        record->data.cudaFree.devPtr
    );
}

/*
 * This function downloads the content of an active memory region to the native device.
 * The structure of this function is as of GHRFunc for compatibility with GHashTable.
 * @param key is a void *devPtr.
 * @param value is the MRecord *record associated with the key.
 * @param user_data is always NULL.
 * @return TRUE always.
 * @exception exit and report the error.
 */
gboolean __sync_mem_instance(gpointer key, gpointer value, gpointer user_data)
{
    void *devPtr = (void *)key;
    MRecord *record = (MRecord *)value;

    void *cache;
    cache = malloc(record->data.cudaMalloc.size);
    if(cache == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache.\n");
    if(mrcudaSymRCUDA->mrcudaMemcpy(
        cache,
        devPtr,
        record->data.cudaMalloc.size,
        cudaMemcpyDeviceToHost
    ) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy memory from rCUDA to host for caching.\n");
    if(mrcudaSymNvidia->mrcudaMemcpy(
        devPtr,
        cache,
        record->data.cudaMalloc.size,
        cudaMemcpyHostToDevice
    ) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache to the native device.\n");
    free(cache);

    return TRUE;
}

/**
 * Download the content of active memory regions to the native device.
 * Exit and report error if an error is found.
 */
void mrcuda_sync_mem()
{
    g_hash_table_foreach_remove(
        __active_memory_table,
        &__sync_mem_instance,
        NULL
    );
}

