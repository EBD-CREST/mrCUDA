#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include <assert.h>

// For manual profiling
#include <sys/time.h>

#include "record.h"
#include "mrcuda.h"

#define _MRCUDA_RECORD_MALLOC_CACHE_SIZE_ 100
static int mrcudaRecordCacheCounter = -1;
MRecord *mrcudaRecordCache = NULL;

MRecord *mrcudaRecordHeadPtr = NULL;
MRecord *mrcudaRecordTailPtr = NULL;

static GHashTable *__activeMemoryTable;
static GHashTable *__activeSymbolTable;
static GHashTable *__fatCubinHandleAddrTable;
static GHashTable *__hostAllocTable;

static long int _totalSyncMemSize = 0;
static int _totalSyncMemCalls = 0;
static double _totalSyncTime = 0.0f;


/**
 * Allocate a new MRecord and appropriately link the new one with the previous.
 * The pointer to the new MRecord is recorded into recordPtr.
 * @param recordPtr the variable that the new MRecord will be recorded into.
 * @return 0 if success, otherwise error number.
 */
static int __mrcuda_record_new(MRecord **recordPtr)
{
    if(!mrcudaRecordCache && (mrcudaRecordCache = calloc(_MRCUDA_RECORD_MALLOC_CACHE_SIZE_, sizeof(MRecord))) == NULL)
        goto __mrcuda_record_new_err_1;
    mrcudaRecordCacheCounter++;
    *recordPtr = &(mrcudaRecordCache[mrcudaRecordCacheCounter]);

    if(mrcudaRecordHeadPtr == NULL)
        mrcudaRecordHeadPtr = *recordPtr;
    else
        mrcudaRecordTailPtr->next = *recordPtr;

    mrcudaRecordTailPtr = *recordPtr;

    if(mrcudaRecordCacheCounter >= _MRCUDA_RECORD_MALLOC_CACHE_SIZE_ - 1)
    {
        mrcudaRecordCacheCounter = -1;
        mrcudaRecordCache = NULL;
    }

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
    __activeMemoryTable = g_hash_table_new(g_direct_hash, g_direct_equal);
    if(__activeMemoryTable == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate memory for __activeMemoryTable.\n");
    __activeSymbolTable = g_hash_table_new(g_direct_hash, g_direct_equal);
    if(__activeSymbolTable == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate memory for __activeSymbolTable.\n");
    __fatCubinHandleAddrTable = g_hash_table_new(g_direct_hash, g_direct_equal);
    if(__fatCubinHandleAddrTable == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate memory for __fatCubinHandleAddrTable.\n");
    __hostAllocTable = g_hash_table_new(g_direct_hash, g_direct_equal);
    if(__hostAllocTable == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate memory for __hostAllocTable.\n");
}

/**
 * Finalize the record/replay module.
 */
void mrcuda_record_fini()
{
    if(__activeMemoryTable)
        g_hash_table_destroy(__activeMemoryTable);
    if(__fatCubinHandleAddrTable)
        g_hash_table_destroy(__fatCubinHandleAddrTable);
    if(__hostAllocTable)
        g_hash_table_destroy(__hostAllocTable);
}

/*******************************************
 * Below this are record functions.        *
 *******************************************/

/**
 * Record a cudaRegisterFatBinary call.
 */
void mrcuda_record_cudaRegisterFatBinary(void* fatCubin, void **fatCubinHandle)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "__cudaRegisterFatBinary";
    recordPtr->skip_mock_stream = 1;
    recordPtr->data.cudaRegisterFatBinary.fatCubin = fatCubin;
    recordPtr->data.cudaRegisterFatBinary.fatCubinHandle = fatCubinHandle;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterFatBinary;

    g_hash_table_insert(__fatCubinHandleAddrTable, fatCubinHandle, &(recordPtr->data.cudaRegisterFatBinary.fatCubinHandle));
}

/**
 * Record a cudaRegisterFunction call.
 */
void mrcuda_record_cudaRegisterFunction(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize)
{
    MRecord *recordPtr;
    void ***fatCubinHandleAddr;

    __mrcuda_record_new_safe(&recordPtr);

    if((fatCubinHandleAddr = g_hash_table_lookup(__fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaRegisterFunction";
    recordPtr->skip_mock_stream = 1;
    recordPtr->data.cudaRegisterFunction.fatCubinHandle = fatCubinHandleAddr;
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
 * Record a cudaRegisterVar call.
 */
void mrcuda_record_cudaRegisterVar(void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global)
{
    MRecord *recordPtr;
    void ***fatCubinHandleAddr;
    
    __mrcuda_record_new_safe(&recordPtr);

    if((fatCubinHandleAddr = g_hash_table_lookup(__fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaRegisterVar";
    recordPtr->skip_mock_stream = 1;
    recordPtr->data.cudaRegisterVar.fatCubinHandle = fatCubinHandleAddr;
    recordPtr->data.cudaRegisterVar.hostVar = hostVar;
    recordPtr->data.cudaRegisterVar.deviceAddress = deviceAddress;
    recordPtr->data.cudaRegisterVar.deviceName = deviceName;
    recordPtr->data.cudaRegisterVar.ext = ext;
    recordPtr->data.cudaRegisterVar.size = size;
    recordPtr->data.cudaRegisterVar.constant = constant;
    recordPtr->data.cudaRegisterVar.global = global;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterVar;

    g_hash_table_insert(__activeSymbolTable, hostVar, recordPtr);
}

/**
 * Record a cudaRegisterTexture call.
 */
void mrcuda_record_cudaRegisterTexture(void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext)
{
    MRecord *recordPtr;
    void ***fatCubinHandleAddr;
    
    __mrcuda_record_new_safe(&recordPtr);

    if((fatCubinHandleAddr = g_hash_table_lookup(__fatCubinHandleAddrTable, fatCubinHandle)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot find the address of the specified fatCubinHandle.\n");

    recordPtr->functionName = "__cudaRegisterTexture";
    recordPtr->skip_mock_stream = 1;
    recordPtr->data.cudaRegisterTexture.fatCubinHandle = fatCubinHandleAddr;
    recordPtr->data.cudaRegisterTexture.hostVar = hostVar;
    recordPtr->data.cudaRegisterTexture.deviceAddress = deviceAddress;
    recordPtr->data.cudaRegisterTexture.deviceName = deviceName;
    recordPtr->data.cudaRegisterTexture.dim = dim;
    recordPtr->data.cudaRegisterTexture.norm = norm;
    recordPtr->data.cudaRegisterTexture.ext = ext;
    recordPtr->replayFunc = &mrcuda_replay_cudaRegisterTexture;
}

/**
 * Record a cudaUnregisterFatBinary call.
 */
void mrcuda_record_cudaUnregisterFatBinary(void **fatCubinHandle)
{
    /*MRecord *recordPtr;
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
void mrcuda_record_cudaMalloc(void **devPtr, size_t size)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);
    
    recordPtr->functionName = "cudaMalloc";
    recordPtr->skip_mock_stream = 0;
    recordPtr->data.cudaMalloc.devPtr = *devPtr;
    recordPtr->data.cudaMalloc.size = size;
    recordPtr->replayFunc = &mrcuda_replay_cudaMalloc;

    g_hash_table_insert(__activeMemoryTable, *devPtr, recordPtr);
}

/**
 * Record a cudaFree call.
 */
void mrcuda_record_cudaFree(void *devPtr)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaFree";
    recordPtr->skip_mock_stream = 0;
    recordPtr->data.cudaFree.devPtr = devPtr;
    recordPtr->replayFunc = &mrcuda_replay_cudaFree;

    g_hash_table_remove(__activeMemoryTable, devPtr);
}

/**
 * Record a cudaBindTexture call.
 */
void mrcuda_record_cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaBindTexture";
    recordPtr->skip_mock_stream = 0;
    if(offset == NULL)
        recordPtr->data.cudaBindTexture.offset = 0;
    else
        recordPtr->data.cudaBindTexture.offset = *offset;
    recordPtr->data.cudaBindTexture.texref = texref;
    recordPtr->data.cudaBindTexture.devPtr = devPtr;
    memcpy(&(recordPtr->data.cudaBindTexture.desc), desc, sizeof(struct cudaChannelFormatDesc));
    recordPtr->data.cudaBindTexture.size = size;
    recordPtr->replayFunc = &mrcuda_replay_cudaBindTexture;
}

/**
 * Record a cudaStreamCreate call.
 */
void mrcuda_record_cudaStreamCreate(cudaStream_t *pStream)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaStreamCreate";
    recordPtr->skip_mock_stream = 0;
    recordPtr->data.cudaStreamCreate.pStream = pStream;
    recordPtr->replayFunc = &mrcuda_replay_cudaStreamCreate;
}

/**
 * Record a cudaHostAlloc call.
 * The dual function of this call is mrcuda_replay_cudaFreeHost.
 */
void mrcuda_record_cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    g_hash_table_insert(__hostAllocTable, *pHost, mrcudaSymDefault);
}

/**
 * Record a cudaSetDeviceFlags call.
 */
void mrcuda_record_cudaSetDeviceFlags(unsigned int flags)
{
    MRecord *recordPtr;

    __mrcuda_record_new_safe(&recordPtr);

    recordPtr->functionName = "cudaSetDeviceFlags";
    recordPtr->skip_mock_stream = 1;
    recordPtr->data.cudaSetDeviceFlags.flags = flags;
    recordPtr->replayFunc = &mrcuda_replay_cudaSetDeviceFlags;
}

/*******************************************
 * Below this are replay functions.        *
 *******************************************/

/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRecord* record)
{
    void **fatCubinHandle;

    fatCubinHandle = mrcudaSymNvidia->__mrcudaRegisterFatBinary(
        record->data.cudaRegisterFatBinary.fatCubin
    );

    record->data.cudaRegisterFatBinary.fatCubinHandle = fatCubinHandle;
}

/**
 * Replay a cudaRegisterFunction call.
 */
void mrcuda_replay_cudaRegisterFunction(MRecord* record)
{
    mrcudaSymNvidia->__mrcudaRegisterFunction(
        *(record->data.cudaRegisterFunction.fatCubinHandle),
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
 * Replay a cudaRegisterVar call.
 */
void mrcuda_replay_cudaRegisterVar(MRecord *record)
{
    mrcudaSymNvidia->__mrcudaRegisterVar(
        *(record->data.cudaRegisterVar.fatCubinHandle),
        record->data.cudaRegisterVar.hostVar,
        record->data.cudaRegisterVar.deviceAddress,
        record->data.cudaRegisterVar.deviceName,
        record->data.cudaRegisterVar.ext,
        record->data.cudaRegisterVar.size,
        record->data.cudaRegisterVar.constant,
        record->data.cudaRegisterVar.global
    );
}

/**
 * Replay a cudaRegisterTexture call.
 */
void mrcuda_replay_cudaRegisterTexture(MRecord *record)
{
    mrcudaSymNvidia->__mrcudaRegisterTexture(
        *(record->data.cudaRegisterTexture.fatCubinHandle),
        record->data.cudaRegisterTexture.hostVar,
        record->data.cudaRegisterTexture.deviceAddress,
        record->data.cudaRegisterTexture.deviceName,
        record->data.cudaRegisterTexture.dim,
        record->data.cudaRegisterTexture.norm,
        record->data.cudaRegisterTexture.ext
    );
}

/**
 * Replay a cudaUnregisterFatBinary call.
 */
void mrcuda_replay_cudaUnregisterFatBinary(MRecord *record)
{
    mrcudaSymNvidia->__mrcudaUnregisterFatBinary(
        *(record->data.cudaUnregisterFatBinary.fatCubinHandle)
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


/**
 * Replay a cudaBindTexture call.
 */
void mrcuda_replay_cudaBindTexture(MRecord* record)
{
    size_t offset;
    mrcudaSymNvidia->mrcudaBindTexture(
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
void mrcuda_replay_cudaStreamCreate(MRecord* record)
{
    mrcudaSymNvidia->mrcudaStreamCreate(
        record->data.cudaStreamCreate.pStream
    );
}

/**
 * Replay a cudaFreeHost call.
 * This function looks for the library used for allocating the ptr.
 * The dual function of this call is mrcuda_record_cudaHostAlloc.
 */
MRCUDASym* mrcuda_replay_cudaFreeHost(void *ptr)
{
    MRCUDASym *calledLib = NULL;
    if((calledLib = g_hash_table_lookup(__hostAllocTable, ptr)) != NULL)
        g_hash_table_remove(__hostAllocTable, ptr);
    else
        calledLib = mrcudaSymDefault;
    return calledLib;
}

/**
 * Replay a cudaSetDeviceFlags call.
 */
void mrcuda_replay_cudaSetDeviceFlags(MRecord* record)
{
    mrcudaSymNvidia->mrcudaSetDeviceFlags(
        record->data.cudaSetDeviceFlags.flags
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
    MRecord *record = (MRecord *)value;

    void *cache;
    cache = malloc(record->data.cudaMalloc.size);
    if(cache == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache for __sync_mem_instance.\n");
    if(mrcudaSymRCUDA->mrcudaMemcpy(
        cache,
        record->data.cudaMalloc.devPtr,
        record->data.cudaMalloc.size,
        cudaMemcpyDeviceToHost
    ) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy memory from rCUDA to host for caching in __sync_mem_instance.\n");
    if(mrcudaSymNvidia->mrcudaMemcpy(
        record->data.cudaMalloc.devPtr,
        cache,
        record->data.cudaMalloc.size,
        cudaMemcpyHostToDevice
    ) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache to the native device in __sync_mem_instance.\n");
    free(cache);

    _totalSyncMemSize += record->data.cudaMalloc.size;
    _totalSyncMemCalls++;

    return TRUE;
}

/*
 * This function downloads the content of an active symbol to the native device.
 * The structure of this function is as of GHRFunc for compatibility with GHashTable.
 * @param key is a void *devPtr.
 * @param value is the MRecord *record associated with the key.
 * @param user_data is always NULL.
 * @return TRUE always.
 * @exception exit and report the error.
 */
gboolean __sync_symbol_instance(gpointer key, gpointer value, gpointer user_data)
{
    MRecord *record = (MRecord *)value;
    void *dst;
    cudaError_t error;

    if((dst = malloc(record->data.cudaRegisterVar.size)) == NULL)
        REPORT_ERROR_AND_EXIT("Cannot allocate the variable cache for __sync_symbol_instance.\n");
    if((error = mrcudaSymRCUDA->mrcudaMemcpyFromSymbol(
        dst, 
        record->data.cudaRegisterVar.hostVar, 
        record->data.cudaRegisterVar.size,
        0,
        cudaMemcpyDeviceToHost
    )) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy from rCUDA to host for caching in __sync_symbol_instance.\n");
    if(mrcudaSymNvidia->mrcudaMemcpyToSymbol(
        record->data.cudaRegisterVar.hostVar,
        dst,
        record->data.cudaRegisterVar.size,
        0,
        cudaMemcpyHostToDevice
    ) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot copy memory from the host's cache to the native device in __sync_symbol_instance.\n");
    free(dst);

    _totalSyncMemSize += record->data.cudaRegisterVar.size;
    _totalSyncMemCalls++;

    return TRUE;
}

/**
 * Download the content of active memory regions to the native device.
 * Exit and report error if an error is found.
 */
void mrcuda_sync_mem()
{
    struct timeval start_time, stop_time;

    gettimeofday(&start_time, NULL);

    g_hash_table_foreach_remove(
        __activeMemoryTable,
        &__sync_mem_instance,
        NULL
    );

    g_hash_table_foreach_remove(
        __activeSymbolTable,
        &__sync_symbol_instance,
        NULL
    );

    gettimeofday(&stop_time, NULL);

    _totalSyncTime += (stop_time.tv_sec + (double)stop_time.tv_usec / 1000000.0f) - (start_time.tv_sec + (double)start_time.tv_usec / 1000000.0f);

    fprintf(stderr, "mrcuda_sync_mem: size: %ld\n", _totalSyncMemSize);
    fprintf(stderr, "mrcuda_sync_mem: num_calls: %d\n", _totalSyncMemCalls);
    fprintf(stderr, "mrcuda_sync_mem: time: %.6f\n", _totalSyncTime);
}

/**
 * Simulate cuda streams on the native CUDA so that the number of streams are equaled to that of rCUDA.
 */
void mrcuda_simulate_stream()
{
    cudaStream_t stream;

    if(mrcudaSymNvidia->mrcudaStreamCreate(&stream) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot simulate a cuda stream on the native CUDA.\n");
    if(mrcudaSymNvidia->mrcudaStreamCreate(&stream) != cudaSuccess)
        REPORT_ERROR_AND_EXIT("Cannot simulate a cuda stream on the native CUDA.\n");
}

