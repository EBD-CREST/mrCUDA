#ifndef __MRCUDA_RECORD__HEADER__
#define __MRCUDA_RECORD__HEADER__

#include <cuda_runtime.h>

#include "common.h"

typedef struct MRecord
{
    char *functionName;
    union 
    {
        struct cudaRegisterFatBinary
        {
            void *fatCubin;
        } cudaRegisterFatBinary;
        struct cudaRegisterFunction
        {
           void **fatCubinHandle;
           const char *hostFun;
           char *deviceFun;
           const char *deviceName;
           int thread_limit;
           uint3 *tid;
           uint3 *bid;
           dim3 *bDim;
           dim3 *gDim;
           int *wSize;
        } cudaRegisterFunction;
        struct cudaMemcpyToSymbol
        {
            void *symbol;
            size_t count;
            size_t offset;
            enum cudaMemcpyKind kind;
        } cudaMemcpyToSymbol;
        struct cudaMemset
        {
            void *devPtr;
            int value;
            size_t count;
        } cudaMemset;
        struct cudaMalloc
        {
            void *devPtr;
            size_t size;
        } cudaMalloc;
        struct cudaFree
        {
            void *devPtr;
        } cudaFree;
        struct cudaBindTexture
        {
            struct textureReference *textref;
            void *devPtr;
            struct cudaChannelFormatDesc *desc;
            size_t size;
        } cudaBindTexture;
        struct cudaCreateChannelDesc
        {
            int x;
            int y;
            int z;
            int w;
            enum cudaChannelFormatKind f;
        } cudaCreateChannelDesc;
        struct cudaSetDeviceFlags
        {
            unsigned int flags;
        } cudaSetDeviceFlags;
    } data;
    void (*replayFunc)(struct MRecord*);
    struct MRecord *next;
} MRecord;

extern MRecord *mrcudaRecordHeadPtr;
extern MRecord *mrcudaRecordTailPtr;

/**
 * Initialize the record/replay module.
 * Exit and report error if found.
 */
void mrcuda_record_init();

/**
 * Finalize the record/replay module.
 */
void mrcuda_record_fini();

/**
 * Record a cudaRegisterFatBinary call.
 */
void mrcuda_record_cudaRegisterFatBinary(void* fatCubin);

/**
 * Record a cudaRegisterFunction call.
 */
void mrcuda_record_cudaRegisterFunction(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize);

/**
 * Record a cudaMemcpyToSymbol call.
 */
void mrcuda_record_cudaMemcpyToSymbol(const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);

/**
 * Record a cudaMemset call.
 */
void mrcuda_record_cudaMemset(void *devPtr, int value, size_t count);

/**
 * Record a cudaMalloc call.
 */
void mrcuda_record_cudaMalloc(void **devPtr, size_t size);

/**
 * Record a cudaFree call.
 */
void mrcuda_record_cudaFree(void *devPtr);

/**
 * Record a cudaBindTexture call.
 */
void mrcuda_record_cudaBindTexture(const struct textureReference *textref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size);

/**
 * Record a cudaCreateChannelDesc call.
 */
void mrcuda_record_cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);

/**
 * Record a cudaSetDeviceFlags call.
 */
void mrcuda_record_cudaSetDeviceFlags(unsigned int flags);

/**
 * Record a cudaStreamCreate call.
 */
void mrcuda_record_cudaStreamCreate();



/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRecord* record);

/**
 * Replay a cudaRegisterFunction call.
 */
void mrcuda_replay_cudaRegisterFunction(MRecord* record);

/**
 * Replay a cudaMemcpyToSymbol call.
 */
void mrcuda_replay_cudaMemcpyToSymbol(MRecord* record);

/**
 * Replay a cudaMemset call.
 */
void mrcuda_replay_cudaMemset(MRecord* record);

/**
 * Replay a cudaMalloc call.
 */
void mrcuda_replay_cudaMalloc(MRecord* record);

/**
 * Replay a cudaFree call.
 */
void mrcuda_replay_cudaFree(MRecord* record);

/**
 * Replay a cudaBindTexture call.
 */
void mrcuda_replay_cudaBindTexture(MRecord* record);

/**
 * Replay a cudaCreateChannelDesc call.
 */
void mrcuda_replay_cudaCreateChannelDesc(MRecord* record);

/**
 * Replay a cudaSetDeviceFlags call.
 */
void mrcuda_replay_cudaSetDeviceFalgs(MRecord* record);

/**
 * Replay a cudaStreamCreate call.
 */
void mrcuda_replay_cudaStreamCreate(MRecord* record);

/**
 * Download the content of active memory regions to the native device.
 * Exit and report error if an error is found.
 */
void mrcuda_sync_mem();


#endif
