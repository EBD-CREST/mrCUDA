#ifndef __MRCUDA_RECORD__HEADER__
#define __MRCUDA_RECORD__HEADER__

#include <cuda_runtime.h>

#include "common.h"
#include "mrcuda.h"

typedef struct MRecord
{
    char *functionName;
    int skip_mock_stream;
    union 
    {
        struct cudaRegisterFatBinary
        {
            void *fatCubin;
            void **fatCubinHandle;
        } cudaRegisterFatBinary;
        struct cudaRegisterFunction
        {
           void ***fatCubinHandle;
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
        struct cudaRegisterVar
        {
            void ***fatCubinHandle;
            char *hostVar;
            char *deviceAddress;
            const char *deviceName;
            int ext;
            int size;
            int constant;
            int global;
        } cudaRegisterVar;
        struct cudaRegisterTexture
        {
            void ***fatCubinHandle;
            const struct textureReference *hostVar;
            const void **deviceAddress;
            const char *deviceName;
            int dim;
            int norm;
            int ext;
        } cudaRegisterTexture;
        struct cudaUnregisterFatBinary
        {
            void ***fatCubinHandle;
        } cudaUnregisterFatBinary;
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
            size_t offset;
            const struct textureReference *texref;
            const void *devPtr;
            struct cudaChannelFormatDesc desc;
            size_t size;
        } cudaBindTexture;
        struct cudaStreamCreate
        {
            cudaStream_t *pStream;
        } cudaStreamCreate;
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

extern unsigned long int __cudaMemcpySizeTotal;
extern unsigned long int __cudaMemcpyToSymbolSizeTotal;

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
void mrcuda_record_cudaRegisterFatBinary(void* fatCubin, void **fatCubinHandle);

/**
 * Record a cudaRegisterFunction call.
 */
void mrcuda_record_cudaRegisterFunction(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize);

/**
 * Record a cudaRegisterVar call.
 */
void mrcuda_record_cudaRegisterVar(void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global);

/**
 * Record a cudaRegisterTexture call.
 */
void mrcuda_record_cudaRegisterTexture(void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext);

/**
 * Record a cudaUnregisterFatBinary call.
 */
void mrcuda_record_cudaUnregisterFatBinary(void **fatCubinHandle);

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
void mrcuda_record_cudaBindTexture(size_t *offset, const struct textureReference *textref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size);

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
void mrcuda_record_cudaStreamCreate(cudaStream_t *pStream);

/**
 * Record a cudaHostAlloc call.
 * The dual function of this call is mrcuda_replay_cudaFreeHost.
 */
void mrcuda_record_cudaHostAlloc(void **pHost, size_t size, unsigned int flags);


/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRecord* record);

/**
 * Replay a cudaRegisterFunction call.
 */
void mrcuda_replay_cudaRegisterFunction(MRecord* record);

/**
 * Replay a cudaRegisterVar call.
 */
void mrcuda_replay_cudaRegisterVar(MRecord *record);

/**
 * Replay a cudaRegisterTexture call.
 */
void mrcuda_replay_cudaRegisterTexture(MRecord *record);

/**
 * Replay a cudaUnregisterFatBinary call.
 */
void mrcuda_replay_cudaUnregisterFatBinary(MRecord *record);

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
void mrcuda_replay_cudaSetDeviceFlags(MRecord* record);

/**
 * Replay a cudaStreamCreate call.
 */
void mrcuda_replay_cudaStreamCreate(MRecord* record);

/**
 * Replay a cudaFreeHost call.
 * This function looks for the library used for allocating the ptr.
 * The dual function of this call is mrcuda_record_cudaHostAlloc.
 */
MRCUDASym* mrcuda_replay_cudaFreeHost(void *ptr);

/**
 * Download the content of active memory regions to the native device.
 * Exit and report error if an error is found.
 */
void mrcuda_sync_mem();

/**
 * Simulate cuda streams on the native CUDA so that the number of streams are equaled to that of rCUDA.
 */
void mrcuda_simulate_stream();

#endif
