#ifndef __MRCUDA_RECORD__HEADER__
#define __MRCUDA_RECORD__HEADER__

#include <cuda_runtime.h>
#include <glib.h>

#include "common.h"
#include "datatypes.h"

extern double recordAccTime;
extern double memsyncAccTime;
extern int memsyncNumCalls;
extern double memsyncSize;

extern MRecordGPU_t *mrecordGPUList;

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
void mrcuda_record_cudaRegisterFatBinary(MRCUDAGPU_t *mrcudaGPU, void* fatCubin, void **fatCubinHandle);

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
);

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
);

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
);

/**
 * Record a cudaUnregisterFatBinary call.
 */
void mrcuda_record_cudaUnregisterFatBinary(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle);

/**
 * Record a cudaMalloc call.
 */
void mrcuda_record_cudaMalloc(MRCUDAGPU_t *mrcudaGPU, void **devPtr, size_t size);

/**
 * Record a cudaFree call.
 */
void mrcuda_record_cudaFree(MRCUDAGPU_t *mrcudaGPU, void *devPtr);

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
);

/**
 * Record a cudaStreamCreate call.
 */
void mrcuda_record_cudaStreamCreate(MRCUDAGPU_t *mrcudaGPU, cudaStream_t *pStream);

/**
 * Record a cudaHostAlloc call.
 * The dual function of this call is mrcuda_replay_cudaFreeHost.
 */
void mrcuda_record_cudaHostAlloc(MRCUDAGPU_t *mrcudaGPU, void **pHost, size_t size, unsigned int flags);

/**
 * Record a cudaSetDeviceFlags call.
 */
void mrcuda_record_cudaSetDeviceFlags(MRCUDAGPU_t *mrcudaGPU, unsigned int flags);


/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaRegisterFunction call.
 */
void mrcuda_replay_cudaRegisterFunction(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaRegisterVar call.
 */
void mrcuda_replay_cudaRegisterVar(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaRegisterTexture call.
 */
void mrcuda_replay_cudaRegisterTexture(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaUnregisterFatBinary call.
 */
void mrcuda_replay_cudaUnregisterFatBinary(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaMalloc call.
 */
void mrcuda_replay_cudaMalloc(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaFree call.
 */
void mrcuda_replay_cudaFree(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaBindTexture call.
 */
void mrcuda_replay_cudaBindTexture(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaStreamCreate call.
 */
void mrcuda_replay_cudaStreamCreate(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Replay a cudaFreeHost call.
 * This function looks for the library used for allocating the ptr.
 * The dual function of this call is mrcuda_record_cudaHostAlloc.
 */
MRCUDASym_t *mrcuda_replay_cudaFreeHost(MRCUDAGPU_t *mrcudaGPU, void *ptr);

/**
 * Replay a cudaSetDeviceFlags call.
 */
void mrcuda_replay_cudaSetDeviceFlags(MRCUDAGPU_t *mrcudaGPU, MRecord_t *record);

/**
 * Download the content of active memory regions to the native device.
 * Exit and report error if an error is found.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t that the sync mem will be performed on.
 */
void mrcuda_sync_mem(MRCUDAGPU_t *mrcudaGPU);

/**
 * Simulate cuda streams on the native CUDA so that the number of streams are equaled to that of rCUDA.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t that the simulate stream will be performed on.
 */
void mrcuda_simulate_stream(MRCUDAGPU_t *mrcudaGPU);

/**
 * Simulate cuCtxCreate on the specified gpuID.
 * If mrcudaGPU->status == MRCUDA_GPU_STATUS_HELPER, ask the helper to handle the command.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t.
 * @param gpuID the ID of the GPU a context will be created on.
 * @return 0 on success; -1 otherwise.
 */
int mrcuda_simulate_cuCtxCreate(MRCUDAGPU_t *mrcudaGPU, int gpuID);

#endif
