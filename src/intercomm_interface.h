#ifndef __MRUCDA_INTERCOMM_INTERFACE__HEADER__
#define __MRCUDA_INTERCOMM_INTERFACE__HEADER__

#include <cuda_runtime.h>

#include "datatypes.h"

/**
 * Initialize a handler with a helper process.
 * @param handler output of initialized handler.
 * @param process a ptr to a helper process.
 * @return 0 on success; -1 otherwise.
 */
int mhelper_int_init(MRCUDASym_t **handler, MHelperProcess_t *process);


/* Interfaces */

/**
 * Create a context on the helper process.
 * @param mrcudaGPU a ptr to a MRCUDAGPU_t a context will be created on.
 * @return 0 on success; -1 otherwise.
 */
int mhelper_int_cuCtxCreate_internal(MRCUDAGPU_t *mrcudaGPU);

void **mhelper_int_cudaRegisterFatBinary(void *fatCubin);
void **mhelper_int_cudaRegisterFatBinary_internal(MRCUDAGPU_t *mrcudaGPU, void *fatCubin);

void mhelper_int_cudaUnregisterFatBinary(void **fatCubinHandle);
void mhelper_int_cudaUnregisterFatBinary_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle);

void mhelper_int_cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
void mhelper_int_cudaRegisterVar_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);

void mhelper_int_cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);
void mhelper_int_cudaRegisterTexture_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);

void mhelper_int_cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
void mhelper_int_cudaRegisterFunction_internal(MRCUDAGPU_t *mrcudaGPU, void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

cudaError_t mhelper_int_cudaLaunch(const void *func);
cudaError_t mhelper_int_cudaLaunch_internal(MRCUDAGPU_t *mrcudaGPU, const void *func);

cudaError_t mhelper_int_cudaHostAlloc(void **pHost,  size_t size,  unsigned int flags);

cudaError_t mhelper_int_cudaDeviceReset(void);
cudaError_t mhelper_int_cudaDeviceReset_internal(MRCUDAGPU_t *mrcudaGPU);

cudaError_t mhelper_int_cudaDeviceSynchronize(void);
cudaError_t mhelper_int_cudaDeviceSynchronize_internal(MRCUDAGPU_t *mrcudaGPU);

cudaError_t mhelper_int_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t mhelper_int_cudaGetDeviceProperties_internal(MRCUDAGPU_t *mrcudaGPU, struct cudaDeviceProp *prop, int device);

cudaError_t mhelper_int_cudaMalloc(void **devPtr, size_t size);
cudaError_t mhelper_int_cudaMalloc_internal(MRCUDAGPU_t *mrcudaGPU, void **devPtr, size_t size);

cudaError_t mhelper_int_cudaFreeHost(void *ptr);

cudaError_t mhelper_int_cudaFree(void *devPtr);
cudaError_t mhelper_int_cudaFree_internal(MRCUDAGPU_t *mrcudaGPU, void *devPtr);

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

#endif /* __MRCUDA_INTERCOMM_INTERFACE__HEADER__ */

