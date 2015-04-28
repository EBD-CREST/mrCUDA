#ifndef __MRCUDA__HEADER__
#define __MRCUDA__HEADER__

#include <cuda_runtime.h>
#include <glib.h>
#include "common.h"
#include "datatypes.h"

extern MRCUDASym_t *mrcudaSymNvidia;
extern MRCUDASym_t *mrcudaSymRCUDA;

extern int mrcudaNumGPUs;
extern MRCUDAGPU_t *mrcudaGPUList;

extern GHashTable *mrcudaGPUThreadMap;

extern MRCUDAState_e mrcudaState;

/**
 * Get the GPU assigned to the calling thread.
 * @return a pointer to the assigned GPU.
 */
MRCUDAGPU_t *mrcuda_get_current_gpu();

/**
 * Set the GPU assigned to the calling thread.
 * @param device virtual device ID.
 */
void mrcuda_set_current_gpu(int device);


/**
 * Initialize mrCUDA.
 * Print error and terminate the program if an error occurs.
 */
void mrcuda_init();

/**
 * Finalize mrCUDA.
 */
int mrcuda_fini();

/**
 * Switch the specified mrcudaGPU from rCUDA to native.
 * @param mrcudaGPU a ptr to the mrcudaGPU to be switched.
 * @param toGPUNumber the native GPU number to be moved to.
 */
void mrcuda_switch(MRCUDAGPU_t *mrcudaGPU, int toGPUNumber);

/**
 * Create a barrier such that subsequent calls are blocked until the barrier is released.
 * @param mrcudaGPU a ptr to the GPU a barrier will be created on.
 */
void mrcuda_function_call_lock(MRCUDAGPU_t *mrcudaGPU);

/**
 * Release the barrier; thus, allow subsequent calls to be processed normally.
 * @param mrcudaGPU a ptr to the GPU the barrier will be released.
 */
void mrcuda_function_call_release(MRCUDAGPU_t *mrcudaGPU);

#endif
