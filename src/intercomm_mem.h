#ifndef __MRCUDA_INTERCOMM_MEM__HEADER__
#define __MRCUDA_INTERCOMM_MEM__HEADER__

#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include "datatypes.h"

/**
 * Malloc memory on shared-memory region.
 * @param size the size of memory to be allocated.
 * @return a ptr to a MRCUDASharedMemLocalInfo_t on success. NULL otherwise.
 */
MRCUDASharedMemLocalInfo_t *mhelper_mem_malloc(size_t size);

/**
 * Detach and destroy the shared region specified by the sharedMemInfo.
 * @param sharedMemInfo the information of the shared region.
 * @return 0 on success; other number otherwise.
 */
int mhelper_mem_free(MRCUDASharedMemLocalInfo_t *sharedMemInfo);

/**
 * Get the memory region associated with the specified sharedMem.
 * @param sharedMem the minimum information of the shared region.
 * @return a ptr to a MRCUDASharedMemLocalInfo_t on success. NULL otherwise.
 */
MRCUDASharedMemLocalInfo_t *mhelper_mem_get(MRCUDASharedMem_t sharedMem);

/**
 * Detach the shared region specified by the sharedMemInfo.
 * @param sharedMemInfo the information of the shared region.
 * @return 0 on success; another number otherwise.
 */
int mhelper_mem_detach(MRCUDASharedMemLocalInfo_t *sharedMemInfo);

#endif /* __MRCUDA_INTERCOMM_MEM__HEADER__ */

