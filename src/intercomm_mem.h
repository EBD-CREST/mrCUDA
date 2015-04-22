#ifndef __MRCUDA_INTERCOMM_MEM__HEADER__
#define __MRCUDA_INTERCOMM_MEM__HEADER__

#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

/**
 * Malloc memory on shared-memory region.
 * @param size the size of memory to be allocated.
 * @param key return value of associating key.
 * @return a ptr to the allocated region on success. NULL otherwise.
 */
void *mhelper_mem_malloc(size_t size, key_t *key);

/**
 * Detach and destroy the shared region specified by the key.
 * @param key the key of the shared region.
 * @return 0 on success; other number otherwise.
 */
int mhelper_mem_free(key_t key);

/**
 * Get the memory region associated with the specified key.
 * @param key the key of the shared region.
 * @return a ptr to the shared region on success. NULL otherwise.
 */
void *mhelper_mem_get(key_t key);

/**
 * Detach the shared region specified by the key.
 * @param key the key of the shared region.
 * @return 0 on success; another number otherwise.
 */
int mhelper_mem_detach(key_t key);

#endif /* __MRCUDA_INTERCOMM_MEM__HEADER__ */

