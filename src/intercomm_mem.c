#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <stdlib.h>

#include "intercomm_mem.h"

#define DEV_RANDOM "/dev/urandom"

static int initRand = 0;

/**
 * Generate a key to be associated with a shared memory region.
 * @return a key.
 */
static key_t generate_key()
{
    FILE *f;
    unsigned int seed;

    if (!initRand) {
        f = fopen(DEV_RANDOM, "r");
        fread(&seed, sizeof(unsigned int), 1, f);
        fclose(f);
        srand(seed);
        initRand = !initRand;
    }
    return (key_t)rand();
}

/**
 * Malloc memory on shared-memory region.
 * @param size the size of memory to be allocated.
 * @param key return value of associating key.
 * @return a ptr to the allocated region on success. NULL otherwise.
 */
void *mhelper_mem_malloc(size_t size, key_t *key)
{
    *key = generate_key();
    shmget(*key, size, IPC_CREAT | IPC_EXCL | 0600);
}

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
