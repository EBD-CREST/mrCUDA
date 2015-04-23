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
 * @return a ptr to a MRCUDASharedMemLocalInfo_t on success. NULL otherwise.
 */
MRCUDASharedMemLocalInfo_t *mhelper_mem_malloc(size_t size)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo = calloc(1, sizeof(MRCUDASharedMemLocalInfo_t));
    if (sharedMemInfo == NULL)
        goto __mhelper_mem_malloc_err_0;
    sharedMemInfo->sharedMem.key = generate_key();
    if ((sharedMemInfo->shmid = shmget(*key, size, IPC_CREAT | IPC_EXCL | 0600)) <= 0)
        goto __mhelper_mem_malloc_err_1;
    if ((sharedMemInfo->startAddr = shmat(shmid, NULL, 0)) == NULL)
        goto __mhelper_mem_malloc_err_2;
    sharedMemInfo->sharedMem.size = size;
    return sharedMemInfo;

__mhelper_mem_malloc_err_2:
    shmctl(sharedMemInfo->shmid, IPC_RMID, NULL);
__mhelper_mem_malloc_err_1:
    free(sharedMemInfo);
__mhelper_mem_malloc_err_0:
    return NULL;
}

/**
 * Detach and destroy the shared region specified by the sharedMemInfo.
 * @param sharedMemInfo the information of the shared region.
 * @return 0 on success; other number otherwise.
 */
int mhelper_mem_free(MRCUDASharedMemLocalInfo_t *sharedMemInfo)
{
    int ret = shmctl(sharedMemInfo->shmid, IPC_RMID, NULL);
    if (ret == 0)
        free(sharedMemInfo);
    return ret;
}

/**
 * Get the memory region associated with the specified sharedMem.
 * @param sharedMem the minimum information of the shared region.
 * @return a ptr to a MRCUDASharedMemLocalInfo_t on success. NULL otherwise.
 */
MRCUDASharedMemLocalInfo_t *mhelper_mem_get(MRCUDASharedMem_t sharedMem)
{
    MRCUDASharedMemLocalInfo_t *sharedMemInfo = calloc(1, sizeof(MRCUDASharedMemLocalInfo_t));
    if (sharedMemInfo == NULL)
        goto __mhelper_mem_get_err_0;
    if ((sharedMemInfo->shmid = shmget(sharedMem.key, sharedMem.size, 0666)) <= 0)
        goto __mhelper_mem_get_err_1;
    if ((sharedMemInfo->startAddr = shmat(shmid, NULL, 0)) == NULL)
        goto __mhelper_mem_get_err_1;
    sharedMemInfo->sharedMem = sharedMem;
    return sharedMemInfo;

__mhelper_mem_get_err_1:
    free(sharedMemInfo);
__mhelper_mem_get_err_0:
    return NULL;
}

/**
 * Detach the shared region specified by the sharedMemInfo.
 * @param sharedMemInfo the information of the shared region.
 * @return 0 on success; another number otherwise.
 */
int mhelper_mem_detach(MRCUDASharedMemLocalInfo_t *sharedMemInfo)
{
    return shmdt(sharedMemInfo->startAddr);
}

