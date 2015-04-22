#ifndef __MRCUDA_INTERCOMM__HEADER__
#define __MRCUDA_INTERCOMM__HEADER__

#include "datatypes.h"
#include "intercomm_mem.h"

/**
 * Create a helper process and assign the mrcudaGPU to it.
 * @param mrcudaGPU the GPU information to assign to the created process.
 * @return a ptr to the created process on success; NULL otherwise.
 */
MHelperProcess_t *mhelper_create(MRCUDAGPU_t *mrcudaGPU);

/**
 * Destroy the helper process.
 * @param process the process to be destroyed.
 * @return 0 on success; another number otherwise.
 */
int mhelper_destroy(MHelperProcess_t *process);

/**
 * Ask the process to execute the command.
 * @param process the process that will execute the specified command.
 * @param command the command to be executed on the process.
 * @return the result of the execution.
 */
MHelperResult_t mhelper_call(MHelperProcess_t *process, MHelperCommand_t command);

#endif /* __MRCUDA_INTERCOMM__HEADER__ */

