#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "record.h"

MRecord *mrcudaRecordHeadPtr = NULL;
MRecord *mrcudaRecordTailPtr = NULL;

/**
 * Allocate a new MRecord and appropriately link the new one with the previous.
 * The pointer to the new MRecord is recorded into recordPtr.
 * @param recordPtr the variable that the new MRecord will be recorded into.
 * @return 0 if success, otherwise error number.
 */
static int __mrcuda_record_new(MRecord **recordPtr)
{
    if((*recordPtr = calloc(1, sizeof(MRecord))) == NULL)
        goto __mrcuda_record_new_err_1;

    if(mrcudaRecordHeadPtr == NULL)
        mrcudaRecordHeadPtr = *recordPtr;
    else
        mrcudaRecordTailPtr->next = *recordPtr;

    mrcudaRecordTailPtr = *recordPtr;
    return 0;

__mrcuda_record_new_err_1:
    return -1;
}

/**
 * Record a cudaRegisterFatBinary call.
 */
void mrcuda_record_cudaRegisterFatBinary(void* fatCubin)
{
    MRecord *recordPtr;

    if(__mrcuda_record_new(&recordPtr) != 0)
        REPORT_ERROR_AND_EXIT("Cannot create a new MRecord.\n");

    recordPtr->functionName = "cudaRegisterFatBinary";
    recordPtr->data.cudaRegisterFatBinary.fatCubin = fatCubin;
    recordPtr->repalyFunc = &mrcuda_replay_cudaRegisterFatBinary;
}

/**
 * Replay a cudaRegisterFatBinary call.
 */
void mrcuda_replay_cudaRegisterFatBinary(MRecord* record)
{
}
