#ifndef __MRCUDA_RECORD__HEADER__
#define __MRCUDA_RECORD__HEADER__

#include <cuda_runtime.h>

typedef struct MRecord
{
    char *functionName;
    union Data
    {
        struct cudaMemcpyToSymbol
        {
            void *symbol;
            size_t count;
            size_t offset;
            enum cudaMemcpyKind kind;
        } cudaMemcpyToSymbol;
        struct cudaMemcpy
        {
            void *dst;
            size_t count;
            enum cudaMemcpyKind kind;
        } cudaMemcpy;
        struct cudaMemset
        {
            void *devPtr;
            int value;
            size_t count;
        } cudaMemset;
        struct cudaMalloc
        {
            size_t size;
        } cudaMalloc;
        struct cudaFree
        {
            void *devPtr;
        } cudaFree;
        struct cudaBindTexture
        {
            size_t *offset;
            struct textureReference *textref;
            void *devPtr;
            struct cudaChannelFormatDesc *desc;
            size_t size;
        } cudaBindTexture;
        struct cudaCreateChannelDesc
        {
            int x;
            int y;
            int z;
            int w;
            enum cudaChannelFormatKind f;
        } cudaCreateChannelDesc;
    } data;
}

#endif
