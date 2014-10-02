// This file is not needed for rCUDA to work
#ifndef rCUDA_H
#define rCUDA_H
#include <cuda_runtime.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

#define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)
template<class T> inline void setupArgument(T param, int *offset) {
        ALIGN_UP(*offset, __alignof(param));
        cudaSetupArgument(&param, sizeof(param), *offset);
        *offset += sizeof(param);
}

#endif
