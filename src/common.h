#ifndef __MRCUDA_COMMON__HEADER__
#define __MRCUDA_COMMON__HEADER__

#include <config.h>

#define STR(x) #x

#if DEBUG
    #define DPRINTF(fmt, ...) \
        do {fprintf(stderr, "FILE: " __FILE__ ", LINE: %d, " fmt, __LINE__, ##__VA_ARGS__);} while(0);
#else
    #define DPRINTF(fmt, ...) \
        do {;;} while(0);
#endif

#define REPORT_ERROR_AND_EXIT(...) \
    do { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
    } while(0);

#endif
