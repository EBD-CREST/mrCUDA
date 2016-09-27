#ifndef __MRCUDA_COMMON__HEADER__
#define __MRCUDA_COMMON__HEADER__

#include <config.h>

#include <stdio.h>
#include <glib.h>
#include <sys/time.h>

#if DEBUG
    #define DPRINTF(fmt, ...) \
        do {fprintf(stderr, "FILE: " __FILE__ ", LINE: %d, " fmt, __LINE__, ##__VA_ARGS__);} while(0)
#else
    #define DPRINTF(fmt, ...) \
        do {;;} while(0)
#endif

#define REPORT_ERROR_AND_EXIT(...) \
    do { \
        perror("FATAL ERROR"); \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
    } while(0)

#define STARTTIMMER() \
    struct timeval t1, t2; \
    gettimeofday(&t1, NULL);

#define ENDTIMMER(acctime) \
    gettimeofday(&t2, NULL); \
    acctime += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

#endif

