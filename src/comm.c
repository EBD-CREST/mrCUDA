#include <pthread.h>

#include "comm.h"

static pthread_t *__mrcudaCommListeningThread;

static char *__mrcudaCommSocketPath;
static void (*__mrcudaCommCallback)(void);

/**
 * This function creates a UNIX socket specified by path and initializes it.
 * If it fails to do so for any reasons, it returns the error number; otherwise, return 0
 * @param path path of the socket to be created.
 * @return 0 if success, otherwise the error number.
 */
static int __mrcuda_comm_socket_init(char *path);

/**
 * Terminate the socket.
 */
static void __mrcuda_comm_socket_terminate();

/**
 * This is the main loop for repeatedly listening to a signal.
 * If it receives a correct signal, it terminates the socket and calls the callback.
 * This function should be called from a different thread since it blocks the execution.
 */
static void *__mrcuda_comm_listening_main_loop(void *arg);

/**
 * This function starts listening to a signal that tells the system to switch to native CUDA.
 * After it receives the signal, this function calls the callback and terminates the socket.
 * This function executes the listening process in a different thread; thus, it returns almost immediately.
 * Note: if the signal is not well form, this function will simply skips that signal and not calls the callback.
 * @param path path for creating a new UNIX socket for listening to the signal.
 * @param callback the function that will be called after received a signal.
 * @return 0 if success, the error number otherwise.
 */
int mrcuda_comm_listen_for_signal(char *path, void (*callback)(void))
{
    int ret = 0;
    if((ret = __mrcuda_comm_socket_init(path)) != 0)
        return ret;
    __mrcudaCommCallback = callback;
    if((ret = pthread_create(__mrcudaCommListeningThread, NULL, &__mrcuda_comm_listening_main_loop, NULL)) != 0)
        __mrcuda_comm_socket_terminate();
    return ret;
}

