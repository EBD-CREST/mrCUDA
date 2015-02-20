#ifndef __MRCUDA_COMM__HEADER__
#define __MRCUDA_COMM__HEADER__

/**
 * This function starts listening to a signal that tells the system to switch to native CUDA.
 * After it receives the signal, this function calls the callback and terminates the socket.
 * This function executes the listening process in a different thread; thus, it returns almost immediately.
 * Note: if the signal is not well form, this function will simply skips that signal and not calls the callback.
 * @param path path for creating a new UNIX socket for listening to the signal.
 * @param callback the function that will be called after received a signal.
 * @return 0 if success, the error number otherwise.
 */
int mrcuda_comm_listen_for_signal(char *path, void (*callback)(void));

#endif
