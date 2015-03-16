#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

#include "comm.h"

#define LISTEN_BACKLOG 1

typedef struct __MRCUDAComm
{
	pthread_t listeningThread;

	char *path;
	void (*callback)(void);

	int fd;
} __MRCUDAComm;

static __MRCUDAComm __mrcudaCommObj;


/**
 * Terminate the socket.
 */
static void __mrcuda_comm_fini()
{
	DPRINTF("ENTER __mrcuda_comm_fini.\n");
	close(__mrcudaCommObj.fd);
	unlink(__mrcudaCommObj.path);
	free(__mrcudaCommObj.path);
	DPRINTF("EXIT __mrcuda_comm_fini.\n");
}

/**
 * This function creates a FIFO file specified by the path.
 * If it fails to do so for any reasons, it returns the error number; otherwise, return 0
 * @param path path of the FIFO file to be created.
 * @return 0 if success, otherwise the error number.
 */
static int __mrcuda_comm_init(char *path)
{
	DPRINTF("ENTER __mrcuda_comm_init.\n");

	DPRINTF("__mrcuda_comm_init allocates __mrcudaCommObj.path\n");
	if((__mrcudaCommObj.path = (char *)malloc(strlen(path) + 1)) == NULL)
		goto __mrcuda_comm_init_err_1;

	DPRINTF("__mrcuda_comm_init strcpy path.\n");
	strcpy(__mrcudaCommObj.path, path);

	DPRINTF("__mrcuda_comm_init mkfifo.\n");
	if(mkfifo(__mrcudaCommObj.path, 0666) == -1)
		goto __mrcuda_comm_init_err_2;

	DPRINTF("EXIT SUCCESS __mrcuda_comm_init.\n");
	return 0;

__mrcuda_comm_init_err_2:
	free(__mrcudaCommObj.path);
__mrcuda_comm_init_err_1:
	DPRINTF("EXIT FAILURE __mrcuda_comm_init.\n");
	return -1;
}


/**
 * This is the main loop for repeatedly listening to a signal.
 * If it receives a correct signal, it terminates the socket and calls the callback.
 * This function should be called from a different thread since it blocks the execution.
 */
static void *__mrcuda_comm_listening_main_loop(void *arg)
{
	DPRINTF("ENTER __mrcuda_comm_listening_main_loop.\n");

	#define BUF_SIZE 1

	char buf[BUF_SIZE];
	ssize_t readSize;

	DPRINTF("__mrcuda_comm_init open file.\n");
	if((__mrcudaCommObj.fd = open(__mrcudaCommObj.path, O_RDONLY)) == -1)
		goto __mrcuda_comm_listening_main_loop_err_1;

	while(1)
	{
		DPRINTF("__mrcuda_comm_listening_main_loop is waiting.\n");
		if((readSize = read(__mrcudaCommObj.fd, buf, BUF_SIZE)) == -1)
			goto __mrcuda_comm_listening_main_loop_err_1;
		DPRINTF("__mrcuda_comm_listening_main_loop received a signal.\n");
		if(strncmp(buf, "1", BUF_SIZE) == 0)
		{
			DPRINTF("__mrcuda_comm_listening_main_loop calls the callback.\n");
			__mrcudaCommObj.callback();
			break;
		}
	}

__mrcuda_comm_listening_main_loop_err_1:
	__mrcuda_comm_fini();

	DPRINTF("EXIT __mrcuda_comm_listening_main_loop.\n");

	#undef BUF_SIZE
}

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
	DPRINTF("ENTER mrcuda_comm_listen_for_signal.\n");
    int ret = 0;
    if((ret = __mrcuda_comm_init(path)) != 0)
        return ret;
	__mrcudaCommObj.callback = callback;

	DPRINTF("mrcuda_comm_listen_for_signal creates a thread.\n");
    if((ret = pthread_create(&(__mrcudaCommObj.listeningThread), NULL, &__mrcuda_comm_listening_main_loop, NULL)) != 0)
        __mrcuda_comm_fini();
	

	DPRINTF("EXIT mrcuda_comm_listen_for_signal.\n");
    return ret;
}

