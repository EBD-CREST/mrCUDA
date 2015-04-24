#include <stdlib.h>
#include <signal.h>
#include <cuda_runtime.h>

#include "common.h"
#include "datatypes.h"
#include "intercomm_mem.h"

static int gpuID;

/**
 * Handle system signal.
 * @param signum the signal number caught.
 */
void sig_handler(int signum)
{
    exit(EXIT_SUCCESS);
}

/**
 * Waiting for a command.
 * @param command a ptr to a MHelperCommand_t that a successfully received command will be written to.
 * @return 0 on success; another number otherwise.
 */
static int receive_command(MHelperCommand_t *command)
{
}

/**
 * Execute the specified command.
 * Output the result of the execution through the result variable.
 * @param command the command to be executed.
 * @param result a ptr to a MHelperResult_t to be outputted to.
 * @return 0 on success; another number otherwise.
 */
static int execute_command(MHelperCommand_t command, MHelperResult_t *result)
{
}

/**
 * Return the result to the caller.
 * @param result the result to be returned.
 * @return 0 on success; another number otherwise.
 */
static int sendback_result(MHelperResult_t result)
{
}

/**
 * Start mhelper's command listening server.
 */
static void run_forever(void)
{
    MHelperCommand_t command;
    MHelperResult_t result;

    while (1) {
        if (receive_command(&command) != 0)
            continue;
        if (execute_command(command, &result) != 0) {
            result.id = command.id;
            result.type = command.type;
            result.internalError = -1;
            result.cudaError = cudaSuccess;
        }
        sendback_result(result);
    }
}

/**
 * Main function
 */
int main(int argc, char **argv)
{
    char *endptr;

    if (argc != 2)
        REPORT_ERROR_AND_EXIT("The number of arguments should be exactly two.\n");
    gpuID = (int)strtol(argv[1], &endptr, 10);
    if (*endptr != '\0')
        REPORT_ERROR_AND_EXIT("The GPU ID argument is invalid.\n");

    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        REPORT_ERROR_AND_EXIT("Cannot register the signal handler.\n");

    run_forever();
    return 0;
}

