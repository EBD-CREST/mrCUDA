#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <check.h>

#include "../src/comm.h"

static int __getSignalFlag = 0;

void process_signal(void)
{
    __getSignalFlag = 1;
}

START_TEST(test_mrcuda_comm_listen_for_signal)
{
    char *path = "/tmp/mrcuda.pipe";
	int fd;
    int ret;

	unlink(path);
    ret = mrcuda_comm_listen_for_signal(path, &process_signal);
    ck_assert(ret == 0);

    fd = open(path, O_WRONLY);
    write(fd, "1", sizeof("1"));
    close(fd);

    while(!__getSignalFlag)
        sleep(1);
}
END_TEST

Suite *comm_suit(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("Comm");

	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, test_mrcuda_comm_listen_for_signal);
	suite_add_tcase(s, tc_core);

	return s;
}

int main(void)
{
	int number_failed;
	Suite *s;
	SRunner *sr;

	s = comm_suit();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	number_failed = srunner_ntests_failed(sr);
	srunner_free(sr);

	return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
