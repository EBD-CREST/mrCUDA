#include <stdlib.h>

#include "../src/record.h"

START_TEST(test_mrcuda_record_cudaRegisterFatBinary)
{
    void *fatCubin = malloc(sizeof(void *) * 10);
    mrcuda_record_cudaRegisterFatBinary(fatCubin);
    ck_assert(mrcudaRecordHeadPtr != NULL);
    ck_assert(mrcudaRecordTailPtr != NULL);
    ck_assert(strcmp(mrcudaRecordTailPtr->functionName, "cudaRegisterFatBinary") == 0);
    ck_assert(mrcudaRecordTailPtr->replayFunction == &mrcuda_replay_cudaRegisterFatBinary);
    ck_assert(mrcudaRecordTailPtr->data.cudaRegisterFatBinary.fatCubin == fatCubin);
    free(fatCubin);
}
END_TEST

Suite *comm_suit(void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create("Record");

	tc_core = tcase_create("Core");

	tcase_add_test(tc_core, test_mrcuda_record_cudaRegisterFatBinary);
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
