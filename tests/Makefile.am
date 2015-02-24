TESTS = check_comm
check_PROGRAMS = check_comm
check_comm_SOURCES = check_comm.c $(top_builddir)/src/comm.h
check_comm_CFLAGS = @CHECK_CFLAGS@ -pthread 
check_comm_LDADD = $(top_builddir)/src/libcomm.a @CHECK_LIBS@
check_comm_LDFLAGS = -pthread -lpthread

