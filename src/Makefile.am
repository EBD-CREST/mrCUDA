bin_PROGRAMS = mhelper
mhelper_SOURCES = mhelper.c intercomm_mem.c
mhelper_LDFLAGS = -lcuda -lcudart $(DEPS_LIBS)
mhelper_CPPFLAGS = -I/usr/local/cuda/include $(DEPS_CFLAGS)

lib_LTLIBRARIES = libcudart.la
libcudart_la_SOURCES = comm.c interface.c mrcuda.c record.c intercomm.c intercomm_mem.c intercomm_interface.c
libcudart_la_LDFLAGS = -avoid-version -shared -ldl $(DEPS_LIBS)
libcudart_la_CPPFLAGS = -I/usr/local/cuda/include $(DEPS_CFLAGS)

lib_LIBRARIES = libcomm.a
libcomm_a_SOURCES = comm.c comm.h common.h
libcomm_a_CPPFLAGS = -lpthread -pthread $(DEPS_CFLAGS)

install-exec-hook:
	${LN_S} ${RCUDA_RCUDACOMMIB} ${libdir}/rCUDAcommIB.so
	${LN_S} ${RCUDA_RCUDACOMMTCP} ${libdir}/rCUDAcommTCP.so

