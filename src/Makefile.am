lib_LTLIBRARIES = libcudart.la
libcudart_la_SOURCES = comm.c interface.c mrcuda.c record.c
libcudart_la_LDFLAGS = -avoid-version -shared -ldl $(DEPS_LIBS)
libcudart_la_CPPFLAGS = -I/usr/local/cuda/include $(DEPS_CFLAGS)

lib_LIBRARIES = libcomm.a
libcomm_a_SOURCES = comm.c comm.h common.h
libcomm_a_CPPFLAGS = -lpthread -pthread
