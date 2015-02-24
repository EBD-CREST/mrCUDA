lib_LTLIBRARIES = libcudart.la
libcudart_la_SOURCES = cudart.c
libcudart_la_LDFLAGS = -avoid-version -shared -ldl
libcudart_la_CPPFLAGS = -I/usr/local/cuda/include

lib_LIBRARIES = libcomm.a
libcomm_a_SOURCES = comm.c comm.h common.h
libcomm_a_CPPFLAGS = -lpthread -pthread
