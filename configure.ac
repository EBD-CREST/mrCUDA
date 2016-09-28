AC_PREREQ([2.69])

AC_INIT([mrcuda], [7.0.0], [markthub.p.aa@m.titech.ac.jp])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([build-aux])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile tests/Makefile scripts/Makefile])
AC_ENABLE_SHARED(yes)
AC_ENABLE_STATIC(no)

AC_ARG_WITH([rcuda], [
    AS_HELP_STRING([--with-rcuda=[RCUDA_PATH]],
        [optionally specify the installation path of rCUDA.]
    )
])

AC_ARG_WITH([nvcc], [
    AS_HELP_STRING([--with-nvcc=[nvcc]],
        [optionally specify nvcc you want to use.]
    )], [
        AC_SUBST(NVCC, "${with_nvcc}")
    ]
)

AC_ARG_WITH([libcudart], [
    AS_HELP_STRING([--with-libcudart=[libcudart]],
        [optionally specify CUDA libcudart you want to use.]
    )], [
        AC_SUBST(NVIDIA_LIBCUDART, "${with_libcudart}")
    ]
)

AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug], [Enable debug output]))
AS_IF([test "x$enable_debug" = "xyes"], [
    AC_DEFINE(DEBUG, 1, [Define if --enable-debug option is found.])
])

# Checks for programs
AM_PROG_AR
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_INSTALL
AC_PROG_LN_S
AM_PATH_PYTHON([2.7])

if test x"${NVCC}" == x""; then
    AC_PATH_PROG(NVCC, nvcc, no)
    if test x"${NVCC}" == x"no"; then
        AC_MSG_ERROR([Cannot find nvcc.])
    fi
fi

if test x"${NVIDIA_LIBCUDART}" == x""; then
    AC_PATH_PROG(NVIDIA_LIBCUDART, [libcudart.so], no, [$LD_LIBRARY_PATH$PATH_SEPARATOR$LIBRARY_PATH])
    if test x"${NVIDIA_LIBCUDART}" == x"no"; then
        AC_MSG_ERROR([Cannot find CUDA's libcudart.so.])
    fi
fi

AC_PATH_PROG(RCUDA_RCUDACOMMIB, rCUDAcommIB.so, no, [$with_rcuda/lib$PATH_SEPARATOR$PATH])
if test x"${RCUDA_RCUDACOMMIB}" == x"no"; then
    AC_MSG_ERROR([Cannot find rCUDA. Make sure rCUDA is installed on your system.])
fi

AC_PATH_PROG(RCUDA_RCUDACOMMTCP, rCUDAcommTCP.so, no, [$with_rcuda/lib$PATH_SEPARATOR$PATH])
if test x"${RCUDA_RCUDACOMMTCP}" == x"no"; then
    AC_MSG_ERROR([Cannot find rCUDA. Make sure rCUDA is installed on your system.])
fi

AC_PATH_PROG(RCUDA_LIBCUDART, libcudart.so, no, [$with_rcuda/lib$PATH_SEPARATOR$PATH])
if test x"${RCUDA_LIBCUDART}" == x"no"; then
    AC_MSG_ERROR([Cannot find rCUDA. Make sure rCUDA is installed on your system.])
fi

# Checks for modules
PKG_CHECK_MODULES([CHECK], [check >= 0.9.4])
PKG_CHECK_MODULES([DEPS], [glib-2.0 >= 2.24.1])
LT_INIT

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lcuda':
AC_CHECK_LIB([cuda], [cuCtxCreate])
# FIXME: Replace `main' with a function in `-lcudart':
AC_CHECK_LIB([cudart], [cudaMemcpy])
# FIXME: Replace `main' with a function in `-ldl':
AC_CHECK_LIB([dl], [dlsym])
# FIXME: Replace `main' with a function in `-lpthread':
AC_CHECK_LIB([pthread], [pthread_mutex_init])

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T

# Checks for library functions.
AC_FUNC_FORK
AC_FUNC_MALLOC
AC_CHECK_FUNCS([dup2 gettimeofday mempcpy mkfifo strtol])

AC_OUTPUT
