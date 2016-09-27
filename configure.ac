AC_PREREQ([2.69])

AC_INIT([mrcuda], [7.0.0], [markthub.p.aa@m.titech.ac.jp])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign])
AM_PROG_AR
AC_PROG_CC
AC_PROG_INSTALL
AM_PROG_CC_C_O
AC_ENABLE_SHARED(yes)
AC_ENABLE_STATIC(no)
PKG_CHECK_MODULES([CHECK], [check >= 0.9.4])
PKG_CHECK_MODULES([DEPS], [glib-2.0 >= 2.24.1])
ACX_PTHREAD
LT_INIT

AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug], [Enable debug output]))

AS_IF([test "x$enable_debug" = "xyes"], [
    AC_DEFINE(DEBUG, 1, [Define if --enable-debug option is found.])
])

NVCC="nvcc"
AC_SUBST(NVCC)

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

AC_CONFIG_MACRO_DIR([build-aux])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile tests/Makefile])
AC_OUTPUT
