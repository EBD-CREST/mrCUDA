AC_INIT([mrcuda], [0.1.0], [markthub.p.aa@m.titech.ac.jp])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
AC_PROG_CC
AC_PROG_INSTALL
AM_PROG_CC_C_O
AC_ENABLE_SHARED(yes)
AC_ENABLE_STATIC(no)
PKG_CHECK_MODULES([CHECK], [check >= 0.9.4])
ACX_PTHREAD
LT_INIT

AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug], [Enable debug output]))

AS_IF([test "x$enable_debug" = "xyes"], [
    AC_DEFINE(DEBUG, 1, [Define if --enable-debug option is found.])
])

NVCC="nvcc"
AC_SUBST(NVCC)

AC_CONFIG_MACRO_DIR([build-aux])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile tests/Makefile])
AC_OUTPUT
