AC_INIT([mrcuda], [0.1.0], [markthub.p.aa@m.titech.ac.jp])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
AC_PROG_CC
AC_PROG_INSTALL
AM_PROG_CC_C_O
AC_ENABLE_SHARED(yes)
AC_ENABLE_STATIC(no)
PKG_CHECK_MODULES([CHECK], [check >= 0.9.4])
LT_INIT

NVCC="nvcc"
AC_SUBST(NVCC)

AC_CONFIG_MACRO_DIR([build-aux])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
