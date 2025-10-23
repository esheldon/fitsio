# ===========================================================================
#        https://www.gnu.org/software/autoconf-archive/ax_cfitsio.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CFITSIO
#
# DESCRIPTION
#
#   This macro searches for the headers and libraries of the cfitsio library
#   of https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html that typically have
#   been installed with package names like cfitsio-devel or libcfitsio-dev
#   on Linux operating systems.
#
#   If compilation of the test program works, the macro AC_SUBST's the
#   variables CFITSIO_LDFLAGS -L flags with the library directory,
#   CFITSIO_LIBS with the -l flag of the library and CFITSIO_CFLAGS with the
#   -I flag of the include directory.
#
#   If the m4-variable CFITSIO_HOME is set, the name of that directory is
#   given preference in the search order to find the (binary) library and
#   include directory.
#
#   If configure is called with the option --with-cfitsio=<DIR>, this
#   directory overrides the standard search directories and <DIR>/lib and
#   <DIR>/include are the only points where cfitsio is searched after. A
#   further fine-tuning is separately available with the configure options
#   --with-cfitsio-include=<DIR> and/or --with-cfitsio-libdir=<DIR> which
#   override only the default search list of directories for the
#   header-files or libraries.
#
#   For most compilations one would run separately
#
#     AC_CHECK_HEADERS([fitsio.h],[],AC_MSG_WARN([Warning: Could  not find cfitsio header fitsio.h ... ])])
#
#   to define (or not to define) HAVE_FITSIO_H; this is not included here.
#
# LICENSE
#
#   Copyright (c) 2025 Richard J. Mathar <mathar@mpia.de>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.

#serial 5

AC_DEFUN([AX_CFITSIO],
[
   ac_with_cfitsio=no
   dnl ${build_cpu} is only known if LT_INIT was used...
   dnl 'uname -i' returns unknown on Fedora, even if it is x86_64
   AC_CANONICAL_HOST
   if test $host_cpu = "x86_64"; then
      ac_search_lib_path="$CFITSIO_HOME/lib64 $CFITSIO_HOME/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib64 /opt/lib64 /opt/local/lib64 /usr/lib"
   else
      ac_search_lib_path="$CFITSIO_HOME/lib /usr/lib /usr/local/lib /opt/opencv/lib /opt/local/lib /usr/lib/i386-linux-gnu /usr/lib64"
   fi
   ac_search_include_path="$CFITSIO_HOME/include /usr/include/cfitsio/ /usr/include  /usr/local/include /opt/include"

   AC_ARG_WITH(cfitsio,
      [AS_HELP_STRING([--with-cfitsio=DIR], [root directory of cfitsio installation])],
      ac_with_cfitsio=$withval
      if test "x${ac_with_cfitsio}" != xyes && test "x${ac_with_cfitsio}" != xno; then
           ac_cfitsio_include="$withval/include"
           ac_cfitsio_libdir="$withval/lib"
      fi
   )

   AC_ARG_WITH([cfitsio-include],
      [AS_HELP_STRING([--with-cfitsio-include=DIR], [cfitsio header files are in DIR])],
      ac_with_cfitsio=$withval
      if test "x${ac_with_cfitsio}" != xyes && test "x${ac_with_cfitsio}" != xno; then
           ac_cfitsio_include="$withval"
      fi
   )

   AC_ARG_WITH([cfitsio-libdir],
      [AS_HELP_STRING([--with-cfitsio-libdir=DIR], [The cfitsio library is in DIR])],
      ac_with_cfitsio=$withval
      if test "x${ac_with_cfitsio}" != xyes && test "x${ac_with_cfitsio}" != xno; then
           ac_cfitsio_libdir="$withval"
      fi
   )

   if test "x${ac_cfitsio_libdir}" == "x"; then
      for ac_search_lib_test in $ac_search_lib_path; do
         if test -r "$ac_search_lib_test"; then
            ls $ac_search_lib_test/libcfitsio*  >/dev/null 2>&1
            if test $? -eq 0; then
               ac_cfitsio_libdir="$ac_search_lib_test/"
               break;
            fi
         fi
      done
   fi

   if test "x${ac_cfitsio_include}" == "x"; then
     for ac_search_include_test in $ac_search_include_path ; do
        if test -r "$ac_search_include_test/fitsio.h"; then
           ac_cfitsio_include="$ac_search_include_test"
           break;
        fi
     done
   fi

   AX_CFITSIO_LOCAL($ac_cfitsio_libdir, $ac_cfitsio_include)
])


dnl Subfunction which intermediatly sets the LDFLAGS and CXXFLAGS
dnl to the search paths found above, tries to compile-link a trivial
dnl program that uses the standard fitsio.h, and results in
dnl either failure or success of the entire search. The LDFLAGS
dnl and CXXFLAGS are restored to the content they had before the
dnl macro call.
AC_DEFUN([AX_CFITSIO_LOCAL],
[
   AC_REQUIRE([AX_SAVE_FLAGS])
   AC_REQUIRE([AX_RESTORE_FLAGS])

   AX_SAVE_FLAGS(cfitsio)

   AC_MSG_NOTICE(Check $1 $2)

   if test "$1"; then
       LDFLAGS="$LDFLAGS -L$1 -lcfitsio -pthread"
   fi
   if test "$2"; then
       CXXFLAGS="$CXXFLAGS -I$2"
   fi

   AC_LANG_PUSH(C)
   AC_MSG_CHECKING(for cfitsio in $1)

   AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <fitsio.h>]], [[ ; ]])],[success=yes],[success=no])

   AC_MSG_RESULT($success)
   AC_LANG_POP()

   AX_RESTORE_FLAGS(cfitsio)

   if test "x$success" != xyes; then
       AC_MSG_ERROR(cfitsio not (properly) installed. Get a recent version from: https://heasarc.gsfc.nasa.gov/docs/software/fitsio/ or the linux package manager)
   else
      if test "$1"; then
          AC_SUBST(CFITSIO_LDFLAGS, "-L$1")
          AC_SUBST(CFITSIO_LIBS, "-lcfitsio")
      fi
      if test "$2"; then
          AC_SUBST(CFITSIO_CFLAGS, "-I$2")
      fi
   fi
])
