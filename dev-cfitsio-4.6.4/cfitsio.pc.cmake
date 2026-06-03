prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=${prefix}
libdir="@CMAKE_INSTALL_FULL_LIBDIR@"
includedir="@CMAKE_INSTALL_FULL_INCLUDEDIR@"

Name: cfitsio
Description: FITS File Subroutine Library
URL: https://heasarc.gsfc.nasa.gov/fitsio/
Version: @CFITSIO_MAJOR@.@CFITSIO_MINOR@.@CFITSIO_MICRO@
Libs: -L${libdir} -lcfitsio
Libs.private: @PKG_CONFIG_LIBS@ -lm
Cflags: -I${includedir}
