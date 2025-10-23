#
# RPM spec file for package cfitsio
# (i.e. support to build packages on RedHat systems like openSUSE, Fedora,
# CentOS, AlmaLinux and others).
#
# Typical use is to copy the tar.gz file into the SOURCES directory
# of the RPM root directory and the spec-file into the SPECS directory,
# to install with the package manager (zypper, dnf,...) the required
# packages
#   zypper install libcurl-devel zlib-devel automake autoconf
#   cp cfitsio-4.6.2.tar.gz ..../SOURCES
#   cp cfitsio.spec ..../SPECS
# and to call rpmbuild:
#   rpmbuild -ba cfitsio.spec
# The result will then be available in the RPMS directory of the RPM root.
#
# Richard J. Mathar (2025-06-20)
#


%define tar_ver 4.6.3
%define so_ver 10
Name:           cfitsio
Version:        4.6.3
Release:        0
Summary:        Library for manipulating FITS data files
License:        ISC
URL:            https://heasarc.gsfc.nasa.gov/fitsio/
Source0:        https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/%{name}-%{tar_ver}.tar.gz
BuildRequires:  gcc-fortran
BuildRequires:  libcurl-devel
BuildRequires:  pkgconfig
BuildRequires:  zlib-devel
BuildRequires:  automake
BuildRequires:  autoconf
BuildRequires:  libtool

%description
CFITSIO is a library of C and Fortran subroutines for reading and writing data
files in FITS (Flexible Image Transport System) data format. CFITSIO provides
simple high-level routines for reading and writing FITS files that insulate the
programmer from the internal complexities of the FITS format. CFITSIO also
provides many advanced features for manipulating and filtering the information
in FITS files.

This package contains some FITS image compression and decompression utilities.

%package devel
Summary:        Headers required when building programs against cfitsio library
Requires:       libcfitsio%{so_ver} = %{version}
Requires:       pkgconfig
%if 0%{?centos_version} != 700
Suggests:       cfitsio-devel-doc = %{version}
%endif
# libcfitsio-devel was last used in openSUSE 13.1 (version 3.350)
Provides:       libcfitsio-devel = %{version}
Obsoletes:      libcfitsio-devel <= 3.350

%description devel
This package contains headers required when building programs against cfitsio
library.

%package devel-doc
Summary:        Documentation for the cfitsio library
# libcfitsio-doc was last used in openSUSE 12.1 (version 3.280)
Obsoletes:      libcfitsio-doc <= 3.280
# libcfitsio-devel-doc was last used in openSUSE 13.1 (version 3.350)
Provides:       libcfitsio-devel-doc = %{version}
Obsoletes:      libcfitsio-devel-doc <= 3.350

%description devel-doc
This package contains documentation for the cfitsio library.

%package -n libcfitsio%{so_ver}
Summary:        Library for manipulating FITS data files

%description -n libcfitsio%{so_ver}
CFITSIO is a library of C and Fortran subroutines for reading and writing data
files in FITS (Flexible Image Transport System) data format. CFITSIO provides
simple high-level routines for reading and writing FITS files that insulate the
programmer from the internal complexities of the FITS format. CFITSIO also
provides many advanced features for manipulating and filtering the information
in FITS files.

%prep
%setup -q -n %{name}-%{tar_ver}

%build
# lines below contain fixes for pkgconfig file bnc#546004, some of them are already fixed by upstream
# so please drop them if they are not needed (in next round of updates)
# Add include dir, multithreading support, zlib dependency
sed -i 's|Cflags: -I${includedir}|Cflags: -D_REENTRANT -I${includedir} -I${includedir}/%{name}|' cfitsio.pc.in
sed -i 's|@LIBS@ -lm|@LIBS@ -lz -lm|' cfitsio.pc.in
# fixes complaints of fedora 35 opensuse build service: avoid -Wl,rpath flags
sed -i 's|-Wl,-rpath,|-L|' configure.ac
sed -i 's|@rpath/||' configure.ac
autoreconf -i -f -s


%configure --enable-reentrant --docdir=%{_docdir}/%{name} --includedir=%{_includedir}/%{name}

%{make_build}
make fpack %{?_smp_mflags}
make funpack %{?_smp_mflags}

%check
# testsuite
# On openSUSE 15.6 this fails with an error concerning invalid libc.so.6
# ELF headers; try to recover by piping everything through the colon command
make testprog %{?_smp_mflags}
LD_LIBRARY_PATH=. ./testprog > testprog.lis || :
diff testprog.lis testprog.out || :
cmp testprog.fit testprog.std || :

%install
mkdir -p %{buildroot}%{_licensedir}/%{name}
mkdir -p %{buildroot}%{_docdir}/%{name}
make DESTDIR=%{buildroot} CFITSIO_INCLUDE=%{buildroot}%{_includedir}/%{name} install
install licenses/License.txt %{buildroot}%{_licensedir}/cfitsio
install -t %{buildroot}%{_docdir}/cfitsio docs/*.pdf docs/*.doc

# Remove static libraries
rm -f %{buildroot}%{_libdir}/libcfitsio.a
# do not distribute libtool files
rm %{buildroot}/%{_libdir}/libcfitsio.la
# do not distribute cookbook and speed (too ambiguous names)
rm %{buildroot}%{_bindir}/{cookbook,speed,smem}

%post -n libcfitsio%{so_ver} -p /sbin/ldconfig
%postun -n libcfitsio%{so_ver} -p /sbin/ldconfig

%files
%doc %{_docdir}/cfitsio/fpackguide.pdf
%license %{_licensedir}/cfitsio/License.txt
%{_bindir}/fpack
%{_bindir}/funpack
%{_bindir}/fitscopy
%{_bindir}/fitsverify
%{_bindir}/imcopy

%files devel
%{_includedir}/%{name}/
%{_libdir}/libcfitsio.so
%{_libdir}/pkgconfig/cfitsio.pc

%files devel-doc
%doc %{_docdir}/cfitsio/{cfitsio.pdf,cfortran.doc,fitsio.doc,fitsio.pdf,quick.pdf}

%files -n libcfitsio%{so_ver}
%{_libdir}/libcfitsio.so.%{so_ver}*

%changelog
