#
# setup script for fitsio, using setuptools
#
# c.f.
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

from __future__ import print_function
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import sys
import os
import subprocess
from subprocess import Popen, PIPE
import glob
import shutil


if "--use-system-fitsio" in sys.argv:
    del sys.argv[sys.argv.index("--use-system-fitsio")]
    USE_SYSTEM_FITSIO = True
else:
    USE_SYSTEM_FITSIO = False or "FITSIO_USE_SYSTEM_FITSIO" in os.environ

if (
    "--system-fitsio-includedir" in sys.argv
    or any(a.startswith("--system-fitsio-includedir=") for a in sys.argv)
):
    if "--system-fitsio-includedir" in sys.argv:
        ind = sys.argv.index("--system-fitsio-includedir")
        SYSTEM_FITSIO_INCLUDEDIR = sys.argv[ind+1]
        del sys.argv[ind+1]
        del sys.argv[ind]
    else:
        for ind in range(len(sys.argv)):
            if sys.argv[ind].startswith("--system-fitsio-includedir="):
                break
        SYSTEM_FITSIO_INCLUDEDIR = sys.argv[ind].split("=", 1)[1]
        del sys.argv[ind]
else:
    SYSTEM_FITSIO_INCLUDEDIR = os.environ.get(
        "FITSIO_SYSTEM_FITSIO_INCLUDEDIR",
        None,
    )


if (
    "--system-fitsio-libdir" in sys.argv
    or any(a.startswith("--system-fitsio-libdir=") for a in sys.argv)
):
    if "--system-fitsio-libdir" in sys.argv:
        ind = sys.argv.index("--system-fitsio-libdir")
        SYSTEM_FITSIO_LIBDIR = sys.argv[ind+1]
        del sys.argv[ind+1]
        del sys.argv[ind]
    else:
        for ind in range(len(sys.argv)):
            if sys.argv[ind].startswith("--system-fitsio-libdir="):
                break
        SYSTEM_FITSIO_LIBDIR = sys.argv[ind].split("=", 1)[1]
        del sys.argv[ind]
else:
    SYSTEM_FITSIO_LIBDIR = os.environ.get(
        "FITSIO_SYSTEM_FITSIO_LIBDIR",
        None,
    )


class build_ext_subclass(build_ext):
    cfitsio_version = '4.4.1-20240617'
    cfitsio_dir = 'cfitsio-%s' % cfitsio_version

    def finalize_options(self):

        build_ext.finalize_options(self)

        self.cfitsio_build_dir = os.path.join(
            self.build_temp, self.cfitsio_dir)
        self.cfitsio_zlib_dir = os.path.join(
            self.cfitsio_build_dir, 'zlib')
        self.cfitsio_patch_dir = os.path.join(
            self.build_temp, 'patches')

        if USE_SYSTEM_FITSIO:
            if SYSTEM_FITSIO_INCLUDEDIR is not None:
                self.include_dirs.insert(0, SYSTEM_FITSIO_INCLUDEDIR)
            if SYSTEM_FITSIO_LIBDIR is not None:
                self.library_dirs.insert(0, SYSTEM_FITSIO_LIBDIR)
        else:
            # We defer configuration of the bundled cfitsio to build_extensions
            # because we will know the compiler there.
            self.include_dirs.insert(0, self.cfitsio_build_dir)

    def run(self):
        # For extensions that require 'numpy' in their include dirs,
        # replace 'numpy' with the actual paths
        import numpy
        np_include = numpy.get_include()

        for extension in self.extensions:
            if 'numpy' in extension.include_dirs:
                idx = extension.include_dirs.index('numpy')
                extension.include_dirs.insert(idx, np_include)
                extension.include_dirs.remove('numpy')

        build_ext.run(self)

    def build_extensions(self):
        if not USE_SYSTEM_FITSIO:

            # Use the compiler for building python to build cfitsio
            # for maximized compatibility.

            # turns out we need to set the include dirs here too
            # directly for the compiler
            self.compiler.include_dirs.insert(0, self.cfitsio_build_dir)

            CCold = self.compiler.compiler
            if 'ccache' in CCold:
                CC = []
                for val in CCold:
                    if val == 'ccache':
                        print("removing ccache from the compiler options")
                        continue

                    CC.append(val)
            else:
                CC = None

            self.configure_cfitsio(
                CC=CC,
                ARCHIVE=self.compiler.archiver,
                RANLIB=self.compiler.ranlib,
            )

            # If configure detected bzlib.h, we have to link to libbz2
            with open(os.path.join(self.cfitsio_build_dir, 'Makefile')) as fp:
                _makefile = fp.read()
                if '-DHAVE_BZIP2=1' in _makefile:
                    self.compiler.add_library('bz2')
                if '-DCFITSIO_HAVE_CURL=1' in _makefile:
                    self.compiler.add_library('curl')

            self.compile_cfitsio()

            # link against the .a library in cfitsio;
            # It should have been a 'static' library of relocatable objects
            # (-fPIC), since we use the python compiler flags

            link_objects = glob.glob(
                os.path.join(self.cfitsio_build_dir, '*.o'))

            self.compiler.set_link_objects(link_objects)

            # Ultimate hack: append the .a files to the dependency list
            # so they will be properly rebuild if cfitsio source is updated.
            for ext in self.extensions:
                ext.depends += link_objects
        else:
            self.compiler.add_library('cfitsio')

            # Check if system cfitsio was compiled with bzip2 and/or curl
            if self.check_system_cfitsio_objects('bzip2'):
                self.compiler.add_library('bz2')
            if self.check_system_cfitsio_objects('curl_'):
                self.compiler.add_library('curl')

            # Make sure the external lib has the fits_use_standard_strings
            # function. If not, then define a macro to tell the wrapper
            # to always return False.
            if not self.check_system_cfitsio_objects(
                    '_fits_use_standard_strings'):
                self.compiler.define_macro(
                    'FITSIO_PYWRAP_ALWAYS_NONSTANDARD_STRINGS')

            self.compiler.add_library('z')

        # fitsio requires libm as well.
        self.compiler.add_library('m')

        # call the original build_extensions

        build_ext.build_extensions(self)

    def patch_cfitsio(self):
        patches = glob.glob('%s/*.patch' % self.cfitsio_patch_dir)
        for patch in patches:
            fname = os.path.basename(patch.replace('.patch', ''))
            try:
                subprocess.check_call(
                    'patch -N --dry-run %s/%s %s' % (
                        self.cfitsio_build_dir, fname, patch),
                    shell=True)
            except subprocess.CalledProcessError:
                pass
            else:
                subprocess.check_call(
                    'patch %s/%s %s' % (
                        self.cfitsio_build_dir, fname, patch),
                    shell=True)

    def configure_cfitsio(self, CC=None, ARCHIVE=None, RANLIB=None):

        # prepare source code and run configure
        def copy_update(dir1, dir2):
            f1 = os.listdir(dir1)
            for f in f1:
                path1 = os.path.join(dir1, f)
                path2 = os.path.join(dir2, f)

                if os.path.isdir(path1):
                    if not os.path.exists(path2):
                        os.makedirs(path2)
                    copy_update(path1, path2)
                else:
                    if not os.path.exists(path2):
                        shutil.copy(path1, path2)
                    else:
                        stat1 = os.stat(path1)
                        stat2 = os.stat(path2)
                        if (stat1.st_mtime > stat2.st_mtime):
                            shutil.copy(path1, path2)

        if not os.path.exists('build'):
            os.makedirs('build')

        if not os.path.exists(self.cfitsio_build_dir):
            os.makedirs(self.cfitsio_build_dir)

        if not os.path.exists(self.cfitsio_patch_dir):
            os.makedirs(self.cfitsio_patch_dir)

        copy_update(self.cfitsio_dir, self.cfitsio_build_dir)
        copy_update('zlib', self.cfitsio_build_dir)
        copy_update('patches', self.cfitsio_patch_dir)

        # we patch the source in the buil dir to avoid mucking with the repo
        self.patch_cfitsio()

        makefile = os.path.join(self.cfitsio_build_dir, 'Makefile')

        if os.path.exists(makefile):
            # Makefile already there
            print("found Makefile so not running configure!", flush=True)
            return

        args = ''

        if "FITSIO_BZIP2_DIR" in os.environ:
            args += ' --with-bzip2="%s"' % os.environ["FITSIO_BZIP2_DIR"]
        else:
            args += ' --with-bzip2'

        if CC is not None:
            args += ' CC="%s"' % ' '.join(CC[:1])
            args += ' CFLAGS="%s -fvisibility=hidden"' % ' '.join(CC[1:])
        else:
            args += ' CFLAGS="${CFLAGS} -fvisibility=hidden"'

        if ARCHIVE:
            args += ' ARCHIVE="%s"' % ' '.join(ARCHIVE)
        if RANLIB:
            args += ' RANLIB="%s"' % ' '.join(RANLIB)

        p = Popen(
            "sh ./configure --enable-standard-strings " + args,
            shell=True,
            cwd=self.cfitsio_build_dir,
        )
        p.wait()
        if p.returncode != 0:
            raise ValueError(
                "could not configure cfitsio %s" % self.cfitsio_version)

    def compile_cfitsio(self):
        p = Popen(
            "make",
            shell=True,
            cwd=self.cfitsio_build_dir,
        )
        p.wait()
        if p.returncode != 0:
            raise ValueError(
                "could not compile cfitsio %s" % self.cfitsio_version)

    def check_system_cfitsio_objects(self, obj_name):
        for lib_dir in self.library_dirs:
            if os.path.isfile('%s/libcfitsio.a' % (lib_dir)):
                p = Popen(
                    "nm -g %s/libcfitsio.a | grep %s" % (lib_dir, obj_name),
                    shell=True,
                    stdout=PIPE,
                    stderr=PIPE,
                )
                if len(p.stdout.read()) > 0:
                    return True
                else:
                    return False
        return False


sources = ["fitsio/fitsio_pywrap.c"]

ext = Extension("fitsio._fitsio_wrap", sources, include_dirs=['numpy'])

description = ("A full featured python library to read from and "
               "write to FITS files.")

with open(os.path.join(os.path.dirname(__file__), "README.md")) as fp:
    long_description = fp.read()

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: GNU General Public License (GPL)",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Intended Audience :: Science/Research",
]

setup(
    name="fitsio",
    version="1.2.6",
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    license="GPL",
    classifiers=classifiers,
    url="https://github.com/esheldon/fitsio",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    setup_requires=['numpy'],
    install_requires=['numpy'],
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext_subclass}
)
