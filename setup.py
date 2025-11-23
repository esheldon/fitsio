#
# setup script for fitsio, using setuptools
#
# c.f.
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

from __future__ import print_function
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import warnings
import tempfile
import tarfile
import sys
import os
import subprocess
from subprocess import PIPE
import glob
import shutil

if "FITSIO_FAIL_ON_BAD_PATCHES" in os.environ:
    if os.environ["FITSIO_FAIL_ON_BAD_PATCHES"].lower() in ["false", "0"]:
        FITSIO_FAIL_ON_BAD_PATCHES = False
    else:
        FITSIO_FAIL_ON_BAD_PATCHES = True
else:
    FITSIO_FAIL_ON_BAD_PATCHES = True

if "--use-system-fitsio" in sys.argv:
    del sys.argv[sys.argv.index("--use-system-fitsio")]
    USE_SYSTEM_FITSIO = True
else:
    USE_SYSTEM_FITSIO = False or "FITSIO_USE_SYSTEM_FITSIO" in os.environ

if "--system-fitsio-includedir" in sys.argv or any(
    a.startswith("--system-fitsio-includedir=") for a in sys.argv
):
    if "--system-fitsio-includedir" in sys.argv:
        ind = sys.argv.index("--system-fitsio-includedir")
        SYSTEM_FITSIO_INCLUDEDIR = sys.argv[ind + 1]
        del sys.argv[ind + 1]
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


if "--system-fitsio-libdir" in sys.argv or any(
    a.startswith("--system-fitsio-libdir=") for a in sys.argv
):
    if "--system-fitsio-libdir" in sys.argv:
        ind = sys.argv.index("--system-fitsio-libdir")
        SYSTEM_FITSIO_LIBDIR = sys.argv[ind + 1]
        del sys.argv[ind + 1]
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


def _print_msg(text):
    print("\n" + "=" * 79 + f"\n{text}\n" + "=" * 79, flush=True)


class build_ext_subclass(build_ext):
    cfitsio_version = '4.6.3'
    cfitsio_dir = 'cfitsio-%s' % cfitsio_version

    def finalize_options(self):
        build_ext.finalize_options(self)

        self.cfitsio_build_dir = os.path.join(
            self.build_temp, self.cfitsio_dir
        )
        self.cfitsio_zlib_dir = os.path.join(self.cfitsio_build_dir, 'zlib')
        self.cfitsio_patch_dir = os.path.join(self.build_temp, 'patches')

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
                        _print_msg("removing ccache from the compiler options")
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
                _have_bzip2 = False
                _have_curl = False
                for line in _makefile.splitlines():
                    for _part in line.split("="):
                        for _eqpart in _part.split():
                            if "-lbz2" in _eqpart:
                                _have_bzip2 = True
                            if "-lcurl" in _eqpart:
                                _have_curl = True
                if _have_bzip2:
                    _print_msg(
                        "found -lbz2 in Makefile\n"
                        "linking Python extension to bzip2"
                    )
                    self.compiler.add_library('bz2')
                    self.compiler.define_macro('FITSIO_HAS_BZIP2_SUPPORT')
                else:
                    _print_msg(
                        "did not find -lbz2 in Makefile\n"
                        "bzip2 support is disabled"
                    )

                if _have_curl:
                    _print_msg(
                        "found -lcurl in Makefile\n"
                        "linking Python extension to curl"
                    )
                    self.compiler.add_library('curl')
                    self.compiler.define_macro('FITSIO_HAS_CURL_SUPPORT')
                else:
                    _print_msg(
                        "did not find -lcurl in Makefile\n"
                        "curl support is disabled"
                    )

            self.compile_cfitsio()

            # link against the .a library in cfitsio;
            # It should have been a 'static' library of relocatable objects
            # (-fPIC), since we use the python compiler flags

            link_objects = glob.glob(
                os.path.join(self.cfitsio_build_dir, '*.o')
            )

            self.compiler.set_link_objects(link_objects)

            # Ultimate hack: append the .a files to the dependency list
            # so they will be properly rebuild if cfitsio source is updated.
            for ext in self.extensions:
                ext.depends += link_objects
        else:
            self.compiler.add_library('cfitsio')

            # Check if system cfitsio was compiled with bzip2 and/or curl
            if self.check_system_cfitsio_objects('bzip2'):
                _print_msg(
                    "found bz2 symbol in system cfitsio library\n"
                    "linking Python extension to bzip2"
                )
                self.compiler.add_library('bz2')
                self.compiler.define_macro('FITSIO_HAS_BZIP2_SUPPORT')
            else:
                _print_msg(
                    "did not find bz2 symbol in system cfitsio library\n"
                    "bzip2 support is disabled"
                )

            if self.check_system_cfitsio_objects('curl_'):
                _print_msg(
                    "found curl_ symbol in system cfitsio library\n"
                    "linking Python extension to curl"
                )
                self.compiler.add_library('curl')
                self.compiler.define_macro('FITSIO_HAS_CURL_SUPPORT')
            else:
                _print_msg(
                    "did not find curl_ symbol in system cfitsio library\n"
                    "curl support is disabled"
                )

            self.compiler.define_macro('FITSIO_USING_SYSTEM_FITSIO')

            self.compiler.add_library('z')

        # fitsio requires libm as well.
        self.compiler.add_library('m')

        # call the original build_extensions
        build_ext.build_extensions(self)

    def patch_cfitsio(self):
        _print_msg("patching cfitsio")

        try:
            subprocess.check_call(["patch", "-v"])
        except subprocess.CalledProcessError as e:
            warnings.warn(
                "`patch` command not found! "
                "Some bugs in cfitsio may not be fixed! "
                "See the patches we carry at "
                "https://github.com/esheldon/fitsio/tree/master/patches."
            )
            if FITSIO_FAIL_ON_BAD_PATCHES:
                raise e
            else:
                return

        patches = glob.glob('%s/*.patch' % self.cfitsio_patch_dir)
        for patch in patches:
            fname = os.path.basename(patch.replace('.patch', ''))
            try:
                subprocess.check_call(
                    [
                        "patch",
                        "-N",
                        "--dry-run",
                        "%s/%s" % (self.cfitsio_build_dir, fname),
                        patch,
                    ]
                )
            except subprocess.CalledProcessError as e:
                warnings.warn(
                    "Failed to apply patch: " + os.path.basename(patch)
                )
                if FITSIO_FAIL_ON_BAD_PATCHES:
                    raise e
            else:
                subprocess.check_call(
                    [
                        "patch",
                        "%s/%s" % (self.cfitsio_build_dir, fname),
                        patch,
                    ],
                )

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
                        if stat1.st_mtime > stat2.st_mtime:
                            shutil.copy(path1, path2)

        if not os.path.exists('build'):
            os.makedirs('build')

        if not os.path.exists(self.cfitsio_build_dir):
            os.makedirs(self.cfitsio_build_dir)

        if not os.path.exists(self.cfitsio_patch_dir):
            os.makedirs(self.cfitsio_patch_dir)

        if sys.version_info.major >= 3 and sys.version_info.minor >= 12:
            tar_kwargs = {"filter": "fully_trusted"}
        else:
            tar_kwargs = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            if os.path.exists(self.cfitsio_dir) and os.path.isdir(
                self.cfitsio_dir
            ):
                _print_msg(
                    "using cfitsio source code from "
                    f"{self.cfitsio_dir} for debugging"
                )
                copy_update(
                    self.cfitsio_dir,
                    self.cfitsio_build_dir,
                )
            else:
                with tarfile.open(self.cfitsio_dir + ".tar.gz", "r:gz") as tar:
                    tar.extractall(path=tmpdir, **tar_kwargs)
                    copy_update(
                        os.path.join(tmpdir, self.cfitsio_dir),
                        self.cfitsio_build_dir,
                    )

        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open("zlib.tar.gz", "r:gz") as tar:
                tar.extractall(path=tmpdir, **tar_kwargs)
                copy_update(
                    os.path.join(tmpdir, "zlib"), self.cfitsio_build_dir
                )

        copy_update('patches', self.cfitsio_patch_dir)

        # we patch the source in the buil dir to avoid mucking with the repo
        self.patch_cfitsio()

        makefile = os.path.join(self.cfitsio_build_dir, 'Makefile')

        if os.path.exists(makefile):
            # Makefile already there
            _print_msg("found Makefile so not running configure!")
            return
        else:
            _print_msg("configuring cfitsio")

        # the latest cfitsio build system links its example
        # programs (e.g., `cookbook`) against the shared library.
        # when we use `-fvisibility=hidden` in the CFLAGS (
        # needed to hide the cfitsio symbols in the python `.so``),
        # the linking against the shared library fails.
        # so we disable shared libraries with (`--disable-shared``)
        # and add `-fPIC` to the flags to ensure the python `.so`
        # works properly later
        args = [
            '--without-fortran',
            '--disable-shared',
        ]
        our_cflags = "-fPIC -fvisibility=hidden"

        if "FITSIO_BZIP2_DIR" in os.environ:
            if not os.environ["FITSIO_BZIP2_DIR"]:
                args += ["--with-bzip2"]
            else:
                args += ['--with-bzip2="%s"' % os.environ["FITSIO_BZIP2_DIR"]]
        else:
            # let autoconf detect if we have bzip2
            args += ['--with-bzip2']

        env = {}
        env.update(os.environ)

        if CC is not None:
            env["CC"] = ' '.join(CC[:1])
            env["CFLAGS"] = ' '.join(CC[1:]) + our_cflags
        else:
            if "CFLAGS" in os.environ:
                env["CFLAGS"] = os.environ["CFLAGS"] + " " + our_cflags
            else:
                env["CFLAGS"] = our_cflags

        if ARCHIVE:
            env["ARCHIVE"] = ' '.join(ARCHIVE)
        if RANLIB:
            env["RANLIB"] = ' '.join(RANLIB)

        res = subprocess.run(
            ["sh", "./configure"] + args,
            cwd=self.cfitsio_build_dir,
            env=env,
        )
        if res.returncode != 0:
            with open(
                os.path.join(self.cfitsio_build_dir, "config.log")
            ) as fp:
                logfile = fp.read()
            raise ValueError(
                "could not configure cfitsio %s: config.log:\n\n%s"
                % (
                    self.cfitsio_version,
                    logfile,
                )
            )

    def compile_cfitsio(self):
        _print_msg("building cfitsio")
        res = subprocess.run(
            "make",
            cwd=self.cfitsio_build_dir,
        )
        if res.returncode != 0:
            raise ValueError(
                "could not compile cfitsio %s" % self.cfitsio_version
            )

    def check_system_cfitsio_objects(self, obj_name):
        for lib_dir in self.library_dirs:
            if os.path.isfile('%s/libcfitsio.a' % (lib_dir)):
                res = subprocess.run(
                    ["nm", "-g", "%s/libcfitsio.a" % lib_dir],
                    stdout=PIPE,
                    stderr=PIPE,
                )
                for line in res.stdout.decode("utf-8").splitlines():
                    if obj_name in line:
                        return True

                return False
        return False


sources = ["fitsio/fitsio_pywrap.c"]

ext = Extension("fitsio._fitsio_wrap", sources, include_dirs=['numpy'])

description = (
    "A full featured python library to read from and write to FITS files."
)

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
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    license="GPL",
    classifiers=classifiers,
    url="https://github.com/esheldon/fitsio",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    setup_requires=['numpy>=1.7', 'setuptools-scm>=8'],
    install_requires=['numpy>=1.7'],
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext_subclass},
    use_scm_version={
        "version_file": "fitsio/_version.py",
        "version_file_template": "__version__ = '{version}'\n",
    },
)
