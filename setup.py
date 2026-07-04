#
# setup script for fitsio, using setuptools
#
# c.f.
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

from __future__ import print_function
from setuptools import setup, Extension
from setuptools.command.build_ext import (
    build_ext,
    new_compiler,
    customize_compiler,
)

import warnings
import tempfile
import tarfile
import shlex
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


if "--system-fitsio-has-curl" in sys.argv:
    del sys.argv[sys.argv.index("--system-fitsio-has-curl")]
    SYSTEM_FITSIO_HAS_CURL = True
else:
    SYSTEM_FITSIO_HAS_CURL = (
        False or "FITSIO_SYSTEM_FITSIO_HAS_CURL" in os.environ
    )


if "--system-fitsio-has-bzip2" in sys.argv:
    del sys.argv[sys.argv.index("--system-fitsio-has-bzip2")]
    SYSTEM_FITSIO_HAS_BZIP2 = True
else:
    SYSTEM_FITSIO_HAS_BZIP2 = (
        False or "FITSIO_SYSTEM_FITSIO_HAS_BZIP2" in os.environ
    )


def _print_msg(text):
    print("\n" + "=" * 79 + f"\n{text}\n" + "=" * 79, flush=True)


def _copy_update(dir1, dir2):
    f1 = os.listdir(dir1)
    for f in f1:
        path1 = os.path.join(dir1, f)
        path2 = os.path.join(dir2, f)

        if os.path.isdir(path1):
            if not os.path.exists(path2):
                os.makedirs(path2)
            _copy_update(path1, path2)
        else:
            if not os.path.exists(path2):
                shutil.copy(path1, path2)
            else:
                stat1 = os.stat(path1)
                stat2 = os.stat(path2)
                if stat1.st_mtime > stat2.st_mtime:
                    shutil.copy(path1, path2)


class build_ext_subclass(build_ext):
    cfitsio_version = '4.6.4'
    cfitsio_dir = 'cfitsio-%s' % cfitsio_version

    def finalize_options(self):
        build_ext.finalize_options(self)

        self.cfitsio_build_dir = os.path.abspath(
            os.path.join(self.build_temp, self.cfitsio_dir)
        )
        self.cfitsio_patch_dir = os.path.abspath(
            os.path.join(self.build_temp, 'patches')
        )
        self.cfitsio_cmake_build_dir = os.path.join(
            self.cfitsio_build_dir, "build"
        )
        self.cfitsio_cmake_prefix_dir = os.path.join(
            self.cfitsio_build_dir, "prefix"
        )

        if USE_SYSTEM_FITSIO:
            if SYSTEM_FITSIO_INCLUDEDIR is not None:
                for pth in SYSTEM_FITSIO_INCLUDEDIR.split(os.pathsep):
                    _print_msg(f"Adding include directory '{pth}'")
                    self.include_dirs.insert(0, pth)
            if SYSTEM_FITSIO_LIBDIR is not None:
                for pth in SYSTEM_FITSIO_LIBDIR.split(os.pathsep):
                    _print_msg(f"Adding lib directory '{pth}'")
                    self.library_dirs.insert(0, pth)
        else:
            if os.name == "nt":
                self.include_dirs.insert(
                    0, os.path.join(self.cfitsio_cmake_prefix_dir, "include")
                )
                self.library_dirs.insert(
                    0, os.path.join(self.cfitsio_cmake_prefix_dir, "lib")
                )
            else:
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
            if os.name != "nt":
                self.build_cfitsio_unix()
            else:
                self.build_cfitsio_win()
                # on windows we build a full static lib so we have to tell
                # the compiler to link against it
                self.compiler.add_library('cfitsio')
        else:
            self.compiler.add_library('cfitsio')

            # Check if system cfitsio was compiled with bzip2 and/or curl
            if SYSTEM_FITSIO_HAS_BZIP2 or self.check_system_cfitsio_objects(
                'bzip2'
            ):
                _print_msg("linking Python extension to bzip2")
                if os.name == "nt":
                    self.compiler.add_library('libbz2')
                else:
                    self.compiler.add_library('bz2')
                self.compiler.define_macro('FITSIO_HAS_BZIP2_SUPPORT')
            else:
                _print_msg("bzip2 support is disabled")

            if SYSTEM_FITSIO_HAS_CURL or self.check_system_cfitsio_objects(
                'curl_'
            ):
                _print_msg("linking Python extension to curl")
                if os.name == "nt":
                    self.compiler.add_library('libcurl')
                else:
                    self.compiler.add_library('curl')
                self.compiler.define_macro('FITSIO_HAS_CURL_SUPPORT')
            else:
                _print_msg("curl support is disabled")

            self.compiler.define_macro('FITSIO_USING_SYSTEM_FITSIO')

            self.compiler.add_library('z')

        # fitsio requires libm as well, but do not need to link it
        # explicitly on windows
        if os.name != "nt":
            self.compiler.add_library('m')

        # call the original build_extensions
        build_ext.build_extensions(self)

    def build_cfitsio_win(self):
        self.extract_cfitsio()

        # we patch the source in the build dir to avoid mucking with the repo
        self.patch_cfitsio()

        os.makedirs(self.cfitsio_cmake_build_dir, exist_ok=True)
        os.makedirs(self.cfitsio_cmake_prefix_dir, exist_ok=True)

        env = {}
        env.update(os.environ)
        # make a new instance to get name without changing current instance
        tmp_cc = new_compiler()
        customize_compiler(tmp_cc)
        tmp_cc.initialize(self.plat_name)
        env["CC"] = tmp_cc.cc
        _print_msg("setting windows compiler to " + env["CC"])
        cmake_cmd = [
            "cmake",
        ]
        if "CMAKE_ARGS" in os.environ:
            cmake_cmd += shlex.split(os.environ["CMAKE_ARGS"], posix=False)
        cmake_cmd += [
            "-G",
            "NMake Makefiles",
            f"-DCMAKE_INSTALL_PREFIX={self.cfitsio_cmake_prefix_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=Off",
            "..",
        ]
        _print_msg("windows cmake command: " + repr(cmake_cmd))
        subprocess.run(
            cmake_cmd,
            check=True,
            cwd=self.cfitsio_cmake_build_dir,
            env=env,
        )
        subprocess.run(
            ["nmake"],
            check=True,
            cwd=self.cfitsio_cmake_build_dir,
        )
        subprocess.run(
            ["nmake", "install"],
            check=True,
            cwd=self.cfitsio_cmake_build_dir,
        )

        # figure out if we have curl support
        r = subprocess.run(
            [
                "dumpbin",
                "/symbols",
                os.path.join(
                    self.cfitsio_cmake_prefix_dir, "lib", "cfitsio.lib"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        found_curl = False
        for line in (r.stdout + r.stderr).splitlines():
            if "External" in line.split() and "_curl_" in line:
                _print_msg("found curl symbol: " + line.strip())
                found_curl = True

        if found_curl:
            _print_msg(
                "found curl in symbols\nlinking Python extension to curl"
            )
            self.compiler.add_library('libcurl')
            self.compiler.define_macro('FITSIO_HAS_CURL_SUPPORT')
        else:
            _print_msg(
                "did not find curl in symbols\ncurl support is disabled"
            )

    def build_cfitsio_unix(self):
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

        self.configure_cfitsio_unix(
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
                    "did not find -lbz2 in Makefile\nbzip2 support is disabled"
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
                    "did not find -lcurl in Makefile\ncurl support is disabled"
                )

        self.compile_cfitsio_unix()

        # link against the .a library in cfitsio;
        # It should have been a 'static' library of relocatable objects
        # (-fPIC), since we use the python compiler flags

        link_objects = glob.glob(os.path.join(self.cfitsio_build_dir, '*.o'))

        self.compiler.set_link_objects(link_objects)

        # Ultimate hack: append the .a files to the dependency list
        # so they will be properly rebuild if cfitsio source is updated.
        for ext in self.extensions:
            ext.depends += link_objects

    def patch_cfitsio(self):
        _print_msg("patching cfitsio")

        if not os.path.exists(self.cfitsio_patch_dir):
            os.makedirs(self.cfitsio_patch_dir)

        _copy_update('patches', self.cfitsio_patch_dir)

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

        patches = glob.glob(os.path.join(self.cfitsio_patch_dir, '*.patch'))
        for patch in patches:
            fname = os.path.basename(patch.replace('.patch', ''))
            try:
                subprocess.check_call(
                    [
                        "patch",
                        "-N",
                        "--dry-run",
                        os.path.join(self.cfitsio_build_dir, fname),
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
                        os.path.join(self.cfitsio_build_dir, fname),
                        patch,
                    ],
                )

    def extract_cfitsio(self):
        if not os.path.exists(self.cfitsio_build_dir):
            os.makedirs(self.cfitsio_build_dir)

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
                _copy_update(
                    self.cfitsio_dir,
                    self.cfitsio_build_dir,
                )
            else:
                with tarfile.open(self.cfitsio_dir + ".tar.gz", "r:gz") as tar:
                    tar.extractall(path=tmpdir, **tar_kwargs)
                    _copy_update(
                        os.path.join(tmpdir, self.cfitsio_dir),
                        self.cfitsio_build_dir,
                    )

        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open("zlib.tar.gz", "r:gz") as tar:
                tar.extractall(path=tmpdir, **tar_kwargs)
                _copy_update(
                    os.path.join(tmpdir, "zlib"), self.cfitsio_build_dir
                )

    def configure_cfitsio_unix(self, CC=None, ARCHIVE=None, RANLIB=None):
        self.extract_cfitsio()

        # we patch the source in the build dir to avoid mucking with the repo
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
            '--enable-reentrant',
        ]
        # we use -fPIC and -fvisibility=hidden to ensure we build a linkable
        # static library with the symbols hidden
        # the -ffp-contract=off flag ensures we do not use FMA instructions on
        # ARM systems so that lossy floating point compression is reproducible
        our_cflags = "-fPIC -fvisibility=hidden -ffp-contract=off"

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

    def compile_cfitsio_unix(self):
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

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext_subclass},
)
