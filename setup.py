import distutils
from distutils.core import setup, Extension, Command
from distutils.command.build_ext import build_ext

import os
from subprocess import Popen
import sys
import numpy
import glob
import shutil
import platform

class build_ext_subclass(build_ext):
    boolean_options = build_ext.boolean_options + ['use-system-fitsio']
    
    user_options = build_ext.user_options + \
            [('use-system-fitsio', None, 
              "Use the cfitsio installed in the system"),

             ('system-fitsio-includedir=', None,
              "Path to look for cfitsio header; default is the system search path."),

             ('system-fitsio-libdir=', None,
              "Path to look for cfitsio library; default is the system search path."),
            ]
    #cfitsio_version = '3280patch'
    cfitsio_version = '3370patch'
    cfitsio_dir = 'cfitsio%s' % cfitsio_version

    def initialize_options(self):
        self.use_system_fitsio = False
        self.system_fitsio_includedir = None
        self.system_fitsio_libdir = None
        build_ext.initialize_options(self)    

    def finalize_options(self):

        build_ext.finalize_options(self)    

        self.cfitsio_build_dir = os.path.join(self.build_temp, self.cfitsio_dir)
        self.cfitsio_zlib_dir = os.path.join(self.cfitsio_build_dir,'zlib')

        if self.use_system_fitsio:
            if self.system_fitsio_includedir:
                self.include_dirs.insert(0, self.system_fitsio_includedir)
            if self.system_fitsio_libdir:
                self.library_dirs.insert(0, self.system_fitsio_libdir)
        else:
            # We defer configuration of the bundled cfitsio to build_extensions
            # because we will know the compiler there.
            self.include_dirs.insert(0, self.cfitsio_build_dir)


    def build_extensions(self):
        if not self.use_system_fitsio:

            # Use the compiler for building python to build cfitsio
            # for maximized compatibility.

            # there is some issue with non-aligned data with optimizations
            # set to '-O3' on some versions of gcc.  It appears to be
            # a disagreement between gcc 4 and gcc 5

            CCold=self.compiler.compiler
            
            CC=[]
            for val in CCold:
                if val=='-O3':
                    print("replacing '-O3' with '-O2' to address "
                          "gcc bug")
                    val='-O2'
                CC.append(val) 
                    
            self.configure_cfitsio(
                CC=CC, 
                ARCHIVE=self.compiler.archiver, 
                RANLIB=self.compiler.ranlib,
            )

            # If configure detected bzlib.h, we have to link to libbz2

            if '-DHAVE_BZIP2=1' in open(os.path.join(self.cfitsio_build_dir, 'Makefile')).read():
                self.compiler.add_library('bz2')

            self.compile_cfitsio()

            # link against the .a library in cfitsio; 
            # It should have been a 'static' library of relocatable objects (-fPIC), 
            # since we use the python compiler flags

            link_objects = glob.glob(os.path.join(self.cfitsio_build_dir,'*.a'))

            self.compiler.set_link_objects(link_objects)

            # Ultimate hack: append the .a files to the dependency list
            # so they will be properly rebuild if cfitsio source is updated.
            for ext in self.extensions:
                ext.depends += link_objects
        else:
            # Include bz2 by default?  Depends on how system cfitsio was built.
            # FIXME: use pkg-config to tell if bz2 shall be included ?
            self.compiler.add_library('cfitsio')
            pass

        # call the original build_extensions

        build_ext.build_extensions(self)

    def configure_cfitsio(self, CC=None, ARCHIVE=None, RANLIB=None):

        # prepare source code and run configure
        def copy_update(dir1,dir2):
            f1 = os.listdir(dir1)
            for f in f1:
                path1 = os.path.join(dir1,f)
                path2 = os.path.join(dir2,f)

                if os.path.isdir(path1):
                    if not os.path.exists(path2):
                        os.makedirs(path2)
                    copy_update(path1,path2)
                else:
                    if not os.path.exists(path2):
                        shutil.copy(path1,path2)
                    else:
                        stat1 = os.stat(path1)
                        stat2 = os.stat(path2)
                        if (stat1.st_mtime > stat2.st_mtime):
                            shutil.copy(path1,path2)


        if not os.path.exists('build'):
            ret=os.makedirs('build')

        if not os.path.exists(self.cfitsio_build_dir):
            ret=os.makedirs(self.cfitsio_build_dir)

        copy_update(self.cfitsio_dir, self.cfitsio_build_dir)

        makefile = os.path.join(self.cfitsio_build_dir, 'Makefile')

        if os.path.exists(makefile):
            # Makefile already there
            return

        args = ''
        args += ' CC="%s"' % ' '.join(CC[:1])
        args += ' CFLAGS="%s"' % ' '.join(CC[1:])
    
        if ARCHIVE:
            args += ' ARCHIVE="%s"' % ' '.join(ARCHIVE)
        if RANLIB:
            args += ' RANLIB="%s"' % ' '.join(RANLIB)

        p = Popen("sh ./configure --with-bzip2 " + args, 
                shell=True, cwd=self.cfitsio_build_dir)
        p.wait()
        if p.returncode != 0:
            raise ValueError("could not configure cfitsio %s" % self.cfitsio_version)

    def compile_cfitsio(self):
        p = Popen("make", 
                shell=True, cwd=self.cfitsio_build_dir)
        p.wait()
        if p.returncode != 0:
            raise ValueError("could not compile cfitsio %s" % self.cfitsio_version)


include_dirs=[numpy.get_include()]
    

sources = ["fitsio/fitsio_pywrap.c"]
data_files=[]

ext=Extension("fitsio._fitsio_wrap", 
              sources, include_dirs=include_dirs)

description = ("A full featured python library to read from and "
               "write to FITS files.")

long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read()

classifiers = ["Development Status :: 5 - Production/Stable"
               ,"License :: OSI Approved :: GNU General Public License (GPL)"
               ,"Topic :: Scientific/Engineering :: Astronomy"
               ,"Intended Audience :: Science/Research"
              ]

setup(name="fitsio", 
      version="0.9.10",
      description=description,
      long_description=long_description,
      license = "GPL",
      classifiers=classifiers,
      url="https://github.com/esheldon/fitsio",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      install_requires=['numpy'],
      packages=['fitsio'],
      data_files=data_files,
      ext_modules=[ext],
      cmdclass = {
        "build_ext": build_ext_subclass,
      }
     )



