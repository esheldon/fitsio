import distutils
from distutils.core import setup, Extension, Command
from distutils.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils.dir_util import remove_tree
import os
import sys
import numpy
import glob
import shutil
import platform
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

#Packaged version
cfitsio_version = '3370'
cfitsio_dir = 'cfitsio%s' % cfitsio_version
cfitsio_build_dir = os.path.join('build',cfitsio_dir)
cfitsio_zlib_dir = os.path.join(cfitsio_build_dir,'zlib')
package_basedir = os.path.abspath(os.curdir)

if "--use-system-fitsio" not in sys.argv:
    compile_fitsio_package = True
else:
    print "Using SYSTEM CFITSIO"
    compile_fitsio_package = False
    sys.argv.remove("--use-system-fitsio")

extra_objects = None
include_dirs=[numpy.get_include()]
if platform.system()=='Darwin':
    extra_compile_args=['-arch','x86_64']
    extra_link_args=['-arch','x86_64']
else:
    extra_compile_args=[]
    extra_link_args=[]
    

# when using "extra_objects" in Extension, changes in the objects do *not*
# cause a re-link!  The only way I know is to force a recompile by removing the
# directory
build_libdir=glob.glob(os.path.join('build','lib*'))
if len(build_libdir) > 0:
    shutil.rmtree(build_libdir[0])

# Sub-class the "clean" command to also remove the 
# cfitsio build dir
class CleanCfitsio(clean):
    def run(self):
        if os.path.exists(cfitsio_build_dir):
            remove_tree(cfitsio_build_dir)
        return clean.run(self)

#Sub-class the build_ext command to also
#build CFITSIO.  Only used if --use-system-fitsio 
#is not specified: then the original build_ext is used.
class BuildExtCfitsio(build_ext):
    def run(self):
        print "Custom builder"
        self.build_cfitsio()

        #Add the extra files build as part of CFITSIO
        #to the extension object
        ext = self.distribution.ext_modules[0]
        ext.extra_objects += glob.glob(
            os.path.join(cfitsio_build_dir,'*.o'))
        ext.extra_objects += glob.glob(
            os.path.join(cfitsio_zlib_dir,'*.o'))
        return build_ext.run(self)

    def build_cfitsio(self):

        makefile = os.path.join(cfitsio_build_dir, 'Makefile')

        if not os.path.exists('build'):
            ret=os.makedirs('build')

        if not os.path.exists(cfitsio_build_dir):
            ret=os.makedirs(cfitsio_build_dir)

        self.copy_update_cfitsio(cfitsio_dir, cfitsio_build_dir)

        if not os.path.exists(makefile):
            self.configure_cfitsio()

        self.compile_cfitsio()


    def configure_cfitsio(self):
        os.chdir(cfitsio_build_dir)
        ret=os.system('sh ./configure')
        if ret != 0:
            raise ValueError("could not configure cfitsio %s" % cfitsio_version)
        os.chdir(package_basedir)

    def compile_cfitsio(self):
        os.chdir(cfitsio_build_dir)
        ret=os.system('make')
        if ret != 0:
            raise ValueError("could not compile cfitsio %s" % cfitsio_version)
        os.chdir(package_basedir)


    @classmethod
    def copy_update_cfitsio(cls, dir1,dir2):
        f1 = os.listdir(dir1)
        for f in f1:
            path1 = os.path.join(dir1,f)
            path2 = os.path.join(dir2,f)

            if os.path.isdir(path1):
                if not os.path.exists(path2):
                    os.makedirs(path2)
                cls.copy_update_cfitsio(path1,path2)
            else:
                if not os.path.exists(path2):
                    shutil.copy(path1,path2)
                else:
                    stat1 = os.stat(path1)
                    stat2 = os.stat(path2)
                    if (stat1.st_mtime > stat2.st_mtime):
                        shutil.copy(path1,path2)




if compile_fitsio_package:
    include_dirs.append(cfitsio_dir)
    builder_class = BuildExtCfitsio
else:
    extra_link_args.append('-lcfitsio')
    builder_class = build_ext

sources = ["fitsio/fitsio_pywrap.c"]
data_files=[]

ext=Extension("fitsio._fitsio_wrap", 
              sources,
              extra_compile_args=extra_compile_args, 
              extra_link_args=extra_link_args,
              include_dirs=include_dirs)

description = ("A full featured python library to read from and "
               "write to FITS files.")

long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read()

classifiers = ["Development Status :: 5 - Production/Stable"
               ,"License :: OSI Approved :: GNU General Public License (GPL)"
               ,"Topic :: Scientific/Engineering :: Astronomy"
               ,"Intended Audience :: Science/Research"
              ]


setup(name="fitsio", 
      version="0.9.7",
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
      cmdclass = {"build_py":build_py, "build_ext":builder_class, "clean":CleanCfitsio})



