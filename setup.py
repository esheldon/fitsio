import distutils
from distutils.core import setup, Extension, Command
import os
import numpy
import glob
import shutil

package_basedir = os.path.abspath(os.curdir)

cfitsio_version = '3280patch'
cfitsio_dir = 'cfitsio%s' % cfitsio_version
cfitsio_build_dir = os.path.join('build',cfitsio_dir)

makefile = os.path.join(cfitsio_build_dir, 'Makefile')

def copy_update(dir1,dir2):
    f1=glob.glob(os.path.join(dir1,'*'))
    f1 = os.listdir(dir1)
    for f in f1:
        path1 = os.path.join(dir1,f)
        path2 = os.path.join(dir2,f)
        if not os.path.exists(path2):
            shutil.copy(path1,path2)
        else:
            stat1 = os.stat(path1)
            stat2 = os.stat(path2)
            if (stat1.st_mtime > stat2.st_mtime):
                shutil.copy(path1,path2)

def configure_cfitsio():
    os.chdir(cfitsio_build_dir)
    ret=os.system('./configure')
    if ret != 0:
        raise ValueError("could not configure cfitsio %s" % cfitsio_version)
    os.chdir(package_basedir)

def compile_cfitsio():
    os.chdir(cfitsio_build_dir)
    ret=os.system('make')
    if ret != 0:
        raise ValueError("could not compile cfitsio %s" % cfitsio_version)
    os.chdir(package_basedir)


if not os.path.exists('build'):
    ret=os.makedirs('build')

if not os.path.exists(cfitsio_build_dir):
    ret=os.makedirs(cfitsio_build_dir)

copy_update(cfitsio_dir, cfitsio_build_dir)

if not os.path.exists(makefile):
    configure_cfitsio()

compile_cfitsio()

data_files=[]


# when using "extra_objects" in Extension, changes in the objects do *not*
# cause a re-link!  The only way I know is to force a recompile by removing the
# directory
build_libdir=glob.glob(os.path.join('build','lib*'))
shutil.rmtree(build_libdir[0])

sources = ["fitsio/fitsio_pywrap.c"]

extra_objects = glob.glob(os.path.join(cfitsio_build_dir,'*.o'))
ext=Extension("fitsio._fitsio_wrap", 
              sources,
              extra_objects=extra_objects)

include_dirs=[cfitsio_dir,numpy.get_include()]
setup(name="fitsio", 
      packages=['fitsio'],
      version="1.0",
      data_files=data_files,
      ext_modules=[ext],
      include_dirs=include_dirs)




