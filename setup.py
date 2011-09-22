import distutils
from distutils.core import setup, Extension, Command
import os
import numpy
import glob

package_basedir = os.path.abspath(os.curdir)

cfitsio_version = '3280'
cfitsio_dir = 'cfitsio%s' % cfitsio_version
cfitsio_build_dir = os.path.join('build',cfitsio_dir)

makefile = os.path.join(cfitsio_build_dir, 'Makefile')

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

ret=os.system('cp -u %s/* %s/' % (cfitsio_dir, cfitsio_build_dir))
if ret != 0:
    raise ValueError("could not copy %s to %s" % (cfitsio_dir, cfitsio_build_dir))

if not os.path.exists(makefile):
    configure_cfitsio()

compile_cfitsio()

data_files=[]


# when using "extra_objects", changes in the objects do
# *not* cause a re-link!  The only way I know is to force
# a recompile by removing the directory
os.system('rm -r build/lib*')

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




