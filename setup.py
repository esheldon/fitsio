import distutils
from distutils.core import setup, Extension, Command
import os
import numpy
import glob

cfitsio_version = '3240'
cfitsio_dir = 'cfitsio%s' % cfitsio_version
makefile = os.path.join(cfitsio_dir, 'Makefile')

def configure_cfitsio():
    os.chdir(cfitsio_dir)
    ret=os.system('./configure')
    if ret != 0:
        raise ValueError("could not configure cfitsio %s" % cfitsio_version)
    os.chdir('..')

def compile_cfitsio():
    os.chdir(cfitsio_dir)
    ret=os.system('make')
    if ret != 0:
        raise ValueError("could not compile cfitsio %s" % cfitsio_version)
    os.chdir('..')

if not os.path.exists(makefile):
    configure_cfitsio()
compile_cfitsio()

data_files=[]


sources = ["fitsio/fitsio_pywrap.c"]

extra_objects = glob.glob(os.path.join(cfitsio_dir,'*.o'))
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




