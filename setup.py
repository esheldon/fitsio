import distutils
from distutils.core import setup, Extension, Command
import os
import numpy

data_files=[]

ext=Extension("fitsio._fitsio_wrap", 
              ["fitsio/fitsio_pywrap.c"],
              libraries=['cfitsio'])
setup(name="fitsio", 
      packages=['fitsio'],
      version="1.0",
      data_files=data_files,
      ext_modules=[ext],
      include_dirs=numpy.get_include())




