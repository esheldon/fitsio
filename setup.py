import distutils
from distutils.core import setup, Extension, Command
import os
import numpy
import glob
import shutil
import platform

package_basedir = os.path.abspath(os.curdir)

cfitsio_version = '3280patch'
cfitsio_dir = 'cfitsio%s' % cfitsio_version
cfitsio_build_dir = os.path.join('build',cfitsio_dir)

makefile = os.path.join(cfitsio_build_dir, 'Makefile')

def copy_update(dir1,dir2):
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
    ret=os.system('sh ./configure')
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
if len(build_libdir) > 0:
    shutil.rmtree(build_libdir[0])

sources = ["fitsio/fitsio_pywrap.c","fitsio/fitsio_pywrap_lists.c"]

extra_objects = glob.glob(os.path.join(cfitsio_build_dir,'*.o'))
if platform.system()=='Darwin':
    extra_compile_args=['-arch','i386','-arch','x86_64']
    extra_link_args=['-arch','i386','-arch','x86_64']
else:
    extra_compile_args=[]
    extra_link_args=[]
ext=Extension("fitsio._fitsio_wrap", 
              sources,
              extra_objects=extra_objects,
              extra_compile_args=extra_compile_args, 
              extra_link_args=extra_link_args)


description = ("A full featured python library to read from and "
               "write to FITS files.")

long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read()

classifiers = ["Development Status :: 5 - Production/Stable"
               ,"License :: OSI Approved :: GNU General Public License (GPL)"
               ,"Topic :: Scientific/Engineering :: Astronomy"
               ,"Intended Audience :: Science/Research"
              ]

include_dirs=[cfitsio_dir,numpy.get_include()]
setup(name="fitsio", 
      version="0.9.5",
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
      include_dirs=include_dirs)




