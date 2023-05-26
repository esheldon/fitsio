A python library to read from and write to FITS files.

[![Build Status (master)](https://travis-ci.com/esheldon/fitsio.svg?branch=master)](https://travis-ci.com/esheldon/fitsio)
[![tests](https://github.com/esheldon/fitsio/workflows/tests/badge.svg)](https://github.com/esheldon/fitsio/actions?query=workflow%3Atests)

## Description

This is a python extension written in c and python.  Data are read into
numerical python arrays.

A version of cfitsio is bundled with this package, there is no need to install
your own, nor will this conflict with a version you have installed.


## Some Features

- Read from and write to image, binary, and ascii table extensions.
- Read arbitrary subsets of table columns and rows without loading all the data
  to memory.
- Read image subsets without reading the whole image.  Write subsets to existing images.
- Write and read variable length table columns.
- Read images and tables using slice notation similar to numpy arrays.  This is like a more
  powerful memmap, since it is column-aware for tables.
- Append rows to an existing table.  Delete row sets and row ranges. Resize tables,
    or insert rows.
- Query the columns and rows in a table.
- Read and write header keywords.
- Read and write images in tile-compressed format (RICE,GZIP,PLIO,HCOMPRESS).
- Read/write gzip files directly.  Read unix compress (.Z,.zip) and bzip2 (.bz2) files.
- TDIM information is used to return array columns in the correct shape.
- Write and read string table columns, including array columns of arbitrary
  shape.
- Read and write complex, bool (logical), unsigned integer, signed bytes types.
- Write checksums into the header and verify them.
- Insert new columns into tables in-place.
- Iterate over rows in a table.  Data are buffered for efficiency.
- python 3 support, including python 3 strings


## Examples

```python
import fitsio
from fitsio import FITS,FITSHDR

# Often you just want to quickly read or write data without bothering to
# create a FITS object.  In that case, you can use the read and write
# convienience functions.

# read all data from the first hdu that has data
filename='data.fits'
data = fitsio.read(filename)

# read a subset of rows and columns from a table
data = fitsio.read(filename, rows=[35,1001], columns=['x','y'], ext=2)

# read the header
h = fitsio.read_header(filename)
# read both data and header
data,h = fitsio.read(filename, header=True)

# open the file and write a new binary table extension with the data
# array, which is a numpy array with fields, or "recarray".

data = np.zeros(10, dtype=[('id','i8'),('ra','f8'),('dec','f8')])
fitsio.write(filename, data)

# Write an image to the same file. By default a new extension is
# added to the file.  use clobber=True to overwrite an existing file
# instead.  To append rows to an existing table, see below.

fitsio.write(filename, image)

#
# the FITS class gives the you the ability to explore the data, and gives
# more control
#

# open a FITS file for reading and explore
fits=fitsio.FITS('data.fits')

# see what is in here; the FITS object prints itself
print(fits)

file: data.fits
mode: READONLY
extnum hdutype         hduname
0      IMAGE_HDU
1      BINARY_TBL      mytable

# at the python or ipython prompt the fits object will
# print itself
>>> fits
file: data.fits
... etc

# explore the extensions, either by extension number or
# extension name if available
>>> fits[0]

file: data.fits
extension: 0
type: IMAGE_HDU
image info:
  data type: f8
  dims: [4096,2048]

# by name; can also use fits[1]
>>> fits['mytable']

file: data.fits
extension: 1
type: BINARY_TBL
extname: mytable
rows: 4328342
column info:
  i1scalar            u1
  f                   f4
  fvec                f4  array[2]
  darr                f8  array[3,2]
  dvarr               f8  varray[10]
  s                   S5
  svec                S6  array[3]
  svar                S0  vstring[8]
  sarr                S2  array[4,3]

# See bottom for how to get more information for an extension

# [-1] to refers the last HDU
>>> fits[-1]
...

# if there are multiple HDUs with the same name, and an EXTVER
# is set, you can use it.  Here extver=2
#    fits['mytable',2]


# read the image from extension zero
img = fits[0].read()
img = fits[0][:,:]

# read a subset of the image without reading the whole image
img = fits[0][25:35, 45:55]


# read all rows and columns from a binary table extension
data = fits[1].read()
data = fits['mytable'].read()
data = fits[1][:]

# read a subset of rows and columns. By default uses a case-insensitive
# match. The result retains the names with original case.  If columns is a
# sequence, a numpy array with fields, or recarray is returned
data = fits[1].read(rows=[1,5], columns=['index','x','y'])

# Similar but using slice notation
# row subsets
data = fits[1][10:20]
data = fits[1][10:20:2]
data = fits[1][[1,5,18]]

# Using EXTNAME and EXTVER values
data = fits['SCI',2][10:20]

# Slicing with reverse (flipped) striding
data = fits[1][40:25]
data = fits[1][40:25:-5]

# all rows of column 'x'
data = fits[1]['x'][:]

# Read a few columns at once. This is more efficient than separate read for
# each column
data = fits[1]['x','y'][:]

# General column and row subsets.
columns=['index','x','y']
rows = [1, 5]
data = fits[1][columns][rows]

# data are returned in the order requested by the user
# and duplicates are preserved
rows = [2, 2, 5]
data = fits[1][columns][rows]

# iterate over rows in a table hdu
# faster if we buffer some rows, let's buffer 1000 at a time
fits=fitsio.FITS(filename,iter_row_buffer=1000)
for row in fits[1]:
    print(row)

# iterate over HDUs in a FITS object
for hdu in fits:
    data=hdu.read()

# Note dvarr shows type varray[10] and svar shows type vstring[8]. These
# are variable length columns and the number specified is the maximum size.
# By default they are read into fixed-length fields in the output array.
# You can over-ride this by constructing the FITS object with the vstorage
# keyword or specifying vstorage when reading.  Sending vstorage='object'
# will store the data in variable size object fields to save memory; the
# default is vstorage='fixed'.  Object fields can also be written out to a
# new FITS file as variable length to save disk space.

fits = fitsio.FITS(filename,vstorage='object')
# OR
data = fits[1].read(vstorage='object')
print(data['dvarr'].dtype)
    dtype('object')


# you can grab a FITS HDU object to simplify notation
hdu1 = fits[1]
data = hdu1['x','y'][35:50]

# get rows that satisfy the input expression.  See "Row Filtering
# Specification" in the cfitsio manual (note no temporary table is
# created in this case, contrary to the cfitsio docs)
w=fits[1].where("x > 0.25 && y < 35.0")
data = fits[1][w]

# read the header
h = fits[0].read_header()
print(h['BITPIX'])
    -64

fits.close()


# now write some data
fits = FITS('test.fits','rw')


# create a rec array.  Note vstr
# is a variable length string
nrows=35
data = np.zeros(nrows, dtype=[('index','i4'),('vstr','O'),('x','f8'),
                              ('arr','f4',(3,4))])
data['index'] = np.arange(nrows,dtype='i4')
data['x'] = np.random.random(nrows)
data['vstr'] = [str(i) for i in xrange(nrows)]
data['arr'] = np.arange(nrows*3*4,dtype='f4').reshape(nrows,3,4)

# create a new table extension and write the data
fits.write(data)

# can also be a list of ordinary arrays if you send the names
array_list=[xarray,yarray,namearray]
names=['x','y','name']
fits.write(array_list, names=names)

# similarly a dict of arrays
fits.write(dict_of_arrays)
fits.write(dict_of_arrays, names=names) # control name order

# append more rows to the table.  The fields in data2 should match columns
# in the table.  missing columns will be filled with zeros
fits[-1].append(data2)

# insert a new column into a table
fits[-1].insert_column('newcol', data)

# insert with a specific colnum
fits[-1].insert_column('newcol', data, colnum=2)

# overwrite rows
fits[-1].write(data)

# overwrite starting at a particular row. The table will grow if needed
fits[-1].write(data, firstrow=350)


# create an image
img=np.arange(2*3,dtype='i4').reshape(2,3)

# write an image in a new HDU (if this is a new file, the primary HDU)
fits.write(img)

# write an image with rice compression
fits.write(img, compress='rice')

# control the compression
fimg=np.random.normal(size=2*3).reshape(2, 3)
fits.write(img, compress='rice', qlevel=16, qmethod='SUBTRACTIVE_DITHER_2')

# lossless gzip compression for integers or floating point
fits.write(img, compress='gzip', qlevel=None)
fits.write(fimg, compress='gzip', qlevel=None)

# overwrite the image
fits[ext].write(img2)

# write into an existing image, starting at the location [300,400]
# the image will be expanded if needed
fits[ext].write(img3, start=[300,400])

# change the shape of the image on disk
fits[ext].reshape([250,100])

# add checksums for the data
fits[-1].write_checksum()

# can later verify data integridy
fits[-1].verify_checksum()

# you can also write a header at the same time.  The header can be
#   - a simple dict (no comments)
#   - a list of dicts with 'name','value','comment' fields
#   - a FITSHDR object

hdict = {'somekey': 35, 'location': 'kitt peak'}
fits.write(data, header=hdict)
hlist = [{'name':'observer', 'value':'ES', 'comment':'who'},
         {'name':'location','value':'CTIO'},
         {'name':'photometric','value':True}]
fits.write(data, header=hlist)
hdr=FITSHDR(hlist)
fits.write(data, header=hdr)

# you can add individual keys to an existing HDU
fits[1].write_key(name, value, comment="my comment")

# Write multiple header keys to an existing HDU. Here records
# is the same as sent with header= above
fits[1].write_keys(records)

# write special COMMENT fields
fits[1].write_comment("observer JS")
fits[1].write_comment("we had good weather")

# write special history fields
fits[1].write_history("processed with software X")
fits[1].write_history("re-processed with software Y")

fits.close()

# using a context, the file is closed automatically after leaving the block
with FITS('path/to/file') as fits:
    data = fits[ext].read()

    # you can check if a header exists using "in":
    if 'blah' in fits:
        data=fits['blah'].read()
    if 2 in f:
        data=fits[2].read()

# methods to get more information about extension.  For extension 1:
f[1].get_info()             # lots of info about the extension
f[1].has_data()             # returns True if data is present in extension
f[1].get_extname()
f[1].get_extver()
f[1].get_extnum()           # return zero-offset extension number
f[1].get_exttype()          # 'BINARY_TBL' or 'ASCII_TBL' or 'IMAGE_HDU'
f[1].get_offsets()          # byte offsets (header_start, data_start, data_end)
f[1].is_compressed()        # for images. True if tile-compressed
f[1].get_colnames()         # for tables
f[1].get_colname(colnum)    # for tables find the name from column number
f[1].get_nrows()            # for tables
f[1].get_rec_dtype()        # for tables
f[1].get_rec_column_descr() # for tables
f[1].get_vstorage()         # for tables, storage mechanism for variable
                            # length columns

# public attributes you can feel free to change as needed
f[1].lower           # If True, lower case colnames on output
f[1].upper           # If True, upper case colnames on output
f[1].case_sensitive  # if True, names are matched case sensitive
```


## Installation

The easiest way is using pip or conda. To get the latest release

    pip install fitsio

    # update fitsio (and everything else)
    pip install fitsio --upgrade

    # if pip refuses to update to a newer version
    pip install fitsio --upgrade --ignore-installed

    # if you only want to upgrade fitsio
    pip install fitsio --no-deps --upgrade --ignore-installed

    # for conda, use conda-forge
    conda install -c conda-forge fitsio

You can also get the latest source tarball release from

    https://pypi.python.org/pypi/fitsio

or the bleeding edge source from github or use git. To check out
the code for the first time

    git clone https://github.com/esheldon/fitsio.git

Or at a later time to update to the latest

    cd fitsio
    git update

Use tar xvfz to untar the file, enter the fitsio directory and type

    python setup.py install

optionally with a prefix

    python setup.py install --prefix=/some/path

## Requirements

- python 2 or python 3
- a C compiler and build tools like `make`, `patch`, etc.
- numpy (See the note below. Generally, numpy 1.11 or later is better.)


### Do not use numpy 1.10.0 or 1.10.1

There is a serious performance regression in numpy 1.10 that results
in fitsio running tens to hundreds of times slower.  A fix may be
forthcoming in a later release.  Please comment here if this
has already impacted your work https://github.com/numpy/numpy/issues/6467


## Tests

The unit tests should all pass for full support.

```bash
pytest fitsio
```

Some tests may fail if certain libraries are not available, such
as bzip2.  This failure only implies that bzipped files cannot
be read, without affecting other functionality.

## Notes on Usage and Features

### cfitsio bundling

We bundle cfitsio partly because many deployed versions of cfitsio in the
wild do not have support for interesting features like tiled image compression.
Bundling a version that meets our needs is a safe alternative.

### array ordering

Since numpy uses C order, FITS uses fortran order, we have to write the TDIM
and image dimensions in reverse order, but write the data as is.  Then we need
to also reverse the dims as read from the header when creating the numpy dtype,
but read as is.

### `distutils` vs `setuptools`

As of version `1.0.0`, `fitsio` has been transitioned to `setuptools` for packaging
and installation. There are many reasons to do this (and to not do this). However,
at a practical level, what this means for you is that you may have trouble uninstalling
older versions with `pip` via `pip uninstall fitsio`. If you do, the best thing to do is
to manually remove the files manually. See this [stackoverflow question](https://stackoverflow.com/questions/402359/how-do-you-uninstall-a-python-package-that-was-installed-using-distutils)
for example.

### python 3 strings

As of version `1.0.0`, fitsio now supports Python 3 strings natively. This support
means that for Python 3, native strings are read from and written correctly to
FITS files. All byte string columns are treated as ASCII-encoded unicode strings
as well. For FITS files written with a previous version of fitsio, the data
in Python 3 will now come back as a string and not a byte string. Note that this
support is not the same as full unicode support. Internally, fitsio only supports
the ASCII character set.

## TODO

- HDU groups: does anyone use these? If so open an issue!
