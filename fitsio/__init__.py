"""
Read and write data to FITS files using the cfitsio library.

This is a python extension written in c and python.  The cfitsio library and
headers are required to compile the code.

Features
--------

- Read and write numpy arrays to and from image and binary table
  extensions.  
- Read and write keywords.
- Read arbitrary subsets of table columns and rows without loading the
  whole file.
- TDIM information is used to return array columns in the correct shape
- Correctly writes and reads string table columns, including array columns
  of arbitrary shape.
- Supports unsigned types the way the FITS standard allows, by converting
  to signed and using zero offsets.  Note the FITS standard does not support
  unsigned 64-bit at all.  Similarly, signed byte are converted to unsigned.
  Be careful of this feature!
- Correctly writes 1 byte integers table columns.
- data are guaranteed to conform to the FITS standard.



Examples
--------

    >>> import fitsio

    # Often you just want to quickly read or write data without bothering to
    # create a FITS object.  In that case, you can use the read and write
    # convienience functions.

    # read all data from the specified extension
    >>> data = fitsio.read(filename, extension)
    >>> h = fitsio.read_header(filename, extension)
    >>> data,h = fitsio.read_header(filename, extension, header=True)

    # open the file and write a new binary table by default a new extension is
    # appended to the file.  use clobber=True to overwrite an existing file
    # instead
    >>> fitsio.write(filename, recarray)


    # the FITS class gives the you the ability to explore the data, and gives
    # more control

    # open a FITS file and explore
    >>> fits=fitsio.FITS('data.fits','r')

    # see what is in here
    >>> fits

    file: data.fits
    mode: READONLY
    extnum hdutype         hduname
    0      IMAGE_HDU
    1      BINARY_TBL      mytable

    # explore the extensions, either by extension number or
    # extension name if available
    >>> fits[0]

    extension: 0
    type: IMAGE_HDU
    image info:
      data type: f8
      dims: [4096,2048]

    >>> fits['mytable']  # can also use fits[1]

    extension: 1
    type: BINARY_TBL
    extname: mytable
    column info:
      i1scalar            u1
      f                   f4
      fvec                f4  array[2]
      darr                f8  array[3,2]
      s                   S5
      svec                S6  array[3]
      sarr                S2  array[4,3]

    # read the image from extension zero
    >>> img = fits[0].read()

    # read all rows and columns from the binary table extension
    >>> data = fits[1].read()
    >>> data = fits['mytable'].read()

    # read a subset of rows and columns
    >>> data = fits[1].read(rows=[1,5], columns=['index','x','y'])

    # read a single column as a simple array.  This is less
    # efficient when you plan to read multiple columns.
    >>> data = fits[1].read_column('x', rows=[1,5])
    
    # read the header
    >>> h = fits[0].read_header()
    >>> h['BITPIX']
    -64

    >>> fits.close()


    # now write some data
    >>> fits = FITS('test.fits','rw')

    # create an image
    >>> img=numpy.arange(20,30)

    # write the data to the primary HDU
    >>> fits.write_image(img)
 
    # create a rec array
    >>> nrows=35
    >>> data = numpy.zeros(nrows, dtype=[('index','i4'),('x','f8'),('arr','f4',(3,4))])
    >>> data['index'] = numpy.arange(nrows,dtype='i4')
    >>> data['x'] = numpy.random.random(nrows)
    >>> data['arr'] = numpy.arange(nrows*3*4,dtype='f4').reshape(nrows,3,4)

    # create a new table extension and write the data
    >>> fits.write_table(data)

    # you can also write a header at the same time.  The header
    # can be a simple dict, or a list of dicts with 'name','value','comment'
    # fields, or a FITSHDR object

    >>> header = {'somekey': 35, 'location': 'kitt peak'}
    >>> fits.write_table(data, header=header)
   
    # you can add individual keys to an existing HDU
    >>> fits[1].write_key(name, value, comment="my comment")

    >>> fits.close()

    # using a context, the file is closed automatically
    # after leaving the block
    with FITS('path/to/file','r') as fits:
        data = fits[ext].read()


Installation
------------
Either download the tar ball (upper right corner "Downloads" on github page) or
use 

    git clone git://github.com/esheldon/fitsio.git

Enter the fitsio directory and type

    python setup.py install

optionally with a prefix 

    python setup.py install --prefix=/some/path

You will need the cfitsio library and headers installed on your system and
visible.

TODO
----

- Full test sweet writing and reading of all types both in read rec mode and
  read single column mode.  Also with subsets of rows.
- Figure out how to make cfitsio flush write buffers.  Only needed
  because my optimal read hack bypasses the buffers.
- add lower,upper keywords to read routines.
- append rows to tables
- read row *ranges* more optimally
- write images with rice compression (note .gz files automagically supported)
- More error checking in c code for python lists and dicts.
- write TDIM using built in routine instead of rolling my own.
- optimize writing tables when there are no unsigned short or long, no
  signed bytes.  Can do one big "fwrite" but need to be careful with
  confusing buffers.
- complex table columns.  bit? logical?
- explore separate classes for image and table HDUs?
- variable length columns


Note on array ordering
----------------------
        
Since numpy uses C order, FITS uses fortran order, we have to write the TDIM
and image dimensions in reverse order, but write the data as is.  Then we need
to also reverse the dims as read from the header when creating the numpy dtype,
but read as is.

"""
from . import fitslib
from .fitslib import FITS
from .fitslib import FITSHDU
from .fitslib import FITSHDR
from .fitslib import read
from .fitslib import read_header
from .fitslib import write
from .fitslib import READONLY
from .fitslib import READWRITE
