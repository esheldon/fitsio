Read and write data to FITS files using the cfitsio library.

Description
-----------

This is a python extension written in c and python.  

A patched version of cfitsio 3.28 is bundled with this package.  This is
because most deployed versions of cfitsio in the wild don't have support for
interesting features like tiled image compression.  Also sometimes very new
versions of cfitsio have bugs, e.g. the unpatched 3.28 (the latest as of
september 2011) cannot read float and double from, or write signed or unsigned
byte images to tile compressed images.  Bundling a version that meets our needs
is a safe alternative.


Features
--------

- Read and write numpy arrays to and from image and binary table
  extensions.  
- Read and write keywords.
- Read and write images in tile-compressed format (RICE,GZIP,PLIO,HCOMPRESS).  
- Read/write gzip files. Read unix compress files (.Z,.zip)
- Read arbitrary subsets of table columns and rows without loading the
  whole file.
- TDIM information is used to return array columns in the correct shape
- Correctly writes and reads string table columns, including array columns
  of arbitrary shape.
- Supports unsigned types the way the FITS standard allows, by converting
  to signed and using zero offsets.  Note the FITS standard does not support
  unsigned 64-bit at all.  Similarly, signed byte are converted to unsigned.
- Correctly writes 1 byte integers table columns.
- read rows using slice notation
- data are guaranteed to conform to the FITS standard.

Known CFITSIO Bugs
------------------
These are bugs in the underlying cfitsio library (version 3.28)
- fits_get_compression_type always returns zero.  fitsio uses ZCMPTYPE header
  key instead, which seems to work for this version.

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

    # open the file and write a binary table. By default a new extension is
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

    # if there are multiple HDUs with the same name, and an EXTVER
    # is set, you can use it.  Here extver=2
    #    fits['mytable',2]


    # read the image from extension zero
    >>> img = fits[0].read()

    # read all rows and columns from the binary table extension
    >>> data = fits[1].read()
    >>> data = fits['mytable'].read()

    # read a subset of rows and columns. By default uses a case-insensitive
    # match but returned array leaves the names with original case
    >>> data = fits[1].read(rows=[1,5], columns=['index','x','y'])

    # read a subset of rows using slice notation (can also use read_slice)
    >>> data = fits[1][10:20]
    >>> data = fits[1][10:20:2]

    # read a single column as a simple array.  This is less
    # efficient when you plan to read multiple columns.
    >>> data = fits[1].read_column('x', rows=[1,5])
    
    # get rows that satisfy the input expression.  See "Row Filtering
    # Specification" in the cfitsio manual
    >>> w=fits[1].where('x > 0.25 && y < 35.0')
    >>> data = fits[1][w]

    # read the header
    >>> h = fits[0].read_header()
    >>> h['BITPIX']
    -64

    >>> fits.close()


    # now write some data
    >>> fits = FITS('test.fits','rw')

    # create an image
    >>> img=numpy.arange(20,30,dtype='i4')

    # write the data to the primary HDU
    >>> fits.write_image(img)

    # write the image with rice compression
    >>> fits.write_image(img, compress='rice')
 
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

Requirements
------------

    - you need a c compiler and build tools like Make
    - You need a recent python, probably >= 2.5, but this has not been
      extensively tested.
    - You need numerical python (numpy).

test
----
The unit tests should all pass for full support.

    import fitsio
    fitsio.test.test()

TODO
----

- Read subsets of *images*
- append rows to tables
- add tests for slice notation and the where function.
- We have row slices. Want to also implement this notation, e.g. 
  for extension 1
        data=fits[1]['colname'][10:30]
        rows=[3,8,11]
        data=fits[1]['colname'][rows]
        data=fits[1]['colname'].read()
        cols=['x','y']
        data=fits[1][cols][10:30]
  That would require a FITSColumnSubset class.
- don't need to update the hdu list quite so often.
- keyword lists are getting long; implement **keys everywhere?  It would
  lengthen the functions because they must extract keywords, but would
  simplify wrapper code.
- More error checking in c code for python lists and dicts.
- optimize writing tables. When there are no unsigned short or long, no signed
  bytes, no strings, this could be simple using fits_write_tblbytes.  If
  strings are present, it is hard to imagine how to do it: perhaps write
  the whole thing and then re-write the string columns?  For unsigned
  stuff we could add the scaling ourselves, but then it is far from
  atomic.
- complex table columns.  bit? logical?
- explore separate classes for image and table HDUs?  Inherit from base class.
- add lower,upper keywords to read routines.
- variable length columns 
- row filtering?
- HDU groups?

Note on array ordering
----------------------
        
Since numpy uses C order, FITS uses fortran order, we have to write the TDIM
and image dimensions in reverse order, but write the data as is.  Then we need
to also reverse the dims as read from the header when creating the numpy dtype,
but read as is.



