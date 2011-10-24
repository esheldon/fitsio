A python library to read and write data to FITS files using cfitsio.

Description
-----------

This is a python extension written in c and python.  Data are read into
numerical python arrays.

A patched version of cfitsio 3.28 is bundled with this package, there
is no need to install your own, nor will this conflict with a version
you have installed; see below for details.  


Some Features
-------------

- Read from and write to image, binary, and ascii table extensions.
- Read arbitrary subsets of table columns and rows without loading the
  whole file.
- Write and read variable length table columns.  Can read into fixed length
  arrays with the maximum size, or object arrays to save memory.
- Read columns and rows using slice notation similar to numpy arrays
  This is like a more powerful memmap, since it is column-aware.
- Append rows to an existing table.
- Query the columns and rows in a table.
- Read and write header keywords.
- Read and write images in tile-compressed format (RICE,GZIP,PLIO,HCOMPRESS).  
- Read/write gzip files directly.  Read unix compress files (.Z,.zip).
- TDIM information is used to return array columns in the correct shape.
- Write and read string table columns, including array columns of arbitrary
  shape.
- Read and write unsigned integer types and signed bytes.
- Write checksums into the header and verify them.
- data are guaranteed to conform to the FITS standard.


Examples
--------

    >>> import fitsio

    # Often you just want to quickly read or write data without bothering to
    # create a FITS object.  In that case, you can use the read and write
    # convienience functions.

    # read all data from the first hdu with data
    >>> data = fitsio.read(filename)
    # read a subset of rows and columns from the specified extension
    >>> data = fitsio.read(filename, rows=rows, columns=columns, ext=ext)
    # read the header, or both at once
    >>> h = fitsio.read_header(filename, extension)
    >>> data,h = fitsio.read(filename, ext=ext, header=True)

    # open the file, write a new binary table extension, and then write  the
    # data from "recarray" into the table. By default a new extension is
    # added to the file.  use clobber=True to overwrite an existing file
    # instead.  To append rows to an existing table, see below.
    >>> fitsio.write(filename, recarray)
    # write an image
    >>> fitsio.write(filename, image)

    #
    # the FITS class gives the you the ability to explore the data, and gives
    # more control
    #

    # open a FITS file for reading and explore
    >>> fits=fitsio.FITS('data.fits')

    # see what is in here; the FITS object prints itself
    >>> fits

    file: data.fits
    mode: READONLY
    extnum hdutype         hduname
    0      IMAGE_HDU
    1      BINARY_TBL      mytable

    # explore the extensions, either by extension number or
    # extension name if available
    >>> fits[0]

    file: data.fits
    extension: 0
    type: IMAGE_HDU
    image info:
      data type: f8
      dims: [4096,2048]

    >>> fits['mytable']  # can also use fits[1]

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

    # [-1] to refers the last HDU
    >>> fits[-1]
    ...

    # if there are multiple HDUs with the same name, and an EXTVER
    # is set, you can use it.  Here extver=2
    #    fits['mytable',2]


    # read the image from extension zero
    >>> img = fits[0].read()

    # read all rows and columns from the binary table extension
    >>> data = fits[1].read()
    >>> data = fits['mytable'].read()

    # read a subset of rows and columns. By default uses a case-insensitive
    # match. The result retains the names with original case.  If columns is a
    # sequence, a recarray is returned
    >>> data = fits[1].read(rows=[1,5], columns=['index','x','y'])

    # Similar but using slice notation
    # row subsets
    >>> data = fits[1][10:20]
    >>> data = fits[1][10:20:2]
    >>> data = fits[1][rowlist]

    # all rows of column 'x'
    >>> data = fits[1]['x'][:]

    # Read a few columns at once. This is more efficient than separate read for
    # each column
    >>> data = fits[1]['x','y'][:]

    # General column and row subsets.
    >>> data = fits[1][columns][rows]


    # Note dvarr shows type varray[10] and svar shows type vstring[8]. These
    # are variable length columns and the number specified is the maximum size.
    # By default they are read into fixed-length fields in the output array.
    # You can over-ride this by constructing the FITS object with the vstorage
    # keyword or specifying vstorage when reading.  Sending vstorage='object'
    # will store the data in variable size object fields to save memory; the
    # default is vstorage='fixed'.  Object fields can also be written out to a
    # new FITS file as variable length to save disk space.

    >>> fits = fitsio.FITS(filename,vstorage='object')
    # OR
    >>> data = fits[1].read(vstorage='object')
    >>> print data['dvarr'].dtype
        dtype('object')


    # you can grab a FITSHDU object to simplify notation
    >>> hdu1 = fits[1]
    >>> data = hdu1['x','y'][35:50]
    
    # get rows that satisfy the input expression.  See "Row Filtering
    # Specification" in the cfitsio manual
    >>> w=fits[1].where("x > 0.25 && y < 35.0")
    >>> data = fits[1][w]

    # read the header
    >>> h = fits[0].read_header()
    >>> h['BITPIX']
    -64

    >>> fits.close()


    # now write some data
    >>> fits = FITS('test.fits','rw')

 
    # create a rec array.  Note vstr
    # is a variable length string
    >>> nrows=35
    >>> data = numpy.zeros(nrows, dtype=[('index','i4'),('vstr','O'),('x','f8'),('arr','f4',(3,4))])
    >>> data['index'] = numpy.arange(nrows,dtype='i4')
    >>> data['x'] = numpy.random.random(nrows)
    >>> data['vstr'] = [str(i) for i in xrange(nrows)]
    >>> data['arr'] = numpy.arange(nrows*3*4,dtype='f4').reshape(nrows,3,4)

    # create a new table extension and write the data
    >>> fits.write(data)

    # note under the hood the above does the following
    >>> fits.create_table_hdu(data=data)
    >>> fits[-1].write(data)
    >>> fits.update_hdu_list()

    # append more rows to the table.  The fields in data2 should match columns
    # in the table.  missing columns will be filled with zeros
    >>> fits[-1].append(data2)


    # create an image
    >>> img=numpy.arange(20,30,dtype='i4')

    # write an image in a new HDU (if this is a new file, the primary HDU)
    >>> fits.write(img)

    # write an image with rice compression
    >>> fits.write(img, compress='rice')


    # add checksums for the data
    >>> fits[-1].write_checksum()

    # you can also write a header at the same time.  The header can be a simple
    # dict (no comments), or a list of dicts with 'name','value','comment'
    # fields, or a FITSHDR object

    >>> header = {'somekey': 35, 'location': 'kitt peak'}
    >>> fits.write_table(data, header=header)
   
    # you can add individual keys to an existing HDU
    >>> fits[1].write_key(name, value, comment="my comment")

    >>> fits.close()

    # using a context, the file is closed automatically after leaving the block
    with FITS('path/to/file') as fits:
        data = fits[ext][:]

Installation
------------
Either download the tar ball ("Downloads" in the center of the github page) or
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

- separate classes for image, ascii and binary table HDUs.  Inherit from base
  class.
- Test variable length columns in ascii tables.
- Read subsets of *images*
- More error checking in c code for python lists and dicts.
- optimize writing tables. When there are no unsigned short or long, no signed
  bytes, no strings, this could be simple using fits_write_tblbytes.  If
  strings are present, it is hard to imagine how to do it: perhaps write
  the whole thing and then re-write the string columns?  For unsigned
  stuff we could add the scaling ourselves, but then it is far from
  atomic.
- complex table columns.  bit? logical?
- add lower,upper keywords to read routines.
- HDU groups?
- Clean up the code

Notes on cfitsio bundling
-------------------------

We bundle partly because many deployed versions of cfitsio in the wild don't
have support for interesting features like tiled image compression.   Bundling
a version that meets our needs is a safe alternative.  The patches to 3.28 fix
the ability to read float and double images from tile-compressed HDUs and add
back support for tile-compressed byte and unsigned byte images.

There are no known bugs in the patched version 3280 included in this package.

Note on array ordering
----------------------
        
Since numpy uses C order, FITS uses fortran order, we have to write the TDIM
and image dimensions in reverse order, but write the data as is.  Then we need
to also reverse the dims as read from the header when creating the numpy dtype,
but read as is.



