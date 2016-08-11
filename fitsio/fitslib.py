"""
fitslib, part of the fitsio package.

See the main docs at https://github.com/esheldon/fitsio

  Copyright (C) 2011  Erin Sheldon, BNL.  erin dot sheldon at gmail dot com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
from __future__ import with_statement, print_function
import sys, os
import numpy
import copy
import warnings

from . import _fitsio_wrap
from .util import FITSRuntimeWarning, cfitsio_version

# for python3 compat
try:
    xrange=xrange
except:
    xrange=range

from functools import reduce

def read(filename, ext=None, extver=None, **keys):
    """
    Convenience function to read data from the specified FITS HDU

    By default, all data are read.  For tables, send columns= and rows= to
    select subsets of the data.  Table data are read into a recarray; use a
    FITS object and read_column() to get a single column as an ordinary array.
    For images, create a FITS object and use slice notation to read subsets.

    Under the hood, a FITS object is constructed and data are read using
    an associated FITSHDU object.

    parameters
    ----------
    filename: string
        A filename. 
    ext: number or string, optional
        The extension.  Either the numerical extension from zero
        or a string extension name. If not sent, data is read from
        the first HDU that has data.
    extver: integer, optional
        FITS allows multiple extensions to have the same name (extname).  These
        extensions can optionally specify an EXTVER version number in the
        header.  Send extver= to select a particular version.  If extver is not
        sent, the first one will be selected.  If ext is an integer, the extver
        is ignored.
    columns: list or array, optional
        An optional set of columns to read from table HDUs.  Default is to
        read all.  Can be string or number.
    rows: optional
        An optional list of rows to read from table HDUS.  Default is to
        read all.
    header: bool, optional
        If True, read the FITS header and return a tuple (data,header)
        Default is False.
    case_sensitive: bool, optional
        Match column names and extension names with case-sensitivity.  Default
        is False.
    lower: bool, optional
        If True, force all columns names to lower case in output
    upper: bool, optional
        If True, force all columns names to upper case in output
    vstorage: string, optional
        Set the default method to store variable length columns.  Can be
        'fixed' or 'object'.  See docs on fitsio.FITS for details.
    """

    with FITS(filename, **keys) as fits:

        header=keys.pop('header',False)

        if ext is None:
            for i in xrange(len(fits)):
                if fits[i].has_data():
                    ext=i
                    break
            if ext is None:
                raise IOError("No extensions have data")

        item=_make_item(ext, extver=extver)

        data = fits[item].read(**keys)
        if header:
            h = fits[item].read_header()
            return data, h
        else:
            return data


def read_header(filename, ext=0, extver=None, case_sensitive=False, **keys):
    """
    Convenience function to read the header from the specified FITS HDU

    The FITSHDR allows access to the values and comments by name and
    number.

    Under the hood, a FITS object is constructed and data are read using
    an associated FITSHDU object.

    parameters
    ----------
    filename: string
        A filename. 
    ext: number or string, optional
        The extension.  Either the numerical extension from zero
        or a string extension name. Default read primary header.
    extver: integer, optional
        FITS allows multiple extensions to have the same name (extname).  These
        extensions can optionally specify an EXTVER version number in the
        header.  Send extver= to select a particular version.  If extver is not
        sent, the first one will be selected.  If ext is an integer, the extver
        is ignored.
    case_sensitive: bool, optional
        Match extension names with case-sensitivity.  Default is False.
    """
    item=_make_item(ext,extver=extver)
    with FITS(filename, case_sensitive=case_sensitive) as fits:
        return fits[item].read_header()

def read_scamp_head(fname, header=None):
    """
    read a SCAMP .head file as a fits header FITSHDR object

    parameters
    ----------
    fname: string
        The path to the SCAMP .head file

    header: FITSHDR, optional
        Optionally combine the header with the input one. The input can
        be any object convertable to a FITSHDR object

    returns
    -------
    header: FITSHDR
        A fits header object of type FITSHDR
    """

    with open(fname) as fobj:
        lines=fobj.readlines()

    lines=[l.strip() for l in lines if l[0:3] != 'END']

    # if header is None an empty FITSHDR is created
    hdr=FITSHDR(header)

    for l in lines:
        hdr.add_record(l)

    return hdr

def _make_item(ext, extver=None):
    if extver is not None:
        # e
        item=(ext,extver)
    else:
        item=ext

    return item

def write(filename, data, extname=None, extver=None, units=None, 
          compress=None, table_type='binary', header=None, 
          clobber=False, **keys):
    """
    Convenience function to create a new HDU and write the data.

    Under the hood, a FITS object is constructed.  If you want to append rows
    to an existing HDU, or modify data in an HDU, please construct a FITS
    object.

    parameters
    ----------
    filename: string
        A filename. 
    data:
        Either a normal n-dimensional array or a recarray.  Images are written
        to a new IMAGE_HDU and recarrays are written to BINARY_TBl or
        ASCII_TBL hdus.
    extname: string, optional
        An optional name for the new header unit.
    extver: integer, optional
        FITS allows multiple extensions to have the same name (extname).
        These extensions can optionally specify an EXTVER version number in
        the header.  Send extver= to set a particular version, which will
        be represented in the header with keyname EXTVER.  The extver must
        be an integer > 0.  If extver is not sent, the first one will be
        selected.  If ext is an integer, the extver is ignored.
    compress: string, optional
        A string representing the compression algorithm for images, default None.
        Can be one of
           'RICE'
           'GZIP'
           'PLIO' (no unsigned or negative integers)
           'HCOMPRESS'
        (case-insensitive) See the cfitsio manual for details.

    header: FITSHDR, list, dict, optional
        A set of header keys to write. The keys are written before the data
        is written to the table, preventing a resizing of the table area.
        
        Can be one of these:
            - FITSHDR object
            - list of dictionaries containing 'name','value' and optionally
              a 'comment' field.
            - a dictionary of keyword-value pairs; no comments are written
              in this case, and the order is arbitrary
        Note required keywords such as NAXIS, XTENSION, etc are cleaed out.

    clobber: bool, optional
        If True, overwrite any existing file. Default is to append
        a new extension on existing files.


    table keywords
    --------------
    These keywords are only active when writing tables.

    units: list
        A list of strings representing units for each column.
    table_type: string, optional
        Either 'binary' or 'ascii', default 'binary'
        Matching is case-insensitive


    """
    with FITS(filename, 'rw', clobber=clobber, **keys) as fits:
        fits.write(data,
                   table_type=table_type,
                   units=units,
                   extname=extname,
                   extver=extver, 
                   compress=compress,
                   header=header,
                   **keys)


ANY_HDU=-1

READONLY=0
READWRITE=1
IMAGE_HDU=0
ASCII_TBL=1
BINARY_TBL=2

NOCOMPRESS=0
RICE_1 = 11
GZIP_1 = 21
PLIO_1 = 31
HCOMPRESS_1 = 41

class FITS(object):
    """
    A class to read and write FITS images and tables.

    This class uses the cfitsio library for almost all relevant work.

    parameters
    ----------
    filename: string
        The filename to open.  
    mode: int/string, optional
        The mode, either a string or integer.
        For reading only
            'r' or 0
        For reading and writing
            'rw' or 1
        You can also use fitsio.READONLY and fitsio.READWRITE.

        Default is 'r'
    clobber: bool, optional
        If the mode is READWRITE, and clobber=True, then remove any existing
        file before opening.
    case_sensitive: bool, optional
        Match column names and extension names with case-sensitivity.  Default
        is False.
    lower: bool, optional
        If True, force all columns names to lower case in output
    upper: bool, optional
        If True, force all columns names to upper case in output
    vstorage: string, optional
        A string describing how, by default, to store variable length columns
        in the output array.  This can be over-ridden when reading by using the
        using vstorage keyword to the individual read methods.  The options are

            'fixed': Use a fixed length field in the array, with
                dimensions equal to the max possible size for column.
                Arrays are padded with zeros.
            'object': Use an object for the field in the array.
                Each element will then be an array of the right type,
                but only using the memory needed to hold that element.

        Default is 'fixed'.  The rationale is that this is the option
            of 'least surprise'
    iter_row_buffer: integer
        Number of rows to buffer when iterating over table HDUs.
        Default is 1.

    See the docs at https://github.com/esheldon/fitsio
    """
    def __init__(self, filename, mode='r', **keys):
        self.keys=keys
        filename = extract_filename(filename)
        self._filename = filename

        #self.mode=keys.get('mode','r')
        self.mode=mode
        self.case_sensitive=keys.get('case_sensitive',False)

        self.verbose = keys.get('verbose',False)
        clobber = keys.get('clobber',False)

        if self.mode not in _int_modemap:
            raise IOError("mode should be one of 'r','rw',"
                             "READONLY,READWRITE")

        self.charmode = _char_modemap[self.mode]
        self.intmode = _int_modemap[self.mode]

        create=0
        if self.mode in [READWRITE,'rw']:
            if clobber:
                create=1
                if os.path.exists(filename):
                    os.remove(filename)
            else:
                if os.path.exists(filename):
                    create=0
                else:
                    create=1
        else:
            if not os.path.exists(filename):
                raise IOError("File not found: '%s'" % filename)

        self._FITS =  _fitsio_wrap.FITS(filename, self.intmode, create)

    def close(self):
        """
        Close the fits file and set relevant metadata to None
        """
        if hasattr(self,'_FITS'):
            if self._FITS is not None:
                self._FITS.close()
                self._FITS=None
        self._filename=None
        self.mode=None
        self.charmode=None
        self.intmode=None
        self.hdu_list=None
        self.hdu_map=None

    def movabs_ext(self, ext):
        """
        Move to the indicated zero-offset extension.

        In general, it is not necessary to use this method explicitly.
        """
        return self._FITS.movabs_hdu(ext+1)

    def movabs_hdu(self, hdunum):
        """
        Move to the indicated one-offset hdu number.

        In general, it is not necessary to use this method explicitly.
        """
        return self._FITS.movabs_hdu(hdunum)

    def movnam_ext(self, extname, hdutype=ANY_HDU, extver=0):
        """
        Move to the indicated extension by name

        In general, it is not necessary to use this method explicitly.

        returns the zero-offset extension number
        """
        hdu = self._FITS.movnam_hdu(hdutype, extname, extver)
        return hdu-1

    def movnam_hdu(self, extname, hdutype=ANY_HDU, extver=0):
        """
        Move to the indicated HDU by name

        In general, it is not necessary to use this method explicitly.

        returns the one-offset extension number
        """
        hdu = self._FITS.movnam_hdu(hdutype, extname, extver)
        return hdu

    def reopen(self):
        """
        close and reopen the fits file with the same mode
        """
        self._FITS.close()
        del self._FITS
        self._FITS =  _fitsio_wrap.FITS(self._filename, self.intmode, 0)
        self.update_hdu_list()

    def write(self, data, units=None, extname=None, extver=None,
              compress=None, tile_dims=None,
              header=None,
              names=None,
              table_type='binary', **keys):
        """
        Write the data to a new HDU.

        This method is a wrapper.  If this is an IMAGE_HDU, write_image is
        called, otherwise write_table is called.

        parameters
        ----------
        data: ndarray
            An n-dimensional image or an array with fields.
        extname: string, optional
            An optional extension name.
        extver: integer, optional
            FITS allows multiple extensions to have the same name (extname).
            These extensions can optionally specify an EXTVER version number in
            the header.  Send extver= to set a particular version, which will
            be represented in the header with keyname EXTVER.  The extver must
            be an integer > 0.  If extver is not sent, the first one will be
            selected.  If ext is an integer, the extver is ignored.
        header: FITSHDR, list, dict, optional
            A set of header keys to write. Can be one of these:
                - FITSHDR object
                - list of dictionaries containing 'name','value' and optionally
                  a 'comment' field.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary
            Note required keywords such as NAXIS, XTENSION, etc are cleaed out.

        Image-only keywords:
            compress: string, optional
                A string representing the compression algorithm for images, default None.
                Can be one of
                    'RICE'
                    'GZIP'
                    'PLIO' (no unsigned or negative integers)
                    'HCOMPRESS'
                (case-insensitive) See the cfitsio manual for details.
        Table-only keywords:
            units: list/dec, optional:
                A list of strings with units for each column.
            table_type: string, optional
                Either 'binary' or 'ascii', default 'binary'
                Matching is case-insensitive

        restrictions
        ------------
        The File must be opened READWRITE
        """

        isimage=False
        if data is None:
            isimage=True
        elif isinstance(data,numpy.ndarray):
            if data.dtype.fields == None:
                isimage=True

        if isimage:
            self.write_image(data, extname=extname, extver=extver, 
                             compress=compress, tile_dims=tile_dims,
                             header=header)
        else:
            self.write_table(data, units=units, 
                             extname=extname, extver=extver, header=header,
                             names=names,
                             table_type=table_type)



    def write_image(self, img, extname=None, extver=None,
                    compress=None, tile_dims=None, header=None):
        """
        Create a new image extension and write the data.  

        parameters
        ----------
        img: ndarray
            An n-dimensional image.
        extname: string, optional
            An optional extension name.
        extver: integer, optional
            FITS allows multiple extensions to have the same name (extname).
            These extensions can optionally specify an EXTVER version number in
            the header.  Send extver= to set a particular version, which will
            be represented in the header with keyname EXTVER.  The extver must
            be an integer > 0.  If extver is not sent, the first one will be
            selected.  If ext is an integer, the extver is ignored.
        compress: string, optional
            A string representing the compression algorithm for images, default None.
            Can be one of
                'RICE'
                'GZIP'
                'PLIO' (no unsigned or negative integers)
                'HCOMPRESS'
            (case-insensitive) See the cfitsio manual for details.
        header: FITSHDR, list, dict, optional
            A set of header keys to write. Can be one of these:
                - FITSHDR object
                - list of dictionaries containing 'name','value' and optionally
                  a 'comment' field.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary
            Note required keywords such as NAXIS, XTENSION, etc are cleaed out.


        restrictions
        ------------
        The File must be opened READWRITE
        """

        self.create_image_hdu(img, 
                              header=header,
                              extname=extname, extver=extver,
                              compress=compress, tile_dims=tile_dims)

        if header is not None:
            self[-1].write_keys(header)
            self[-1]._update_info()

        #if img is not None:
        #    self[-1].write(img)


    def create_image_hdu(self,
                         img=None,
                         dims=None,
                         dtype=None,
                         extname=None,
                         extver=None,
                         compress=None,
                         tile_dims=None,
                         header=None):
        """
        Create a new, empty image HDU and reload the hdu list.  Either
        create from an input image or from input dims and dtype

            fits.create_image_hdu(image, ...)
            fits.create_image_hdu(dims=dims, dtype=dtype)

        If an image is sent, the data are also written.

        You can write data into the new extension using
            fits[extension].write(image)

        Alternatively you can skip calling this function and instead just use 

            fits.write(image)
            or
            fits.write_image(image)

        which will create the new image extension for you with the appropriate
        structure, and write the data.

        parameters
        ----------
        img: ndarray, optional
            An image with which to determine the properties of the HDU. The
            data will be written.
        dims: sequence, optional
            A sequence describing the dimensions of the image to be created
            on disk.  You must also send a dtype=
        dtype: numpy data type
            When sending dims= also send the data type.  Can be of the
            various numpy data type declaration styles, e.g. 'f8',
            numpy.float64.
        extname: string, optional
            An optional extension name.
        extver: integer, optional
            FITS allows multiple extensions to have the same name (extname).
            These extensions can optionally specify an EXTVER version number in
            the header.  Send extver= to set a particular version, which will
            be represented in the header with keyname EXTVER.  The extver must
            be an integer > 0.  If extver is not sent, the first one will be
            selected.  If ext is an integer, the extver is ignored.
        compress: string, optional
            A string representing the compression algorithm for images, default None.
            Can be one of
                'RICE'
                'GZIP'
                'PLIO' (no unsigned or negative integers)
                'HCOMPRESS'
            (case-insensitive) See the cfitsio manual for details.

        header: FITSHDR, list, dict, optional
            This is only used to determine how many slots to reserve for
            header keywords


        restrictions
        ------------
        The File must be opened READWRITE
        """

        if (img is not None) or (img is None and dims is None):
            from_image=True
        elif dims is not None:
            from_image=False

        if from_image:
            img2send=img
            if img is not None:
                dims=img.shape
                dtstr = img.dtype.descr[0][1][1:]
                if img.size == 0:
                    raise ValueError("data must have at least 1 row")

                # data must be c-contiguous and native byte order
                if not img.flags['C_CONTIGUOUS']:
                    # this always makes a copy
                    img2send = numpy.ascontiguousarray(img)
                    array_to_native(img2send, inplace=True)
                else:
                    img2send = array_to_native(img, inplace=False)

            else:
                self._ensure_empty_image_ok()
                compress=None
                tile_dims=None

            # we get dims from the input image
            dims2send=None
        else:
            # img was None and dims was sent
            if dtype is None:
                raise ValueError("send dtype= with dims=")

            # this must work!
            dtype=numpy.dtype(dtype)
            dtstr = dtype.descr[0][1][1:]
            # use the example image to build the type in C
            img2send=numpy.zeros(1, dtype=dtype)

            # sending an array simplifies access
            dims2send = numpy.array(dims,dtype='i8',ndmin=1)

        if img2send is not None:
            if img2send.dtype.fields is not None:
                raise ValueError("got record data type, expected regular ndarray")

        if extname is not None and extver is not None:
            extver = check_extver(extver)
        if extver is None:
            # will be ignored
            extver = 0
        if extname is None:
            # will be ignored
            extname=""

        comptype = get_compress_type(compress)
        tile_dims = get_tile_dims(tile_dims, dims)

        if img2send is not None:
            check_comptype_img(comptype, dtstr)

        if header is not None:
            nkeys=len(header)
        else:
            nkeys=0

        self._FITS.create_image_hdu(img2send,
                                    nkeys,
                                    dims=dims2send,
                                    comptype=comptype, 
                                    tile_dims=tile_dims,
                                    extname=extname,
                                    extver=extver)


        # don't rebuild the whole list unless this is the first hdu
        # to be created
        self.update_hdu_list(rebuild=False)


    def _ensure_empty_image_ok(self):
        """
        Only allow empty HDU for first HDU and if there is no
        data there already
        """
        if len(self) > 1:
            raise RuntimeError("Cannot write None image at extension %d" % len(self))
        if 'ndims' in self[0]._info:
            raise RuntimeError("Can only write None images to extension zero, "
                               "which already exists")


    def write_table(self, data, table_type='binary', 
                    names=None, formats=None, units=None, 
                    extname=None, extver=None, header=None):
        """
        Create a new table extension and write the data.

        The table definition is taken from the fields in the input array.  If
        you want to append new rows to the table, access the HDU directly and
        use the write() function, e.g.

            fits[extension].append(data)

        parameters
        ----------
        data: recarray
            A numpy array with fields.  The table definition will be
            determined from this array.
        table_type: string, optional
            Either 'binary' or 'ascii', default 'binary'
            Matching is case-insensitive
        extname: string, optional
            An optional string for the extension name.
        extver: integer, optional
            FITS allows multiple extensions to have the same name (extname).
            These extensions can optionally specify an EXTVER version number in
            the header.  Send extver= to set a particular version, which will
            be represented in the header with keyname EXTVER.  The extver must
            be an integer > 0.  If extver is not sent, the first one will be
            selected.  If ext is an integer, the extver is ignored.
        units: list/dec, optional:
            A list of strings with units for each column.
        header: FITSHDR, list, dict, optional
            A set of header keys to write. The keys are written before the data
            is written to the table, preventing a resizing of the table area.

            Can be one of these:
                - FITSHDR object
                - list of dictionaries containing 'name','value' and optionally
                  a 'comment' field.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary
            Note required keywords such as NAXIS, XTENSION, etc are cleaed out.

        restrictions
        ------------
        The File must be opened READWRITE
        """

        """
        if data.dtype.fields == None:
            raise ValueError("data must have fields")
        if data.size == 0:
            raise ValueError("data must have at least 1 row")
        """

        self.create_table_hdu(data=data, 
                              header=header,
                              names=names,
                              units=units, 
                              extname=extname,
                              extver=extver,
                              table_type=table_type)

        if header is not None:
            self[-1].write_keys(header)
            self[-1]._update_info()

        self[-1].write(data,names=names)

    def create_table_hdu(self, data=None, dtype=None, 
                         header=None,
                         names=None, formats=None,
                         units=None, dims=None, extname=None, extver=None, 
                         table_type='binary'):
        """
        Create a new, empty table extension and reload the hdu list.

        There are three ways to do it:
            1) send a numpy dtype, from which the formats in the fits file will
               be determined.
            2) Send an array in data= keyword.  this is required if you have
                object fields for writing to variable length columns.
            3) send the names,formats and dims yourself

        You can then write data into the new extension using
            fits[extension].write(array)
        If you want to write to a single column
            fits[extension].write_column(array)
        But be careful as the other columns will be left zeroed.

        Often you will instead just use write_table to do this all
        atomically.

            fits.write_table(recarray)

        write_table will create the new table extension for you with the
        appropriate fields.

        parameters
        ----------
        dtype: numpy dtype or descriptor, optional
            If you have an array with fields, you can just send arr.dtype.  You
            can also use a list of tuples, e.g. [('x','f8'),('index','i4')] or
            a dictionary representation.
        data: a numpy array with fields, optional
              or a dictionary

            An array or dict from which to determine the table definition.  You
            must use this instead of sending a descriptor if you have object
            array fields, as this is the only way to determine the type and max
            size.

        names: list of strings, optional
            The list of field names
        formats: list of strings, optional
            The TFORM format strings for each field.
        dims: list of strings, optional
            An optional list of dimension strings for each field.  Should
            match the repeat count for the formats fields. Be careful of
            the order since FITS is more like fortran. See the descr2tabledef
            function.

        table_type: string, optional
            Either 'binary' or 'ascii', default 'binary'
            Matching is case-insensitive
        units: list of strings, optional
            An optional list of unit strings for each field.
        extname: string, optional
            An optional extension name.
        extver: integer, optional
            FITS allows multiple extensions to have the same name (extname).
            These extensions can optionally specify an EXTVER version number in
            the header.  Send extver= to set a particular version, which will
            be represented in the header with keyname EXTVER.  The extver must
            be an integer > 0.  If extver is not sent, the first one will be
            selected.  If ext is an integer, the extver is ignored.

        header: FITSHDR, list, dict, optional
            This is only used to determine how many slots to reserve for
            header keywords


        restrictions
        ------------
        The File must be opened READWRITE
        """

        table_type_int=_extract_table_type(table_type)

        if data is not None:
            if isinstance(data,numpy.ndarray):
                names, formats, dims = array2tabledef(data, table_type=table_type)
            elif isinstance(data, (list,dict)):
                names, formats, dims = collection2tabledef(data, names=names,
                                                           table_type=table_type)
            else:
                raise ValueError("data must be an ndarray with fields or a dict")
        elif dtype is not None:
            dtype=numpy.dtype(dtype)
            names, formats, dims = descr2tabledef(dtype.descr)
        else:
            if names is None or formats is None:
                raise ValueError("send either dtype=, data=, or names= and formats=")

            if not isinstance(names,list) or not isinstance(formats,list):
                raise ValueError("names and formats should be lists")
            if len(names) != len(formats):
                raise ValueError("names and formats must be same length")

            if dims is not None:
                if not isinstance(dims,list):
                    raise ValueError("dims should be a list")
                if len(dims) != len(names):
                    raise ValueError("names and dims must be same length")

        if units is not None:
            if not isinstance(units,list):
                raise ValueError("units should be a list")
            if len(units) != len(names):
                raise ValueError("names and units must be same length")
        if extname is not None:
            if not isinstance(extname,str):
                raise ValueError("extension name must be a string")

        if extname is not None and extver is not None:
            extver = check_extver(extver)
        if extver is None:
            # will be ignored
            extver = 0
        if extname is None:
            # will be ignored
            extname=""

        if header is not None:
            nkeys=len(header)
        else:
            nkeys=0

        # note we can create extname in the c code for tables, but not images
        self._FITS.create_table_hdu(table_type_int, nkeys,
                                    names, formats, tunit=units, tdim=dims, 
                                    extname=extname, extver=extver)

        # don't rebuild the whole list unless this is the first hdu
        # to be created
        self.update_hdu_list(rebuild=False)

    def update_hdu_list(self, rebuild=True):
        """
        Force an update of the entire HDU list

        Normally you don't need to call this method directly

        if rebuild is false or the hdu_list is not yet set, the list is
        rebuilt from scratch
        """

        if not hasattr(self,'hdu_list'):
            rebuild=True

        if rebuild:
            self.hdu_list = []
            self.hdu_map={}

            # we don't know how many hdus there are, so iterate
            # until we can't open any more 
            ext_start=0
        else:
            # start from last
            ext_start=len(self)

        ext=ext_start
        while True:
            try:
                self._append_hdu_info(ext)
            except IOError:
                break
            except RuntimeError:
                break
            
            ext = ext + 1

    def _append_hdu_info(self, ext):
        """
        internal routine

        append info for indiciated extension
        """

        # raised IOError if not found
        hdu_type=self._FITS.movabs_hdu(ext+1)

        if hdu_type==IMAGE_HDU:
            hdu=ImageHDU(self._FITS, ext, **self.keys)
        elif hdu_type==BINARY_TBL:
            hdu=TableHDU(self._FITS, ext, **self.keys)
        elif hdu_type==ASCII_TBL:
            hdu=AsciiTableHDU(self._FITS, ext, **self.keys)
        else:
            mess=("extension %s is of unknown type %s "
                  "this is probably a bug")
            mess=mess % (ext,hdu_type)
            raise IOError(mess)

        self.hdu_list.append(hdu)
        self.hdu_map[ext] = hdu

        extname=hdu.get_extname()
        if not self.case_sensitive:
            extname=extname.lower()
        if extname != '':
            # this will guarantee we default to *first* version,
            # if version is not requested, using __getitem__
            if extname not in self.hdu_map:
                self.hdu_map[extname] = hdu
            
            ver=hdu.get_extver()
            if ver > 0:
                key='%s-%s' % (extname,ver)
                self.hdu_map[key] = hdu


    def __iter__(self):
        """
        begin iteration over HDUs
        """
        if not hasattr(self,'hdu_list'):
            self.update_hdu_list()
        self._iter_index=0
        return self

    def next(self):
        """
        Move to the next iteration
        """
        if self._iter_index == len(self.hdu_list):
            raise StopIteration
        hdu=self.hdu_list[self._iter_index]
        self._iter_index += 1
        return hdu
    __next__=next

    def __len__(self):
        """
        get the number of extensions
        """
        if not hasattr(self,'hdu_list'):
            self.update_hdu_list()
        return len(self.hdu_list)

    def _extract_item(self,item):
        """
        utility function to extract an "item", meaning
        a extension number,name plus version.
        """
        ver=0
        if isinstance(item,tuple):
            ver_sent=True
            nitem=len(item)
            if nitem == 1:
                ext=item[0]
            elif nitem == 2:
                ext,ver=item
        else:
            ver_sent=False
            ext=item
        return ext,ver,ver_sent

    def __getitem__(self, item):
        """
        Get an hdu by number, name, and possibly version
        """
        if not hasattr(self, 'hdu_list'):
            self.update_hdu_list()

        ext,ver,ver_sent = self._extract_item(item)

        try:
            # if it is an int
            hdu = self.hdu_list[ext]
        except:
            # might be a string
            ext='%s' % ext
            if not self.case_sensitive:
                mess='(case insensitive)'
                ext=ext.lower()
            else:
                mess='(case sensitive)'

            if ver > 0:
                key = '%s-%s' % (ext,ver)
                if key not in self.hdu_map:
                    raise IOError("extension not found: %s, "
                                  "version %s %s" % (ext,ver,mess))
                hdu = self.hdu_map[key]
            else:
                if ext not in self.hdu_map:
                    raise IOError("extension not found: %s %s" % (ext,mess))
                hdu = self.hdu_map[ext]

        return hdu

    def __contains__(self, item):
        """
        tell whether specified extension exists, possibly
        with version sent as well
        """
        try:
            hdu=self[item]
            return True
        except:
            return False
            
    def __repr__(self):
        """
        Text representation of some fits file metadata
        """
        spacing = ' '*2
        if not hasattr(self, 'hdu_list'):
            self.update_hdu_list()

        rep = ['']
        rep.append("%sfile: %s" % (spacing,self._filename))
        rep.append("%smode: %s" % (spacing,_modeprint_map[self.intmode]))

        rep.append('%sextnum %-15s %s' % (spacing,"hdutype","hduname[v]"))
        for i,hdu in enumerate(self.hdu_list):
            t = hdu._info['hdutype']
            name = hdu.get_extname()
            if name != '':
                ver=hdu.get_extver()
                if ver != 0:
                    name = '%s[%s]' % (name,ver) 

            rep.append("%s%-6d %-15s %s" % (spacing, i, _hdu_type_map[t], name))

        rep = '\n'.join(rep)
        return rep

    #def __del__(self):
    #    self.close()
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()



class HDUBase(object):
    """
    A representation of a FITS HDU

    construction parameters
    -----------------------
    fits: FITS object
        An instance of a _fistio_wrap.FITS object.  This is the low-level
        python object, not the FITS object defined above.
    ext: integer
        The extension number.
    case_sensitive: bool, optional
        Match column names and extension names with case-sensitivity.  Default
        is False.
    lower: bool, optional
        If True, force all columns names to lower case in output
    upper: bool, optional
        If True, force all columns names to upper case in output
    vstorage: string, optional
        Set the default method to store variable length columns.  Can be
        'fixed' or 'object'.  See docs on fitsio.FITS for details.
    iter_row_buffer: integer
        Number of rows to buffer when iterating over table HDUs.
        Default is 1.
    """
    def __init__(self, fits, ext, **keys):
        self._FITS = fits
        self._ext = ext

        self._update_info()
        self._filename = self._FITS.filename()

    def get_extnum(self):
        """
        Get the extension number
        """
        return self._ext

    def get_extname(self):
        """
        Get the name for this extension, can be an empty string
        """
        name = self._info['extname']
        if name.strip() == '':
            name = self._info['hduname']
        return name.strip()

    def get_extver(self):
        """
        Get the version for this extension.

        Used when a name is given to multiple extensions
        """
        ver=self._info['extver']
        if ver == 0:
            ver=self._info['hduver']
        return ver

    def get_exttype(self, num=False):
        """
        Get the extension type

        By default the result is a string that mirrors
        the enumerated type names in cfitsio
            'IMAGE_HDU', 'ASCII_TBL', 'BINARY_TBL'
        which have numeric values
            0 1 2
        send num=True to get the numbers.  The values
            fitsio.IMAGE_HDU .ASCII_TBL, and .BINARY_TBL
        are available for comparison

        parameters
        ----------
        num: bool, optional
            Return the numeric values.
        """
        if num:
            return self._info['hdutype']
        else:
            name=_hdu_type_map[self._info['hdutype']]
            return name

    def get_offsets(self):
        """
        returns
        -------
        a dictionary with these entries

        header_start:
            byte offset from beginning of the file to the start
            of the header
        data_start:
            byte offset from beginning of the file to the start
            of the data section
        data_end:
            byte offset from beginning of the file to the end
            of the data section

        Note these are also in the information dictionary, which
        you can access with get_info()
        """
        return dict(
            header_start=self._info['header_start'],
            data_start=self._info['data_start'],
            data_end=self._info['data_end'],
        )

    def get_info(self):
        """
        Get a copy of the internal dictionary holding extension information
        """
        return copy.deepcopy(self._info)

    def get_filename(self):
        """
        Get a copy of the filename for this fits file
        """
        return copy.copy(self._filename)

    def write_checksum(self):
        """
        Write the checksum into the header for this HDU.

        Computes the checksum for the HDU, both the data portion alone (DATASUM
        keyword) and the checksum complement for the entire HDU (CHECKSUM).

        returns
        -------
        A dict with keys 'datasum' and 'hdusum'
        """
        return self._FITS.write_checksum(self._ext+1)

    def verify_checksum(self):
        """
        Verify the checksum in the header for this HDU.
        """
        res = self._FITS.verify_checksum(self._ext+1)
        if res['dataok'] != 1:
            raise ValueError("data checksum failed")
        if res['hduok'] != 1:
            raise ValueError("hdu checksum failed")

    def write_comment(self, comment):
        """
        Write a comment into the header
        """
        self._FITS.write_comment(self._ext+1, str(comment))

    def write_history(self, history):
        """
        Write history text into the header
        """
        self._FITS.write_history(self._ext+1, str(history))

    def write_key(self, keyname, value, comment=""):
        """
        Write the input value to the header

        parameters
        ----------
        keyname: string
            Name of keyword to write/update
        value: scalar
            Value to write, can be string float or integer type,
            including numpy scalar types.
        comment: string, optional
            An optional comment to write for this key

        Notes
        -----
        Write COMMENT and HISTORY using the write_comment and write_history
        methods
        """

        if value is None:
            value=''

        if isinstance(value,bool):
            if value:
                v=1
            else:
                v=0
            self._FITS.write_logical_key(self._ext+1,
                                         str(keyname),
                                         v,
                                         str(comment))
        elif isinstance(value, _stypes):
            self._FITS.write_string_key(self._ext+1,
                                        str(keyname),
                                        str(value),
                                        str(comment))
        elif isinstance(value, _ftypes):
            self._FITS.write_double_key(self._ext+1,
                                        str(keyname),
                                        float(value),
                                        str(comment))
        elif isinstance(value, _itypes):
            self._FITS.write_long_key(self._ext+1,
                                      str(keyname),
                                      int(value),
                                      str(comment))
        elif isinstance(value,(tuple,list)):
            vl=[str(el) for el in value]
            sval=','.join(vl)
            self._FITS.write_string_key(self._ext+1,
                                        str(keyname),
                                        sval,
                                        str(comment))
        else:
            sval=str(value)
            mess=("warning, keyword '%s' has non-standard "
                  "value type %s, "
                  "Converting to string: '%s'")
            warnings.warn(mess % (keyname,type(value),sval), FITSRuntimeWarning)
            self._FITS.write_string_key(self._ext+1,
                                        str(keyname),
                                        sval,
                                        str(comment))

    def write_keys(self, records_in, clean=True):
        """
        Write the keywords to the header.

        parameters
        ----------
        records: FITSHDR or list or dict
            Can be one of these:
                - FITSHDR object
                - list of dictionaries containing 'name','value' and optionally
                  a 'comment' field.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary
        clean: boolean
            If True, trim out the standard fits header keywords that are
            created on HDU creation, such as EXTEND, SIMPLE, STTYPE, TFORM,
            TDIM, XTENSION, BITPIX, NAXIS, etc.

        Notes
        -----
        Input keys named COMMENT and HISTORY are written using the
        write_comment and write_history methods.
        """

        if isinstance(records_in,FITSHDR):
            hdr = records_in
        else:
            hdr = FITSHDR(records_in)

        if clean:
            is_table = isinstance(self, TableHDU)
            hdr.clean(is_table=is_table)

        for r in hdr.records():
            name=r['name'].upper()
            value=r['value']

            if name=='COMMENT':
                self.write_comment(value)
            elif name=='HISTORY':
                self.write_history(value)
            else:
                comment=r.get('comment','')
                self.write_key(name,value,comment=comment)


    def read_header(self):
        """
        Read the header as a FITSHDR

        The FITSHDR allows access to the values and comments by name and
        number.
        """
        # note converting strings
        return FITSHDR(self.read_header_list(), convert=True)


    def read_header_list(self):
        """
        Read the header as a list of dictionaries.

        You will usually use read_header instead, which just sends the output
        of this functioin to the constructor of a FITSHDR, which allows access
        to the values and comments by name and number.

        Each dictionary is
            'name': the keyword name
            'value': the value field as a string
            'comment': the comment field as a string.
        """
        return self._FITS.read_header(self._ext+1)


    def _update_info(self):
        """
        Update metadata for this HDU
        """
        try:
            self._FITS.movabs_hdu(self._ext+1)
        except IOError:
            raise RuntimeError("no such hdu")

        self._info = self._FITS.get_hdu_info(self._ext+1)

    def _get_repr_list(self):
        """
        Get some representation data common to all HDU types
        """
        spacing = ' '*2
        text = ['']
        text.append("%sfile: %s" % (spacing,self._filename))
        text.append("%sextension: %d" % (spacing,self._info['hdunum']-1))
        text.append("%stype: %s" % (spacing,_hdu_type_map[self._info['hdutype']]))

        extname=self.get_extname()
        if extname != "":
            text.append("%sextname: %s" % (spacing,extname))
        extver=self.get_extver()
        if extver != 0:
            text.append("%sextver: %s" % (spacing,extver))
        
        return text, spacing

class TableHDU(HDUBase):
    """
    A class representing a table HDU
    """
    def __init__(self, fits, ext, **keys):
        super(TableHDU,self).__init__(fits, ext, **keys)

        self.lower=keys.get('lower',False)
        self.upper=keys.get('upper',False)
        self._vstorage=keys.get('vstorage','fixed')
        self.case_sensitive=keys.get('case_sensitive',False)
        self._iter_row_buffer=keys.get('iter_row_buffer',1)

        if self._info['hdutype'] == ASCII_TBL:
            self._table_type_str='ascii'
        else:
            self._table_type_str='binary'

    def get_nrows(self):
        """
        Get number of rows in the table.
        """
        nrows=self._info.get('nrows',None)
        if nrows is None:
            raise ValueError("nrows not in info table; this is a bug")
        return nrows

    def get_colnames(self):
        """
        Get a copy of the column names for a table HDU
        """
        return copy.copy(self._colnames)

    def get_colname(self, colnum):
        """
        Get the name associated with the given column number

        parameters
        ----------
        colnum: integer
            The number for the column, zero offset
        """
        if colnum < 0 or colnum > (len(self._colnames)-1):
            raise ValueError("colnum out of range [0,%s-1]" % (0,len(self._colnames)))
        return self._colnames[colnum]

    def get_vstorage(self):
        """
        Get a string representing the storage method for variable length 
        columns
        """
        return copy.copy(self._vstorage)

    def has_data(self):
        """
        Determine if this HDU has any data

        Check that the row count is not zero
        """
        if self._info['nrows'] > 0:
            return True
        else:
            return False

    def where(self, expression):
        """
        Return the indices where the expression evaluates to true.

        parameters
        ----------
        expression: string
            A fits row selection expression.  E.g.
            "x > 3 && y < 5"
        """

        return self._FITS.where(self._ext+1, expression)

    def write(self, data, **keys):
        """
        Write data into this HDU

        parameters
        ----------
        data: ndarray or list of ndarray
            A numerical python array.  Should be an ordinary array for image
            HDUs, should have fields for tables.  To write an ordinary array to
            a column in a table HDU, use write_column.  If data already exists
            in this HDU, it will be overwritten.  See the append(() method to
            append new rows to a table HDU.
        firstrow: integer, optional
            At which row you should begin writing to tables.  Be sure you know
            what you are doing!  For appending see the append() method.
            Default 0.
        columns: list, optional
            If data is a list of arrays, you must send columns as a list
            of names or column numbers

            You can also send names=
        names: list, optional
            same as columns=
        """

        slow = keys.get('slow',False)

        isrec=False
        if isinstance(data,(list,dict)):
            if isinstance(data,list):
                data_list=data
                columns_all = keys.get('columns',None)
                if columns_all is None:
                    columns_all=keys.get('names',None)
                    if columns_all is None:
                        raise ValueError("you must send columns with a list of arrays")

            else:
                columns_all=list(data.keys())
                data_list=[data[n] for n in columns_all]

            colnums_all = [self._extract_colnum(c) for c in columns_all]
            names = [self.get_colname(c) for c in colnums_all]

            isobj=numpy.zeros(len(data_list),dtype=numpy.bool)
            for i in xrange(len(data_list)):
                isobj[i] = is_object(data_list[i])

        else:
            if data.dtype.fields is None:
                raise ValueError("You are writing to a table, so I expected "
                                 "an array with fields as input. If you want "
                                 "to write a simple array, you should use "
                                 "write_column to write to a single column, "
                                 "or instead write to an image hdu")

            if data.shape is ():
                raise ValueError("cannot write data with shape ()")

            isrec=True
            names=data.dtype.names
            # only write object types (variable-length columns) after
            # writing the main table
            isobj = fields_are_object(data)

            data_list = []
            colnums_all=[]
            for i,name in enumerate(names):
                colnum = self._extract_colnum(name)
                data_list.append(data[name])
                colnums_all.append(colnum)
     
        if slow:
            for i,name in enumerate(names):
                if not isobj[i]:
                    self.write_column(name, data_list[i], **keys)
        else:

            nonobj_colnums = []
            nonobj_arrays = []
            for i in xrange(len(data_list)):
                if not isobj[i]:
                    nonobj_colnums.append(colnums_all[i])
                    if isrec:
                        # this still leaves possibility of f-order sub-arrays..
                        colref=array_to_native(data_list[i],inplace=False)
                    else:
                        colref=array_to_native_c(data_list[i],inplace=False)
                    nonobj_arrays.append(colref)

            if len(nonobj_arrays) > 0:
                firstrow=keys.get('firstrow',0)
                self._FITS.write_columns(self._ext+1, nonobj_colnums, nonobj_arrays, 
                                         firstrow=firstrow+1)

        # writing the object arrays always occurs the same way
        # need to make sure this works for array fields
        for i,name in enumerate(names):
            if isobj[i]:
                self.write_var_column(name, data_list[i], **keys)

        self._update_info()

    def write_column(self, column, data, **keys):
        """
        Write data to a column in this HDU

        This HDU must be a table HDU.

        parameters
        ----------
        column: scalar string/integer
            The column in which to write.  Can be the name or number (0 offset)
        column: ndarray
            Numerical python array to write.  This should match the
            shape of the column.  You are probably better using fits.write_table()
            to be sure.
        firstrow: integer, optional
            At which row you should begin writing.  Be sure you know what you
            are doing!  For appending see the append() method.  Default 0.
        """

        firstrow=keys.get('firstrow',0)

        colnum = self._extract_colnum(column)

        # need it to be contiguous and native byte order.  For now, make a
        # copy.  but we may be able to avoid this with some care.

        if not data.flags['C_CONTIGUOUS']:
            # this always makes a copy
            data_send = numpy.ascontiguousarray(data)
            # this is a copy, we can make sure it is native
            # and modify in place if needed
            array_to_native(data_send, inplace=True)
        else:
            # we can avoid the copy with a try-finally block and
            # some logic
            data_send = array_to_native(data, inplace=False)

        self._FITS.write_column(self._ext+1, colnum+1, data_send, 
                                firstrow=firstrow+1)
        del data_send
        self._update_info()

    def write_var_column(self, column, data, firstrow=0, **keys):
        """
        Write data to a variable-length column in this HDU

        This HDU must be a table HDU.

        parameters
        ----------
        column: scalar string/integer
            The column in which to write.  Can be the name or number (0 offset)
        column: ndarray
            Numerical python array to write.  This must be an object array.
        firstrow: integer, optional
            At which row you should begin writing.  Be sure you know what you
            are doing!  For appending see the append() method.  Default 0.
        """

        if not is_object(data):
            raise ValueError("Only object fields can be written to "
                             "variable-length arrays")
        colnum = self._extract_colnum(column)

        self._FITS.write_var_column(self._ext+1, colnum+1, data,
                                    firstrow=firstrow+1)
        self._update_info()

    def insert_column(self, name, data, colnum=None):
        """
        Insert a new column.

        parameters
        ----------
        name: string
            The column name
        data:
            The data to write into the new column.
        colnum: int, optional
            The column number for the new column, zero-offset.  Default
            is to add the new column after the existing ones.

        Notes
        -----
        This method is used un-modified by ascii tables as well.
        """

        if name in self._colnames:
            raise ValueError("column '%s' already exists" % name)

        descr=data.dtype.descr
        if len(descr) > 1:
            raise ValueError("you can only insert a single column, "
                             "requested: %s" % descr)

        this_descr = descr[0]
        this_descr = [name, this_descr[1]]
        if len(data.shape) > 1:
            this_descr += [data.shape[1:]]
        this_descr = tuple(this_descr)

        name, fmt, dims = npy2fits(this_descr, 
                                   table_type=self._table_type_str)
        if dims is not None:
            dims=[dims]

        if colnum is None:
            new_colnum = len(self._info['colinfo']) + 1
        else:
            new_colnum = colnum+1
        self._FITS.insert_col(self._ext+1, new_colnum, name, fmt, tdim=dims)
        self._update_info()

        self.write_column(name, data)

    def append(self, data, **keys):
        """
        Append new rows to a table HDU

        parameters
        ----------
        data: ndarray or list of arrays

            A numerical python array with fields (recarray) or a list of
            arrays.  Should have the same fields as the existing table. If only
            a subset of the table columns are present, the other columns are
            filled with zeros.

        columns: list, optional
            if a list of arrays is sent, also send the columns
            of names or column numbers
        """

        firstrow=self._info['nrows']

        #if data.dtype.fields is None:
        #    raise ValueError("got an ordinary array, can only append recarrays.  "
        #                     "using this method")

        # make sure these columns exist
        #for n in data.dtype.names:
        #    colnum = self._extract_colnum(n)

        keys['firstrow'] = firstrow
        self.write(data, **keys)

    def read(self, **keys):
        """
        read data from this HDU

        By default, all data are read.  
        
        send columns= and rows= to select subsets of the data.
        Table data are read into a recarray; use read_column() to get a single
        column as an ordinary array.  You can alternatively use slice notation
            fits=fitsio.FITS(filename)
            fits[ext][:]
            fits[ext][2:5]
            fits[ext][200:235:2]
            fits[ext][rows]
            fits[ext][cols][rows]

        parameters
        ----------
        columns: optional
            An optional set of columns to read from table HDUs.  Default is to
            read all.  Can be string or number.  If a sequence, a recarray
            is always returned.  If a scalar, an ordinary array is returned.
        rows: optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        """

        columns = keys.get('columns',None)
        rows    = keys.get('rows',None)

        if columns is not None:
            if 'columns' in keys: 
                del keys['columns']
            data = self.read_columns(columns, **keys)
        elif rows is not None:
            if 'rows' in keys: 
                del keys['rows']
            data = self.read_rows(rows, **keys)
        else:
            data = self._read_all(**keys)

        return data

    def _read_all(self, **keys):
        """
        Read all data in the HDU.

        parameters
        ----------
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        """

        dtype, offsets, isvar = self.get_rec_dtype(**keys)

        w,=numpy.where(isvar == True)
        if w.size > 0:
            vstorage = keys.get('vstorage',self._vstorage)
            colnums = self._extract_colnums()
            rows=None
            array = self._read_rec_with_var(colnums, rows, dtype,
                                            offsets, isvar, vstorage)
        else:

            firstrow=1
            nrows = self._info['nrows']
            array = numpy.zeros(nrows, dtype=dtype)

            self._FITS.read_as_rec(self._ext+1, 1, nrows, array)

            for colnum,name in enumerate(array.dtype.names):
                self._rescale_and_convert_field_inplace(array,
                                          name,
                                          self._info['colinfo'][colnum]['tscale'], 
                                          self._info['colinfo'][colnum]['tzero'])
        lower=keys.get('lower',False)
        upper=keys.get('upper',False)
        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        return array

    def read_column(self, col, **keys):
        """
        Read the specified column

        Alternatively, you can use slice notation
            fits=fitsio.FITS(filename)
            fits[ext][colname][:]
            fits[ext][colname][2:5]
            fits[ext][colname][200:235:2]
            fits[ext][colname][rows]

        Note, if reading multiple columns, it is more efficient to use
        read(columns=) or slice notation with a list of column names.

        parameters
        ----------
        col: string/int,  required
            The column name or number.
        rows: optional
            An optional set of row numbers to read.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        """

        res = self.read_columns([col], **keys)
        colname = res.dtype.names[0]
        return res[colname]

        '''
        # deprecated

        rows=keys.get('rows',None)
        colnum = self._extract_colnum(col)
        # ensures unique, contiguous
        rows = self._extract_rows(rows)

        if self._info['colinfo'][colnum]['eqtype'] < 0:
            vstorage=keys.get('vstorage',self._vstorage)
            return self._read_var_column(colnum, rows, vstorage)
        else:
            npy_type, shape = self._get_simple_dtype_and_shape(colnum, rows=rows)

            array = numpy.zeros(shape, dtype=npy_type)

            self._FITS.read_column(self._ext+1,colnum+1, array, rows)
            
            array=self._rescale_and_convert(array, 
                                            self._info['colinfo'][colnum]['tscale'], 
                                            self._info['colinfo'][colnum]['tzero'])

        return array
        '''

    def read_rows(self, rows, **keys):
        """
        Read the specified rows.

        parameters
        ----------
        rows: list,array
            A list or array of row indices.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        """
        if rows is None:
            # we actually want all rows!
            return self._read_all()

        if self._info['hdutype'] == ASCII_TBL:
            keys['rows'] = rows
            return self.read(**keys)

        rows = self._extract_rows(rows)
        dtype, offsets, isvar = self.get_rec_dtype(**keys)

        w,=numpy.where(isvar == True)
        if w.size > 0:
            vstorage = keys.get('vstorage',self._vstorage)
            colnums=self._extract_colnums()
            return self._read_rec_with_var(colnums, rows, dtype, offsets, isvar, vstorage)
        else:
            array = numpy.zeros(rows.size, dtype=dtype)
            self._FITS.read_rows_as_rec(self._ext+1, array, rows)

            for colnum,name in enumerate(array.dtype.names):
                self._rescale_and_convert_field_inplace(array,
                                          name,
                                          self._info['colinfo'][colnum]['tscale'], 
                                          self._info['colinfo'][colnum]['tzero'])

        lower=keys.get('lower',False)
        upper=keys.get('upper',False)
        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)


        return array


    def read_columns(self, columns, **keys):
        """
        read a subset of columns from this binary table HDU

        By default, all rows are read.  Send rows= to select subsets of the
        data.  Table data are read into a recarray for multiple columns,
        plain array for a single column.

        parameters
        ----------
        columns: list/array
            An optional set of columns to read from table HDUs.  Can be string
            or number. If a sequence, a recarray is always returned.  If a
            scalar, an ordinary array is returned.
        rows: list/array, optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        """

        if self._info['hdutype'] == ASCII_TBL:
            keys['columns'] = columns
            return self.read(**keys)

        rows = keys.get('rows',None)

        # if columns is None, returns all.  Guaranteed to be unique and sorted
        colnums = self._extract_colnums(columns)
        if isinstance(colnums,int):
            # scalar sent, don't read as a recarray
            return self.read_column(columns, **keys)

        # if rows is None still returns None, and is correctly interpreted
        # by the reader to mean all
        rows = self._extract_rows(rows)

        # this is the full dtype for all columns
        dtype, offsets, isvar = self.get_rec_dtype(colnums=colnums, **keys)

        w,=numpy.where(isvar == True)
        if w.size > 0:
            vstorage = keys.get('vstorage',self._vstorage)
            array = self._read_rec_with_var(colnums, rows, dtype, offsets, isvar, vstorage)
        else:

            if rows is None:
                nrows = self._info['nrows']
            else:
                nrows = rows.size
            array = numpy.zeros(nrows, dtype=dtype)

            colnumsp = colnums[:].copy()
            colnumsp[:] += 1
            self._FITS.read_columns_as_rec(self._ext+1, colnumsp, array, rows)

            for i in xrange(colnums.size):
                colnum = int(colnums[i])
                name = array.dtype.names[i]
                self._rescale_and_convert_field_inplace(array,
                                          name,
                                          self._info['colinfo'][colnum]['tscale'], 
                                          self._info['colinfo'][colnum]['tzero'])

        lower=keys.get('lower',False)
        upper=keys.get('upper',False)
        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        return array

    def read_slice(self, firstrow, lastrow, step=1, **keys):
        """
        Read the specified row slice from a table.

        Read all rows between firstrow and lastrow (non-inclusive, as per
        python slice notation).  Note you must use slice notation for
        images, e.g. f[ext][20:30, 40:50]

        parameters
        ----------
        firstrow: integer
            The first row to read
        lastrow: integer
            The last row to read, non-inclusive.  This follows the python list
            slice convention that one does not include the last element.
        step: integer, optional
            Step between rows, default 1. e.g., if step is 2, skip every other row.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        """

        if self._info['hdutype'] == ASCII_TBL:
            rows = numpy.arange(firstrow, lastrow, step, dtype='i8')
            keys['rows'] = rows
            return self.read_ascii(**keys)

        step=keys.get('step',1)
        if self._info['hdutype'] == IMAGE_HDU:
            raise ValueError("slices currently only supported for tables")

        maxrow = self._info['nrows']
        if firstrow < 0 or lastrow > maxrow:
            raise ValueError("slice must specify a sub-range of [%d,%d]" % (0,maxrow))

        dtype, offsets, isvar = self.get_rec_dtype(**keys)

        w,=numpy.where(isvar == True)
        if w.size > 0:
            vstorage = keys.get('vstorage',self._vstorage)
            rows=numpy.arange(firstrow,lastrow,step,dtype='i8')
            colnums=self._extract_colnums()
            array = self._read_rec_with_var(colnums, rows, dtype, offsets, isvar, vstorage)
        else:
            if step != 1:
                rows = numpy.arange(firstrow, lastrow, step, dtype='i8')
                array = self.read(rows=rows)
            else:
                # no +1 because lastrow is non-inclusive
                nrows=lastrow-firstrow
                array = numpy.zeros(nrows, dtype=dtype)

                # only first needs to be +1.  This is becuase the c code is inclusive
                self._FITS.read_as_rec(self._ext+1, firstrow+1, lastrow, array)

                for colnum,name in enumerate(array.dtype.names):
                    self._rescale_and_convert_field_inplace(array, 
                                              name,
                                              self._info['colinfo'][colnum]['tscale'], 
                                              self._info['colinfo'][colnum]['tzero'])

        lower=keys.get('lower',False)
        upper=keys.get('upper',False)
        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)



        return array

    def get_rec_dtype(self, **keys):
        """
        Get the dtype for the specified columns

        parameters
        ----------
        colnums: integer array
            The column numbers, 0 offset
        vstorage: string, optional
            See docs in read_columns
        """
        colnums=keys.get('colnums',None)
        vstorage = keys.get('vstorage',self._vstorage)

        if colnums is None:
            colnums = self._extract_colnums()


        descr = []
        isvararray = numpy.zeros(len(colnums),dtype=numpy.bool)
        for i,colnum in enumerate(colnums):
            dt,isvar = self.get_rec_column_descr(colnum, vstorage)
            descr.append(dt)
            isvararray[i] = isvar
        dtype=numpy.dtype(descr)

        offsets = numpy.zeros(len(colnums),dtype='i8')
        for i,n in enumerate(dtype.names):
            offsets[i] = dtype.fields[n][1]
        return dtype, offsets, isvararray

    def _get_simple_dtype_and_shape(self, colnum, rows=None):
        """
        When reading a single column, we want the basic data
        type and the shape of the array.

        for scalar columns, shape is just nrows, otherwise
        it is (nrows, dim1, dim2)

        Note if rows= is sent and only a single row is requested,
        the shape will be (dim2,dim2)


        """

        # basic datatype
        npy_type,isvar = self._get_tbl_numpy_dtype(colnum)
        info = self._info['colinfo'][colnum]
        name = info['name']

        if rows is None:
            nrows = self._info['nrows']
        else:
            nrows = rows.size

        shape = None
        tdim = info['tdim']

        shape = tdim2shape(tdim, name, is_string=(npy_type[0] == 'S'))
        if shape is not None:
            if nrows > 1:
                if not isinstance(shape,tuple):
                    # vector
                    shape = (nrows,shape)
                else:
                    # multi-dimensional
                    shape = tuple( [nrows] + list(shape) )
        else:
            # scalar
            shape = nrows
        return npy_type, shape

    def get_rec_column_descr(self, colnum, vstorage):
        """
        Get a descriptor entry for the specified column.

        parameters
        ----------
        colnum: integer
            The column number, 0 offset
        vstorage: string
            See docs in read_columns
        """
        npy_type,isvar = self._get_tbl_numpy_dtype(colnum)
        name = self._info['colinfo'][colnum]['name']

        if isvar:
            if vstorage == 'object':
                descr=(name,'O')
            else:
                tform = self._info['colinfo'][colnum]['tform']
                max_size = extract_vararray_max(tform)

                if max_size <= 0:
                    name=self._info['colinfo'][colnum]['name']
                    mess='Will read as an object field'
                    if max_size < 0:
                        mess="Column '%s': No maximum size: '%s'. %s"
                        mess=mess % (name,tform,mess)
                        warnings.warn(mess, FITSRuntimeWarning)
                    else:
                        mess="Column '%s': Max size is zero: '%s'. %s"
                        mess=mess % (name,tform,mess)
                        warnings.warn(mess, FITSRuntimeWarning)

                    # we are forced to read this as an object array
                    return self.get_rec_column_descr(colnum, 'object')

                if npy_type[0] == 'S':
                    # variable length string columns cannot
                    # themselves be arrays I don't think
                    npy_type = 'S%d' % max_size
                    descr=(name,npy_type)
                else:
                    descr=(name,npy_type,max_size)
        else:
            tdim = self._info['colinfo'][colnum]['tdim']
            shape = tdim2shape(tdim, name, is_string=(npy_type[0] == 'S'))
            if shape is not None:
                descr=(name,npy_type,shape)
            else:
                descr=(name,npy_type)
        return descr,isvar


    def _read_rec_with_var(self, colnums, rows, dtype, offsets, isvar, vstorage):
        """

        Read columns from a table into a rec array, including variable length
        columns.  This is special because, for efficiency, it involves reading
        from the main table as normal but skipping the columns in the array
        that are variable.  Then reading the variable length columns, with
        accounting for strides appropriately.

        row and column numbers should be checked before calling this function

        """

        colnumsp=colnums+1
        if rows is None:
            nrows = self._info['nrows']
        else:
            nrows = rows.size
        array = numpy.zeros(nrows, dtype=dtype)

        # read from the main table first
        wnotvar,=numpy.where(isvar == False)
        if wnotvar.size > 0:
            thesecol=colnumsp[wnotvar] # this will be contiguous (not true for slices)
            theseoff=offsets[wnotvar]
            self._FITS.read_columns_as_rec_byoffset(self._ext+1,
                                                    thesecol,
                                                    theseoff,
                                                    array,
                                                    rows)
            for i in xrange(thesecol.size):

                name = array.dtype.names[wnotvar[i]]
                colnum = thesecol[i]-1
                self._rescale_and_convert_field_inplace(array, 
                                          name,
                                          self._info['colinfo'][colnum]['tscale'], 
                                          self._info['colinfo'][colnum]['tzero'])


        # now read the variable length arrays we may be able to speed this up
        # by storing directly instead of reading first into a list
        wvar,=numpy.where(isvar == True)
        if wvar.size > 0:
            thesecol=colnumsp[wvar] # this will be contiguous (not true for slices)
            for i in xrange(thesecol.size):
                colnump = thesecol[i]
                name = array.dtype.names[wvar[i]]
                dlist = self._FITS.read_var_column_as_list(self._ext+1,colnump,rows)
                if isinstance(dlist[0],str):
                    is_string=True
                else:
                    is_string=False

                if array[name].dtype.descr[0][1][1] == 'O':
                    # storing in object array
                    # get references to each, no copy made
                    for irow,item in enumerate(dlist):
                        array[name][irow] = item
                else: 
                    for irow,item in enumerate(dlist):
                        if is_string:
                            array[name][irow]= item
                        else:
                            ncopy = len(item)

                            if sys.version_info > (3,0,0):
                                ts = array[name].dtype.descr[0][1][1]
                                if ts != 'S':
                                    array[name][irow][0:ncopy] = item[:]
                                else:
                                    array[name][irow] = item
                            else:
                                array[name][irow][0:ncopy] = item[:]

        return array

    def _extract_rows(self, rows):
        """
        Extract an array of rows from an input scalar or sequence
        """
        if rows is not None:
            rows = numpy.array(rows, ndmin=1, copy=False, dtype='i8')
            # returns unique, sorted
            rows = numpy.unique(rows)

            maxrow = self._info['nrows']-1
            if rows[0] < 0 or rows[-1] > maxrow:
                raise ValueError("rows must be in [%d,%d]" % (0,maxrow))
        return rows

    def _process_slice(self, arg):
        """
        process the input slice for use calling the C code
        """
        start = arg.start
        stop = arg.stop
        step = arg.step

        nrows=self._info['nrows']
        if step is None:
            step=1
        if start is None:
            start = 0
        if stop is None:
            stop = nrows

        if start < 0:
            start = nrows + start
            if start < 0:
                raise IndexError("Index out of bounds")

        if stop < 0:
            stop = nrows + start + 1

        if stop < start:
            # will return an empty struct
            stop = start

        if stop > nrows:
            stop=nrows
        return slice(start, stop, step)

    def _slice2rows(self, start, stop, step=None):
        """
        Convert a slice to an explicit array of rows
        """
        nrows=self._info['nrows']
        if start is None:
            start=0
        if stop is None:
            stop=nrows
        if step is None:
            step=1

        tstart = self._fix_range(start)
        tstop  = self._fix_range(stop)
        if tstart == 0 and tstop == nrows:
            # this is faster: if all fields are also requested, then a 
            # single fread will be done
            return None
        if stop < start:
            raise ValueError("start is greater than stop in slice")
        return numpy.arange(tstart, tstop, step, dtype='i8')

    def _fix_range(self, num, isslice=True):
        """
        Ensure the input is within range.

        If el=True, then don't treat as a slice element
        """

        nrows = self._info['nrows']
        if isslice:
            # include the end
            if num < 0:
                num=nrows + (1+num)
            elif num > nrows:
                num=nrows
        else:
            # single element
            if num < 0:
                num=nrows + num
            elif num > (nrows-1):
                num=nrows-1

        return num

    def _rescale_and_convert_field_inplace(self, array, name, scale, zero):
        """
        Apply fits scalings.  Also, convert bool to proper
        numpy boolean values
        """
        self._rescale_array(array[name], scale, zero)
        if array[name].dtype==numpy.bool:
            array[name] = self._convert_bool_array(array[name])
            
        return array

    def _rescale_and_convert(self, array, scale, zero, name=None):
        """
        Apply fits scalings.  Also, convert bool to proper
        numpy boolean values
        """
        self._rescale_array(array, scale, zero)
        if array.dtype==numpy.bool:
            array = self._convert_bool_array(array)
            
        return array


    def _rescale_array(self, array, scale, zero):
        """
        Scale the input array
        """
        if scale != 1.0:
            sval=numpy.array(scale,dtype=array.dtype)
            array *= sval
        if zero != 0.0:
            zval=numpy.array(zero,dtype=array.dtype)
            array += zval

    def _convert_bool_array(self, array):
        """
        cfitsio reads as characters 'T' and 'F' -- convert to real boolean
        If input is a fits bool, convert to numpy boolean
        """

        output = (array.view(numpy.int8) == ord('T')).astype(numpy.bool)
        return output

    def _get_tbl_numpy_dtype(self, colnum, include_endianness=True):
        """
        Get numpy type for the input column
        """
        table_type = self._info['hdutype']
        table_type_string = _hdu_type_map[table_type]
        try:
            ftype = self._info['colinfo'][colnum]['eqtype']
            if table_type == ASCII_TBL:
                npy_type = _table_fits2npy_ascii[abs(ftype)]
            else:
                npy_type = _table_fits2npy[abs(ftype)]
        except KeyError:
            raise KeyError("unsupported %s fits data "
                           "type: %d" % (table_type_string, ftype))

        isvar=False
        if ftype < 0:
            isvar=True
        if include_endianness:
            # if binary we will read the big endian bytes directly,
            # if ascii we read into native byte order
            if table_type == ASCII_TBL:
                addstr=''
            else:
                addstr='>'
            if npy_type not in ['u1','i1','S']:
                npy_type = addstr+npy_type

        if npy_type == 'S':
            width = self._info['colinfo'][colnum]['width']
            npy_type = 'S%d' % width
        return npy_type, isvar


    def _process_args_as_rows_or_columns(self, arg, unpack=False):
        """
        We must be able to interpret the args as as either a column name or
        row number, or sequences thereof.  Numpy arrays and slices are also
        fine.

        Examples:
            'field'
            35
            [35,55,86]
            ['f1',f2',...]
        Can also be tuples or arrays.
        """

        isslice = False
        isrows = False
        result=arg
        if isinstance(arg, (tuple,list,numpy.ndarray)):
            # a sequence was entered
            if isstring(arg[0]):
                pass
            else:
                isrows=True
                result = arg
        elif isstring(arg):
            # a single string was entered
            pass
        elif isinstance(arg, slice):
            isrows=True
            if unpack:
                result = self._slice2rows(arg.start, arg.stop, arg.step)
            else:
                isslice=True
                result = self._process_slice(arg)
        else:
            # a single object was entered.  Probably should apply some more 
            # checking on this
            isrows=True

        return result, isrows, isslice

    def _read_var_column(self, colnum, rows, vstorage):
        """

        first read as a list of arrays, then copy into either a fixed length
        array or an array of objects, depending on vstorage.

        """

        if sys.version_info > (3,0,0):
            stype=bytes
        else:
            stype=str

        dlist = self._FITS.read_var_column_as_list(self._ext+1,colnum+1,rows)

        if vstorage == 'fixed':

            tform = self._info['colinfo'][colnum]['tform']
            max_size = extract_vararray_max(tform)

            if max_size <= 0:
                name=self._info['colinfo'][colnum]['name']
                mess='Will read as an object field'
                if max_size < 0:
                    mess="Column '%s': No maximum size: '%s'. %s"
                    mess=mess % (name,tform,mess)
                    warnings.warn(mess, FITSRuntimeWarning)
                else:
                    mess="Column '%s': Max size is zero: '%s'. %s"
                    mess=mess % (name,tform,mess)
                    warnings.warn(mess, FITSRuntimeWarning)

                # we are forced to read this as an object array
                return self._read_var_column(colnum, rows, 'object')

            if isinstance(dlist[0],stype):
                descr = 'S%d' % max_size
                array = numpy.fromiter(dlist, descr)
            else:
                descr=dlist[0].dtype.str
                array = numpy.zeros( (len(dlist), max_size), dtype=descr)

                for irow,item in enumerate(dlist):
                    ncopy = len(item)
                    array[irow,0:ncopy] = item[:]
        else:
            array=numpy.zeros(len(dlist), dtype='O')
            for irow,item in enumerate(dlist):
                array[irow] = item

        return array

    def _extract_colnums(self, columns=None):
        """
        Extract an array of columns from the input
        """
        if columns is None:
            return numpy.arange(self._ncol, dtype='i8')
        
        if not isinstance(columns,(tuple,list,numpy.ndarray)):
            # is a scalar
            return self._extract_colnum(columns)

        colnums = numpy.zeros(len(columns), dtype='i8')
        for i in xrange(colnums.size):
            colnums[i] = self._extract_colnum(columns[i])

        # returns unique sorted
        colnums = numpy.unique(colnums)
        return colnums

    def _extract_colnum(self, col):
        """
        Get the column number for the input column
        """
        if isinteger(col):
            colnum = col

            if (colnum < 0) or (colnum > (self._ncol-1)):
                raise ValueError("column number should be in [0,%d]" % (0,self._ncol-1))
        else:
            colstr='%s' % col
            try:
                if self.case_sensitive:
                    mess="column name '%s' not found (case sensitive)" % col
                    colnum = self._colnames.index(colstr)
                else:
                    mess="column name '%s' not found (case insensitive)" % col
                    colnum = self._colnames_lower.index(colstr.lower())
            except ValueError:
                raise ValueError(mess)
        return int(colnum)

    def _update_info(self):
        """
        Call parent method and make sure this is in fact a
        table HDU.  Set some convenience data.
        """
        super(TableHDU,self)._update_info()
        if self._info['hdutype'] == IMAGE_HDU:
            mess="Extension %s is not a Table HDU" % self.ext
            raise ValueError(mess)
        if 'colinfo' in self._info:
            self._colnames = [i['name'] for i in self._info['colinfo']]
            self._colnames_lower = [i['name'].lower() for i in self._info['colinfo']]
            self._ncol = len(self._colnames)

    def __getitem__(self, arg):
        """
        Get data from a table using python [] notation.  
        
        You can use [] to extract column and row subsets, or read everything.
        The notation is essentially the same as numpy [] notation, except that
        a sequence of column names may also be given.  Examples reading from
        "filename", extension "ext"

            fits=fitsio.FITS(filename)
            fits[ext][:]
            fits[ext][2:5]
            fits[ext][200:235:2]
            fits[ext][rows]
            fits[ext][cols][rows]

        Note data are only read once the rows are specified.

        Note you can only read variable length arrays the default way,
        using this function, so set it as you want on construction.

        This function is used for ascii tables as well
        """

        res, isrows, isslice = \
            self._process_args_as_rows_or_columns(arg)

        if isrows:
            # rows were entered: read all columns
            if isslice:
                array = self.read_slice(res.start, res.stop, res.step)
            else:
                # will also get here if slice is entered but this
                # is an ascii table
                array = self.read(rows=res)
        else:
            return TableColumnSubset(self, res)

        if self.lower:
            _names_to_lower_if_recarray(array)
        elif self.upper:
            _names_to_upper_if_recarray(array)

        return array

    def __iter__(self):
        """
        Get an iterator for a table

        e.g.
        f=fitsio.FITS(fname)
        hdu1 = f[1]
        for row in hdu1:
            ...
        """

        # always start with first row
        self._iter_row=0

        # for iterating we must assume the number of rows will not change
        self._iter_nrows=self.get_nrows()

        self._buffer_iter_rows(0)
        return self

    def next(self):
        """
        get the next row when iterating

        e.g.
        f=fitsio.FITS(fname)
        hdu1 = f[1]
        for row in hdu1:
            ...

        By default read one row at a time.  Send iter_row_buffer to get a more 
        efficient buffering.
        """
        return self._get_next_buffered_row()
    __next__=next

    def _get_next_buffered_row(self):
        """
        Get the next row for iteration.
        """
        if self._iter_row == self._iter_nrows:
            raise StopIteration

        if self._row_buffer_index >= self._iter_row_buffer:
            self._buffer_iter_rows(self._iter_row)

        data=self._row_buffer[self._row_buffer_index]
        self._iter_row += 1
        self._row_buffer_index += 1
        return data

    def _buffer_iter_rows(self, start):
        """
        Read in the buffer for iteration
        """
        self._row_buffer = self[start:start+self._iter_row_buffer]

        # start back at the front of the buffer
        self._row_buffer_index = 0

    def __repr__(self):
        """
        textual representation for some metadata
        """
        text, spacing = self._get_repr_list()

        text.append('%srows: %d' % (spacing,self._info['nrows']))
        text.append('%scolumn info:' % spacing)

        cspacing = ' '*4
        nspace = 4
        nname = 15
        ntype = 6
        format = cspacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
        pformat = cspacing + "%-" + str(nname) + "s\n %" + str(nspace+nname+ntype) + "s  %s"

        for colnum,c in enumerate(self._info['colinfo']):
            if len(c['name']) > nname:
                f = pformat
            else:
                f = format

            dt,isvar = self._get_tbl_numpy_dtype(colnum, include_endianness=False)
            if isvar:
                tform = self._info['colinfo'][colnum]['tform']
                if dt[0] == 'S':
                    dt = 'S0'
                    dimstr='vstring[%d]' % extract_vararray_max(tform)
                else:
                    dimstr = 'varray[%s]' % extract_vararray_max(tform)
            else:
                if dt[0] == 'S':
                    is_string=True
                else:
                    is_string=False
                dimstr = _get_col_dimstr(c['tdim'],is_string=is_string)

            s = f % (c['name'],dt,dimstr)
            text.append(s)

        text = '\n'.join(text)
        return text


class AsciiTableHDU(TableHDU):
    def read(self, **keys):
        """
        read a data from an ascii table HDU

        By default, all rows are read.  Send rows= to select subsets of the
        data.  Table data are read into a recarray for multiple columns,
        plain array for a single column.

        parameters
        ----------
        columns: list/array
            An optional set of columns to read from table HDUs.  Can be string
            or number. If a sequence, a recarray is always returned.  If a
            scalar, an ordinary array is returned.
        rows: list/array, optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        """

        rows = keys.get('rows',None)
        columns = keys.get('columns',None)

        # if columns is None, returns all.  Guaranteed to be unique and sorted
        colnums = self._extract_colnums(columns)
        if isinstance(colnums,int):
            # scalar sent, don't read as a recarray
            return self.read_column(columns, **keys)

        rows = self._extract_rows(rows)
        if rows is None:
            nrows = self._info['nrows']
        else:
            nrows = rows.size

        # if rows is None still returns None, and is correctly interpreted
        # by the reader to mean all
        rows = self._extract_rows(rows)

        # this is the full dtype for all columns
        dtype, offsets, isvar = self.get_rec_dtype(colnums=colnums, **keys)
        array = numpy.zeros(nrows, dtype=dtype)

        # note reading into existing data
        wnotvar,=numpy.where(isvar == False)
        if wnotvar.size > 0:
            for i in wnotvar:
                colnum = colnums[i]
                name=array.dtype.names[i]
                a=array[name].copy()
                self._FITS.read_column(self._ext+1,colnum+1, a, rows)
                array[name] = a
                del a

        wvar,=numpy.where(isvar == True)
        if wvar.size > 0:
            for i in wvar:
                colnum = colnums[i]
                name = array.dtype.names[i]
                dlist = self._FITS.read_var_column_as_list(self._ext+1,colnum+1,rows)
                if isinstance(dlist[0],str):
                    is_string=True
                else:
                    is_string=False

                if array[name].dtype.descr[0][1][1] == 'O':
                    # storing in object array
                    # get references to each, no copy made
                    for irow,item in enumerate(dlist):
                        array[name][irow] = item
                else: 
                    for irow,item in enumerate(dlist):
                        if is_string:
                            array[name][irow]= item
                        else:
                            ncopy = len(item)
                            array[name][irow][0:ncopy] = item[:]

        lower=keys.get('lower',False)
        upper=keys.get('upper',False)
        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)


        return array
    read_ascii=read

class ImageHDU(HDUBase):
    def _update_info(self):
        """
        Call parent method and make sure this is in fact a
        image HDU.  Set dims in C order
        """
        super(ImageHDU,self)._update_info()

        if self._info['hdutype'] != IMAGE_HDU:
            mess="Extension %s is not a Image HDU" % self.ext
            raise ValueError(mess)

        # convert to c order
        if 'dims' in self._info:
            self._info['dims'] = list( reversed(self._info['dims']) )

    def has_data(self):
        """
        Determine if this HDU has any data

        For images, check that the dimensions are not zero.

        For tables, check that the row count is not zero
        """
        ndims = self._info.get('ndims',0)
        if ndims == 0:
            return False
        else:
            return True

    def is_compressed(self):
        """
        returns true of this extension is compressed
        """
        return self._info['is_compressed_image']==1

    def get_comptype(self):
        """
        Get the compression type.

        None if the image is not compressed.
        """
        return self._info['comptype']

    def get_dims(self):
        """
        get the shape of the image.  Returns () for empty
        """
        if self._info['ndims'] != 0:
            dims = self._info['dims']
        else:
            dims = ()

        return dims

    def reshape(self, dims):
        """
        reshape an existing image to the requested dimensions

        parameters
        ----------
        dims: sequence
            Any sequence convertible to i8
        """

        adims = numpy.array(dims, ndmin=1, dtype='i8')
        self._FITS.reshape_image(self._ext+1, adims)

    def write(self, img, start=0, **keys):
        """
        Write the image into this HDU

        If data already exist in this HDU, they will be overwritten.  If the
        image to write is larger than the image on disk, or if the start 
        position is such that the write would extend beyond the existing
        dimensions, the on-disk image is expanded.

        parameters
        ----------
        img: ndarray
            A simple numpy ndarray
        start: integer or sequence
            Where to start writing data.  Can be an integer offset
            into the entire array, or a sequence determining where
            in N-dimensional space to start.
        """

        dims=self.get_dims()

        if img.dtype.fields is not None:
            raise ValueError("got recarray, expected regular ndarray")
        if img.size == 0:
            raise ValueError("data must have at least 1 row")

        # data must be c-contiguous and native byte order
        if not img.flags['C_CONTIGUOUS']:
            # this always makes a copy
            img_send = numpy.ascontiguousarray(img)
            array_to_native(img_send, inplace=True)
        else:
            img_send = array_to_native(img, inplace=False)

        if not numpy.isscalar(start):
            # convert to scalar offset
            # note we use the on-disk data type to get itemsize

            offset = _convert_full_start_to_offset(dims, start)
        else:
            offset = start

        # see if we need to resize the image
        if self.has_data():
            self._expand_if_needed(dims, img.shape, start, offset)

        self._FITS.write_image(self._ext+1, img_send, offset+1)
        self._update_info()


    def read(self, **keys):
        """
        Read the image.

        If the HDU is an IMAGE_HDU, read the corresponding image.  Compression
        and scaling are dealt with properly.
        """
        if not self.has_data():
            return None

        dtype, shape = self._get_dtype_and_shape()
        array = numpy.zeros(shape, dtype=dtype)
        self._FITS.read_image(self._ext+1, array)
        return array


    def _get_dtype_and_shape(self):
        """
        Get the numpy dtype and shape for image
        """
        npy_dtype = self._get_image_numpy_dtype()

        if self._info['ndims'] != 0:
            shape = self._info['dims']
        else:
            raise IOError("no image present in HDU")

        return npy_dtype, shape

    def _get_image_numpy_dtype(self):
        """
        Get the numpy dtype for the image
        """
        try:
            ftype = self._info['img_equiv_type']
            npy_type = _image_bitpix2npy[ftype]
        except KeyError:
            raise KeyError("unsupported fits data type: %d" % ftype)

        return npy_type

    def __getitem__(self, arg):
        """
        Get data from an image using python [] slice notation.

        e.g., [2:25, 4:45].
        """
        return self._read_image_slice(arg)

    def _read_image_slice(self, arg):
        """
        workhorse to read a slice
        """
        if 'ndims' not in self._info:
            raise ValueError("Attempt to slice empty extension")

        if isinstance(arg, slice):
            # one-dimensional, e.g. 2:20
            return self._read_image_slice((arg,))

        if not isinstance(arg, tuple):
            raise ValueError("arguments must be slices, one for each "
                             "dimension, e.g. [2:5] or [2:5,8:25] etc.")

        # should be a tuple of slices, one for each dimension
        # e.g. [2:3, 8:100]
        nd = len(arg)
        if nd != self._info['ndims']:
            raise ValueError("Got slice dimensions %d, "
                             "expected %d" % (nd,self._info['ndims']))


        targ=arg
        arg=[]
        for a in targ:
            if isinstance(a,slice):
                arg.append(a)
            elif isinstance(a,int):
                arg.append( slice(a,a+1,1) )
            else:
                raise ValueError("arguments must be slices, e.g. 2:12")

        dims=self._info['dims']
        arrdims = []
        first = []
        last = []
        steps = []

        # check the args and reverse dimensions since
        # fits is backwards from numpy
        dim=0
        for slc in arg:
            start = slc.start
            stop = slc.stop
            step = slc.step

            if start is None:
                start=0
            if stop is None:
                stop = dims[dim]
            if step is None:
                step=1
            if step < 1:
                raise ValueError("slice steps must be >= 1")

            if start < 0:
                start = dims[dim] + start
                if start < 0:
                    raise IndexError("Index out of bounds")

            if stop < 0:
                stop = dims[dim] + start + 1

            # move to 1-offset
            start = start + 1

            if stop < start:
                raise ValueError("python slices but include at least one "
                                 "element, got %s" % slc)
            if stop > dims[dim]:
                stop = dims[dim]

            first.append(start)
            last.append(stop)
            steps.append(step)
            arrdims.append(stop-start+1)

            dim += 1

        first.reverse()
        last.reverse()
        steps.reverse()
        first = numpy.array(first, dtype='i8')
        last  = numpy.array(last, dtype='i8')
        steps = numpy.array(steps, dtype='i8')

        npy_dtype = self._get_image_numpy_dtype()
        array = numpy.zeros(arrdims, dtype=npy_dtype)
        self._FITS.read_image_slice(self._ext+1, first, last, steps, array)
        return array

    def _expand_if_needed(self, dims, write_dims, start, offset):
        """
        expand the on-disk image if the indended write will extend
        beyond the existing dimensions
        """
        from operator import mul

        if numpy.isscalar(start):
            start_is_scalar=True
        else:
            start_is_scalar=False

        existing_size=reduce(mul, dims, 1)
        required_size = offset + reduce(mul, write_dims, 1)

        if required_size > existing_size:
            print("    required size:",required_size,"existing size:",existing_size)
            # we need to expand the image
            ndim=len(dims)
            idim=len(write_dims)

            if start_is_scalar:
                if start == 0:
                    start=[0]*ndim
                else:
                    raise ValueError("When expanding "
                                     "an existing image while writing, the start keyword "
                                     "must have the same number of dimensions "
                                     "as the image or be exactly 0, got %s " % start)

            if idim != ndim:
                raise ValueError("When expanding "
                                 "an existing image while writing, the input image "
                                 "must have the same number of dimensions "
                                 "as the original.  "
                                 "Got %d instead of %d" % (idim,ndim))
            new_dims = []
            for i in xrange(ndim):
                required_dim = start[i] + write_dims[i]

                if required_dim < dims[i]:
                    # careful not to shrink the image!
                    dimsize=dims[i]
                else:
                    dimsize=required_dim

                new_dims.append(dimsize)

            print("    reshaping image to:",new_dims)
            self.reshape(new_dims)

    def __repr__(self):
        """
        Representation for ImageHDU
        """
        text, spacing = self._get_repr_list()
        text.append("%simage info:" % spacing)
        cspacing = ' '*4

        # need this check for when we haven't written data yet
        if 'ndims' in self._info:
            if self._info['comptype'] is not None:
                text.append("%scompression: %s" % (cspacing,self._info['comptype']))

            if self._info['ndims'] != 0:
                dimstr = [str(d) for d in self._info['dims']]
                dimstr = ",".join(dimstr)
            else:
                dimstr=''

            dt = _image_bitpix2npy[self._info['img_equiv_type']]
            text.append("%sdata type: %s" % (cspacing,dt))
            text.append("%sdims: [%s]" % (cspacing,dimstr))

        text = '\n'.join(text)
        return text

def _get_col_dimstr(tdim, is_string=False):
    """
    not for variable length
    """
    dimstr=''
    if tdim is None:
        dimstr='array[bad TDIM]'
    else:
        if is_string:
            if len(tdim) > 1:
                dimstr = [str(d) for d in tdim[1:]]
        else:
            if len(tdim) > 1 or tdim[0] > 1:
                dimstr = [str(d) for d in tdim]
        if dimstr != '':
            dimstr = ','.join(dimstr)
            dimstr = 'array[%s]' % dimstr

    return dimstr

class TableColumnSubset(object):
    """

    A class representing a subset of the the columns on disk.  When called
    with .read() or [ rows ]  the data are read from disk.

    Useful because subsets can be passed around to functions, or chained
    with a row selection.
    
    This class is returned when using [ ] notation to specify fields in a
    TableHDU class

        fits = fitsio.FITS(fname)
        colsub = fits[ext][field_list]

    returns a TableColumnSubset object.  To read rows:

        data = fits[ext][field_list][row_list] 

        colsub = fits[ext][field_list]
        data = colsub[row_list]
        data = colsub.read(rows=row_list)

    to read all, use .read() with no args or [:]
    """

    def __init__(self, fitshdu, columns):
        """
        Input is the SFile instance and a list of column names.
        """

        self.columns = columns
        if isstring(columns) or isinteger(columns):
            # this is to check if it exists
            self.colnums = [fitshdu._extract_colnum(columns)]

            self.is_scalar=True
            self.columns_list = [columns]
        else:
            # this is to check if it exists
            self.colnums = fitshdu._extract_colnums(columns)

            self.is_scalar=False
            self.columns_list = columns

        self.fitshdu = fitshdu 

    def read(self, **keys):
        """
        Read the data from disk and return as a numpy array
        """

        if self.is_scalar:
            data = self.fitshdu.read_column(self.columns, **keys)
        else:
            c=keys.get('columns',None)
            if c is None:
                keys['columns'] = self.columns
            data = self.fitshdu.read(**keys)

        return data

    def __getitem__(self, arg):
        """
        If columns are sent, then the columns will just get reset and
        we'll return a new object

        If rows are sent, they are read and the result returned.
        """

        # we have to unpack the rows if we are reading a subset
        # of the columns because our slice operator only works
        # on whole rows.  We could allow rows= keyword to
        # be a slice...

        res, isrows, isslice = \
            self.fitshdu._process_args_as_rows_or_columns(arg, unpack=True)
        if isrows:
            # rows was entered: read all current column subset
            return self.read(rows=res)

        # columns was entered.  Return a subset objects
        return TableColumnSubset(self.fitshdu, columns=res)


    def __repr__(self):
        """
        Representation for TableColumnSubset
        """
        spacing = ' '*2
        cspacing = ' '*4

        hdu = self.fitshdu
        info = self.fitshdu._info
        colinfo = info['colinfo']

        text = []
        text.append("%sfile: %s" % (spacing,hdu._filename))
        text.append("%sextension: %d" % (spacing,info['hdunum']-1))
        text.append("%stype: %s" % (spacing,_hdu_type_map[info['hdutype']]))
        text.append('%srows: %d' % (spacing,info['nrows']))
        text.append("%scolumn subset:" %  spacing)

        cspacing = ' '*4
        nspace = 4
        nname = 15
        ntype = 6
        format = cspacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
        pformat = cspacing + "%-" + str(nname) + "s\n %" + str(nspace+nname+ntype) + "s  %s"

        for colnum in self.colnums:
            cinfo = colinfo[colnum]

            if len(cinfo['name']) > nname:
                f = pformat
            else:
                f = format

            dt,isvar = hdu._get_tbl_numpy_dtype(colnum, include_endianness=False)
            if isvar:
                tform = cinfo['tform']
                if dt[0] == 'S':
                    dt = 'S0'
                    dimstr='vstring[%d]' % extract_vararray_max(tform)
                else:
                    dimstr = 'varray[%s]' % extract_vararray_max(tform)
            else:
                dimstr = _get_col_dimstr(cinfo['tdim'])

            s = f % (cinfo['name'],dt,dimstr)
            text.append(s)


        s = "\n".join(text)
        return s











def extract_vararray_max(tform):
    """
    Extract number from PX(number)
    """

    first=tform.find('(')
    last=tform.rfind(')')
    
    if first == -1 or last == -1:
        # no max length specified
        return -1

    maxnum=int(tform[first+1:last])
    return maxnum

def check_extver(extver):
    if extver is None:
        return 0
    extver=int(extver)
    if extver <= 0:
        raise ValueError("extver must be > 0")
    return extver

def extract_filename(filename):
    filename=filename.strip()
    if filename[0] == "!":
        filename=filename[1:]
    filename = os.path.expandvars(filename)
    filename = os.path.expanduser(filename)
    return filename

def tdim2shape(tdim, name, is_string=False):
    shape=None
    if tdim is None:
        raise ValueError("field '%s' has malformed TDIM" % name)

    if len(tdim) > 1 or tdim[0] > 1:
        if is_string:
            shape = list( reversed(tdim[1:]) )
        else:
            shape = list( reversed(tdim) )

        if len(shape) == 1:
            shape = shape[0]
        else:
            shape = tuple(shape)

    return shape

def array2tabledef(data, table_type='binary'):
    """
    Similar to descr2tabledef but if there are object columns a type
    and max length will be extracted and used for the tabledef
    """
    is_ascii = (table_type=='ascii')

    if data.dtype.fields is None:
        raise ValueError("data must have fields")
    names=[]
    formats=[]
    dims=[]

    descr=data.dtype.descr
    for d in descr:
        # these have the form '<f4' or '|S25', etc.  Extract the pure type
        npy_dtype = d[1][1:]
        if is_ascii:
            if npy_dtype in ['u1','i1']:
                raise ValueError("1-byte integers are not supported for ascii tables: '%s'" % npy_dtype)
            if npy_dtype in ['u2']:
                raise ValueError("unsigned 2-byte integers are not supported for ascii tables: '%s'" % npy_dtype)

        if npy_dtype[0] == 'O':
            # this will be a variable length column 1Pt(len) where t is the
            # type and len is max length.  Each element must be convertible to
            # the same type as the first
            name=d[0]
            form, dim = npy_obj2fits(data,name)
        elif npy_dtype[0] == "V":
            continue
        else:
            name, form, dim = npy2fits(d,table_type=table_type)

        if name == '':
            raise ValueError("field name is an empty string")

        """
        if is_ascii:
            if dim is not None:
                raise ValueError("array columns are not supported for ascii tables")
        """
        names.append(name)
        formats.append(form)
        dims.append(dim)

    return names, formats, dims

def collection2tabledef(data, names=None, table_type='binary'):
    if isinstance(data,dict):
        if names is None:
            names = list(data.keys())
        isdict=True
    elif isinstance(data,list):
        if names is None:
            raise ValueError("For list of array, send names=")
        isdict=False
    else:
        raise ValueError("expected a dict")

    is_ascii = (table_type=='ascii')
    formats=[]
    dims=[]

    for i,name in enumerate(names):

        if isdict:
            this_data=data[name]
        else:
            this_data=data[i]

        dt=this_data.dtype.descr[0]
        dname=dt[1][1:]

        if is_ascii:
            if dname in ['u1','i1']:
                raise ValueError("1-byte integers are not supported for ascii tables: '%s'" % dname)
            if dname in ['u2']:
                raise ValueError("unsigned 2-byte integers are not supported for ascii tables: '%s'" % dname)

        if dname[0] == 'O':
            # this will be a variable length column 1Pt(len) where t is the
            # type and len is max length.  Each element must be convertible to
            # the same type as the first
            form, dim = npy_obj2fits(this_data)
        else:
            send_dt=dt
            if len(this_data.shape) > 1:
                send_dt=list(dt) + [this_data.shape[1:]]
            _, form, dim = npy2fits(send_dt,table_type=table_type)

        formats.append(form)
        dims.append(dim)

    return names, formats, dims


def descr2tabledef(descr, table_type='binary'):
    """
    Create a FITS table def from the input numpy descriptor.

    parameters
    ----------
    descr: list
        A numpy recarray type descriptor  array.dtype.descr

    returns
    -------
    names, formats, dims: tuple of lists
        These are the ttyp, tform and tdim header entries
        for each field.  dim entries may be None
    """
    names=[]
    formats=[]
    dims=[]

    for d in descr:

        """
        npy_dtype = d[1][1:]
        if is_ascii and npy_dtype in ['u1','i1']:
            raise ValueError("1-byte integers are not supported for ascii tables")
        """

        name, form, dim = npy2fits(d,table_type=table_type)

        if name == '':
            raise ValueError("field name is an empty string")

        """
        if is_ascii:
            if dim is not None:
                raise ValueError("array columns are not supported for ascii tables")
        """

        names.append(name)
        formats.append(form)
        dims.append(dim)

    return names, formats, dims

def npy_obj2fits(data, name=None):
    # this will be a variable length column 1Pt(len) where t is the
    # type and len is max length.  Each element must be convertible to
    # the same type as the first

    if sys.version_info > (3,0,0):
        stype=bytes
    else:
        stype=str

    if name is None:
        d = data.dtype.descr
        first=data[0]
    else:
        d = data[name].dtype.descr
        first = data[name][0]

    # note numpy._string is an instance of str in python2, bytes
    # in python3
    if isinstance(first, stype):
        fits_dtype = _table_npy2fits_form['S']
    else:
        arr0 = numpy.array(first,copy=False)
        dtype0 = arr0.dtype
        npy_dtype = dtype0.descr[0][1][1:]
        if npy_dtype[0] == 'S':
            raise ValueError("Field '%s' is an arrays of strings, this is "
                             "not allowed in variable length columns" % name)
        if npy_dtype not in _table_npy2fits_form:
            raise ValueError("Field '%s' has unsupported type '%s'" % (name,npy_dtype))
        fits_dtype = _table_npy2fits_form[npy_dtype]

    # Q uses 64-bit addressing, should try at some point but the cfitsio manual
    # says it is experimental
    #form = '1Q%s' % fits_dtype
    form = '1P%s' % fits_dtype
    dim=None

    return form, dim



def npy2fits(d, table_type='binary'):
    """
    d is the full element from the descr
    """
    npy_dtype = d[1][1:]
    if npy_dtype[0] == 'S':
        name, form, dim = npy_string2fits(d,table_type=table_type)
    else:
        name, form, dim = npy_num2fits(d, table_type=table_type)

    return name, form, dim

def npy_num2fits(d, table_type='binary'):
    """
    d is the full element from the descr

    For vector,array columns the form is the total counts
    followed by the code.

    For array columns with dimension greater than 1, the dim is set to
        (dim1, dim2, ...)
    So it is treated like an extra dimension
        
    """

    dim = None

    name = d[0]

    npy_dtype = d[1][1:]
    if npy_dtype[0] == 'S':
        raise ValueError("got S type: use npy_string2fits")

    if npy_dtype not in _table_npy2fits_form:
        raise ValueError("unsupported type '%s'" % npy_dtype)

    if table_type=='binary':
        form = _table_npy2fits_form[npy_dtype]
    else:
        form = _table_npy2fits_form_ascii[npy_dtype]

    # now the dimensions
    if len(d) > 2:
        if table_type == 'ascii':
            raise ValueError("Ascii table columns must be scalar, got %s" % str(d))

        # Note, depending on numpy version, even 1-d can be a tuple
        if isinstance(d[2], tuple):
            count=reduce(lambda x, y: x*y, d[2])
            form = '%d%s' % (count,form)

            if len(d[2]) > 1:
                # this is multi-dimensional array column.  the form
                # should be total elements followed by A
                dim = list(reversed(d[2]))
                dim = [str(e) for e in dim]
                dim = '(' + ','.join(dim)+')'
        else:
            # this is a vector (1d array) column
            count = d[2]
            form = '%d%s' % (count,form)

    return name, form, dim


def npy_string2fits(d,table_type='binary'):
    """
    d is the full element from the descr

    form for strings is the total number of bytes followed by A.  Thus
    for vector or array columns it is the size of the string times the
    total number of elements in the array.

    Then the dim is set to
        (sizeofeachstring, dim1, dim2, ...)
    So it is treated like an extra dimension
        
    """

    dim = None

    name = d[0]

    npy_dtype = d[1][1:]
    if npy_dtype[0] != 'S':
        raise ValueError("expected S type")

    # get the size of each string
    string_size_str = npy_dtype[1:]
    string_size = int(string_size_str)

    # now the dimensions
    if len(d) == 2:
        if table_type == 'ascii':
            form = 'A'+string_size_str
        else:
            form = string_size_str+'A'
    else:
        if table_type == 'ascii':
            raise ValueError("Ascii table columns must be scalar, got %s" % str(d))
        if isinstance(d[2], tuple):
            # this is an array column.  the form
            # should be total elements followed by A
            #count = 1
            #count = [count*el for el in d[2]]
            count=reduce(lambda x, y: x*y, d[2])
            count = string_size*count
            form = '%dA' % count

            # will have to do tests to see if this is the right order
            dim = list(reversed(d[2]))
            #dim = d[2]
            dim = [string_size_str] + [str(e) for e in dim]
            dim = '(' + ','.join(dim)+')'
        else:
            # this is a vector (1d array) column
            count = string_size*d[2]
            form = '%dA' % count

            # will have to do tests to see if this is the right order
            dim = [string_size_str, str(d[2])]
            dim = '(' + ','.join(dim)+')'

    return name, form, dim

class FITSHDR(object):
    """
    A class representing a FITS header.

    parameters
    ----------
    record_list: optional
        A list of dicts, or dict, or another FITSHDR
    convert: bool, optional
        If True, convert strings.  E.g. '3' gets
        converted to 3 and "'hello'" gets converted
        to 'hello' and 'T'/'F' to True/False. Default
        is False.

        If the input is a card string, convert is implied True

    examples:

        hdr=FITSHDR()

        # set a simple value
        hdr['blah'] = 35

        # set from a dict to include a comment.
        rec={'name':'fromdict', 'value':3, 'comment':'my comment'}
        hdr.add_record(rec)

        # can do the same with a full FITSRecord
        rec=FITSRecord( {'name':'temp', 'value':35, 'comment':'temp in C'} )
        hdr.add_record(rec)

        # in the above, the record is replaced if one with the same name
        # exists, except for COMMENT and HISTORY, which can exist as 
        # duplicates

        # print the header
        print(hdr)

        # print a single record
        print(hdr['fromdict'])


        # can also set from a card
        hdr.add_record('test    =                   77')
        # using a FITSRecord object (internally uses FITSCard)
        card=FITSRecord('test    =                   77')
        hdr.add_record(card)

        # can also construct with a record list
        recs=[{'name':'test', 'value':35, 'comment':'a comment'},
              {'name':'blah', 'value':'some string'}]
        hdr=FITSHDR(recs)

        # if you have no comments, you can construct with a simple dict
        recs={'day':'saturday',
              'telescope':'blanco'}
        hdr=FITSHDR(recs)

    """
    def __init__(self, record_list=None, convert=False):

        self._record_list = []
        self._record_map = {}
        self._index_map={}

        if isinstance(record_list,FITSHDR):
            for r in record_list.records():
                self.add_record(r, convert=convert)
        elif isinstance(record_list, dict):
            for k in record_list:
                r = {'name':k, 'value':record_list[k]}
                self.add_record(r, convert=convert)
        elif isinstance(record_list, list):
            for r in record_list:
                self.add_record(r, convert=convert)
        elif record_list is not None:
                raise ValueError("expected a dict or list of dicts or FITSHDR")


    def add_record(self, record_in, convert=False):
        """
        Add a new record.  Strip quotes from around strings.

        This will over-write if the key already exists, except
        for COMMENT and HISTORY fields

        parameters
        -----------
        record:
            The record, either a dict or a header card string
            or a FITSRecord or FITSCard
        convert: bool, optional
            If True, convert strings.  E.g. '3' gets
            converted to 3 and "'hello'" gets converted
            to 'hello' and 'T'/'F' to True/False. Default
            is False.

            If the input is a card string, convert is implied True
        """
        record = FITSRecord(record_in, convert=convert)

        # only append when this name already exists if it is
        # a comment or history field, otherwise simply over-write
        key=record['name'].upper()

        key_exists = key in self._record_map

        if not key_exists or key == 'COMMENT' or key == 'HISTORY':
            # append new record
            self._record_list.append(record)
            index=len(self._record_list)-1
            self._index_map[key] = index
        else:
            # over-write existing
            index = self._index_map[key]
            self._record_list[index] = record

        self._record_map[key] = record

    def _add_to_map(self, record):
        key=record['name'].upper()
        self._record_map[key] = record

    def get_comment(self, item):
        """
        Get the comment for the requested entry
        """
        key=item.upper()
        if key not in self._record_map:
            raise ValueError("unknown record: %s" % key)
        if 'comment' not in self._record_map[key]:
            return None
        else:
            return self._record_map[key]['comment']

    def records(self):
        """
        Return the list of full records as a list of dictionaries.
        """
        return self._record_list

    def keys(self):
        """
        Return a copy of the current key list.
        """
        return [e['name'] for e in self._record_list]

    def delete(self, name):
        """
        Delete the specified entry if it exists.
        """
        if isinstance(name, (list,tuple)):
            for xx in name:
                self.delete(xx)
        else:
            if name in self._record_map:
                del self._record_map[name]
                self._record_list = [r for r in self._record_list if r['name'] != name]

    def clean(self, is_table=False):
        """
        Remove reserved keywords from the header.

        These are keywords that the fits writer must write in order
        to maintain consistency between header and data.

        keywords
        --------
        is_table: bool, optional
            Set True if this is a table, so extra keywords will be cleaned
        """

        rmnames = ['SIMPLE','EXTEND','XTENSION','BITPIX','PCOUNT','GCOUNT',
                   'THEAP',
                   'EXTNAME',
                   'BLANK',
                   'ZQUANTIZ','ZDITHER0','ZIMAGE','ZCMPTYPE',
                   'ZSIMPLE','ZTENSION','ZPCOUNT','ZGCOUNT',
                   'ZBITPIX','ZEXTEND',
                   #'FZTILELN','FZALGOR',
                   'CHECKSUM','DATASUM']

        if is_table:
            # these are not allowed in tables
            rmnames += [
                'BUNIT','BSCALE','BZERO',
            ]

        self.delete(rmnames)

        r = self._record_map.get('NAXIS',None)
        if r is not None:
            naxis = int(r['value'])
            self.delete('NAXIS')

            rmnames = ['NAXIS%d' % i for i in xrange(1,naxis+1)]
            self.delete(rmnames)

        r = self._record_map.get('ZNAXIS',None)
        self.delete('ZNAXIS')
        if r is not None:

            znaxis = int(r['value'])

            rmnames = ['ZNAXIS%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZTILE%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZNAME%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZVAL%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)

        
        r = self._record_map.get('TFIELDS',None)
        if r is not None:
            tfields = int(r['value'])
            self.delete('TFIELDS')

            if tfields > 0:

                nbase = ['TFORM','TTYPE','TDIM','TUNIT','TSCAL','TZERO',
                         'TNULL','TDISP','TDMIN','TDMAX','TDESC','TROTA',
                         'TRPIX','TRVAL','TDELT','TCUNI',
                         #'FZALG'
                        ]
                for i in xrange(1,tfields+1):
                    names=['%s%d' % (n,i) for n in nbase]
                    self.delete(names)
            

    def get(self, item, default_value=None):
        """
        Get the requested header entry by keyword name
        """
        key=item.upper()
        if key not in self._record_map:
            return default_value

        return self._record_map[key]['value']

    def __len__(self):
        return len(self._record_list)

    def __contains__(self, item):
        if isinstance(item, FITSRecord):
            name=item['name']
        elif isinstance(item, dict):
            name=item.get('name',None)
            if name is None:
                raise ValueError("dict record must have 'name' field")
        else:
            name=item
        
        name=name.upper()
        return name in self._record_map

    def __setitem__(self, item, value):
        if isinstance(value, (dict,FITSRecord)):
            if item.upper() != value['name'].upper():
                raise ValueError("when setting using a FITSRecord, the "
                                 "name field must match")
            rec=value
        else:
            rec = {'name':item, 'value':value}

        self.add_record(rec)

    def __getitem__(self, item):
        key=item.upper()
        if key not in self._record_map:
            raise ValueError("unknown record: %s" % key)
        return self._record_map[key]['value']

    def __iter__(self):
        self._current=0
        return self

    def next(self):
        """
        for iteration over the header entries
        """
        if self._current < len(self._record_list):
            rec=self._record_list[self._current]
            key=rec['name']
            self._current += 1
            return key
        else:
            raise StopIteration
    __next__=next

    def _record2card(self, record):
        """
        when we add new records they don't have a card,
        this sort of fakes it up similar to what cfitsio
        does, just for display purposes.  e.g.

            DBL     =            23.299843
            LNG     =              3423432
            KEYSNC  = 'hello   '
            KEYSC   = 'hello   '           / a comment for string
            KEYDC   =     3.14159265358979 / a comment for pi
            KEYLC   =            323423432 / a comment for long
        
        basically, 
            - 8 chars, left aligned, for the keyword name
            - a space
            - 20 chars for value, left aligned for strings, right aligned for
              numbers
            - if there is a comment, one space followed by / then another space
              then the comment out to 80 chars

        """
        name = record['name']
        value = record['value']


        if name == 'COMMENT':
            card = 'COMMENT   %s' % value
        elif name=='HISTORY':
            card = 'HISTORY   %s' % value
        else:
            card = '%-8s= ' % name[0:8]
            # these may be string representations of data, or actual strings
            if isstring(value):
                value = str(value)
                if len(value) > 0:
                    if value[0] != "'":
                        # this is a string representing a string header field
                        # make it look like it will look in the header
                        value = "'" + value + "'"
                        vstr = '%-20s' % value
                    else:
                        vstr = "%20s" % value
                else:
                    vstr="''"
            else:
                vstr = '%20s' % value
                    
            card += vstr

            if 'comment' in record:
                card += ' / %s' % record['comment']

        return card[0:80]

    def __repr__(self):
        rep=['']
        for r in self._record_list:
            if 'card_string' not in r:
                card = self._record2card(r)
            else:
                card = r['card_string']

            rep.append(card)
        return '\n'.join(rep)


class FITSRecord(dict):
    """
    Class to represent a FITS header record

    parameters
    ----------
    record: string or dict
        If a string, it should represent a FITS header card

        If a dict it should have 'name' and 'value' fields.
        Can have a 'comment' field.
        
    examples
    --------

    # from a dict.  Can include a comment
    rec=FITSRecord( {'name':'temp', 'value':35, 'comment':'temperature in C'} )

    # from a card
    card=FITSRecord('test    =                   77 / My comment')

    """
    def __init__(self, record, convert=False):
        self.set_record(record, convert=convert)

    def set_record(self, record, convert=False):
        """
        check the record is valid and convert to a dict

        parameters
        ----------
        record: string
            Dict representing a record or a string representing a FITS header
            card
        convert: bool, optional
            If True, convert strings.  E.g. '3' gets
            converted to 3 and "'hello'" gets converted
            to 'hello' and 'T'/'F' to True/False. Default
            is False.

            If the input is a card string, convert is implied True
        """
        import copy

        if isstring(record):
            card=FITSCard(record)
            self.update(card)

            self.verify()

        else:

            if isinstance(record,FITSRecord):
                self.update(record)
            elif isinstance(record,dict):
                # if the card is present, always construct the record from that
                if 'card_string' in record:
                    self.set_record(record['card_string'])
                else:
                    # we will need to verify it
                    self.update(record)
            else:
                raise ValueError("record must be a string card or "
                                 "dictionary or FITSRecord")

            self.verify()

            if convert:
                self['value_orig'] = copy.copy(self['value'])
                if isstring(self['value']):
                    self['value'] = self._convert_value(self['value_orig'])

    def verify(self):
        """
        make sure name,value exist
        """
        if 'name' not in self:
            raise ValueError("each record must have a 'name' field")
        if 'value' not in self:
            raise ValueError("each record must have a 'value' field")

    def _convert_value(self, value_orig):
        """
        things like 6 and 1.25 are converted with ast.literal_value

        Things like 'hello' are stripped of quotes
        """
        import ast
        try:
            value = ast.literal_eval(value_orig)
        except:
            value = self._convert_quoted_string(value_orig)
        return value

    def _convert_quoted_string(self, value):
        """
        Possibly remove quotes around strings.  Deal with bool
        """
        # Strip extra quotes from strings if needed
        if value.startswith("'") and value.endswith("'"):
            val = value[1:-1]
        elif value=='T':
            val=True
        elif value=='F':
            val=False
        else:
            val=value

        return val

TYP_STRUC_KEY=10
TYP_CMPRS_KEY=  20
TYP_SCAL_KEY =  30
TYP_NULL_KEY =  40
TYP_DIM_KEY  =  50
TYP_RANG_KEY =  60
TYP_UNIT_KEY =  70
TYP_DISP_KEY =  80
TYP_HDUID_KEY=  90
TYP_CKSUM_KEY= 100
TYP_WCS_KEY  = 110
TYP_REFSYS_KEY= 120
TYP_COMM_KEY  = 130
TYP_CONT_KEY  = 140
TYP_USER_KEY  = 150

class FITSCard(FITSRecord):
    """
    class to represent ordinary FITS cards.

    CONTINUE not supported

    examples
    --------

    # from a card
    card=FITSRecord('test    =                   77 / My comment')
    """
    def __init__(self, card_string):
        self.set_card(card_string)

    def set_card(self, card_string):
        self['card_string']=card_string

        self._check_equals()

        self._check_type()
        self._check_len()

        front=card_string[0:7]
        if (not self.has_equals() or front=='COMMENT' or front=='HISTORY'):

            if front=='CONTINU':
                raise ValueError("CONTINUE not supported")

            if front=='HISTORY':
                self._set_as_history()
            else:
                # note anything without an = and not history is
                # treated as comment; this is built into cfitsio
                # as well
                self._set_as_comment()
        else:
            self._set_as_key()

    def has_equals(self):
        """
        True if = is in position 8
        """
        return self._has_equals

    def _check_equals(self):
        """
        check for = in position 8, set attribute _has_equals
        """
        card_string=self['card_string']
        if len(card_string) < 9:
            self._has_equals=False
        elif card_string[8]=='=':
            self._has_equals=True
        else:
            self._has_equals=False

    def _set_as_key(self):
        card_string=self['card_string']
        res=_fitsio_wrap.parse_card(card_string)
        keyclass, name, value, dtype, comment=res

        if keyclass==140:
            raise ValueError("bad card '%s'.  CONTINUE not "
                             "supported" % card_string)

        self['class'] = keyclass
        self['name'] = name
        self['value_orig'] = value
        self['value'] = self._convert_value(value)
        self['dtype'] = dtype
        self['comment'] = comment

    def _set_as_comment(self):
        comment=self._extract_comm_or_hist_value()

        self['class'] = TYP_COMM_KEY
        self['name']  = 'COMMENT'
        self['value'] =  comment

    def _set_as_history(self):
        history=self._extract_comm_or_hist_value()

        self['class'] = TYP_COMM_KEY
        self['name']  = 'HISTORY'
        self['value'] =  history

    def _extract_comm_or_hist_value(self):
        card_string=self['card_string']
        if self._has_equals:
            if len(card_string) >= 9:
                value=card_string[9:]
            else:
                value=''
        else:
            if len(card_string) >= 8:
                #value=card_string[7:]
                value=card_string[8:]
            else:
                value=''
        return value

    def _check_type(self):
        card_string=self['card_string']
        if not isstring(card_string):
            raise TypeError("card must be a string, got type %s" % type(card_string))

    def _check_len(self):
        ln=len(self['card_string'])
        if ln > 80:
            mess="len(card) is %d.  cards must have length < 80"
            raise ValueError(mess)

def get_tile_dims(tile_dims, imshape):
    """
    Just make sure the tile dims has the appropriate number of dimensions
    """

    if tile_dims is None:
        td=None
    else:
        td = numpy.array(tile_dims, dtype='i8')
        nd=len(imshape)
        if td.size != nd:
            msg="expected tile_dims to have %d dims, got %d" % (td.size,nd)
            raise ValueError(msg)

    return td

def get_compress_type(compress):
    if compress is not None:
        compress = str(compress).upper()
    if compress not in _compress_map:
        raise ValueError("compress must be one of %s" % list(_compress_map.keys()))
    return _compress_map[compress]

def check_comptype_img(comptype, dtype_str):

    if comptype == NOCOMPRESS:
        return

    if dtype_str == 'i8':
        # no i8 allowed for tile-compressed images
        raise ValueError("8-byte integers not supported when  using tile compression")

    if comptype == PLIO_1:
        # no unsigned for plio
        if dtype_str[0] == 'u':
            raise ValueError("unsigned integers currently not "
                             "allowed when writing using PLIO "
                             "tile compression")

def isstring(arg):
    return isinstance(arg, _stypes)

def isinteger(arg):
    return isinstance(arg, _itypes)

def fields_are_object(arr):
    isobj=numpy.zeros(len(arr.dtype.names),dtype=numpy.bool)
    for i,name in enumerate(arr.dtype.names):
        if is_object(arr[name]):
            isobj[i] = True
    return isobj
def is_object(arr):
    if arr.dtype.descr[0][1][1] == 'O':
        return True
    else:
        return False

def array_to_native_c(array_in, inplace=False):
    # copy only made if not C order
    arr=numpy.array(array_in, order='C', copy=False)
    return array_to_native(arr, inplace=inplace)

 
def array_to_native(array, inplace=False):
    if numpy.little_endian:
        machine_little=True
    else:
        machine_little=False

    data_little=False
    if array.dtype.names is None:

        if array.dtype.base.byteorder=='|':
            # strings and 1 byte integers
            return array

        data_little = is_little_endian(array)
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if is_little_endian(array[fname]):
                data_little=True
                break

    if ( (machine_little and not data_little) 
            or (not machine_little and data_little) ):
        output = array.byteswap(inplace)
    else:
        output = array

    return output



def is_little_endian(array):
    """
    Return True if array is little endian, False otherwise. 

    Parameters
    ----------
    array: numpy array
        A numerical python array.

    Returns
    -------
    Truth value:
        True for little-endian

    Notes
    -----
    Strings are neither big or little endian.  The input must be a simple numpy
    array, not an array with fields.

    """

    if numpy.little_endian:
        machine_little=True
    else:
        machine_little=False

    byteorder = array.dtype.base.byteorder
    return (byteorder == '<') or (machine_little and byteorder == '=')


def _extract_table_type(type):
    """
    Get the numerical table type
    """
    if isinstance(type,str):
        type=type.lower()
        if type[0:7] == 'binary':
            table_type = BINARY_TBL
        elif type[0:6] == 'ascii':
            table_type = ASCII_TBL
        else:
            raise ValueError("table type string should begin with 'binary' or 'ascii' (case insensitive)")
    else:
        type=int(type)
        if type not in [BINARY_TBL,ASCII_TBL]:
            raise ValueError("table type num should be BINARY_TBL (%d) or ASCII_TBL (%d)" % (BINARY_TBL,ASCII_TBL))
        table_type=type

    return table_type


def _names_to_lower_if_recarray(data):
    if data.dtype.names is not None:
        data.dtype.names = [n.lower() for n in data.dtype.names]
def _names_to_upper_if_recarray(data):
    if data.dtype.names is not None:
        data.dtype.names = [n.upper() for n in data.dtype.names]

def _convert_full_start_to_offset(dims, start):
    # convert to scalar offset
    # note we use the on-disk data type to get itemsize
    ndim=len(dims)

    # convert sequence to pixel start
    if len(start) != ndim:
        m="start has len %d, which does not match requested dims %d"
        raise ValueError(m % (len(start),ndim))

    # this is really strides / itemsize
    strides=[1]
    for i in xrange(1,ndim):
        strides.append( strides[i-1] * dims[ndim-i] )

    strides.reverse()
    s=start
    start_index = sum( [s[i]*strides[i] for i in xrange(ndim)] )

    return start_index


# this doesn't work
#GZIP_2 = 22

_compress_map={None:NOCOMPRESS,
               'RICE': RICE_1,
               'RICE_1': RICE_1,
               'GZIP': GZIP_1,
               'GZIP_1': GZIP_1,
               'PLIO': PLIO_1,
               'PLIO_1': PLIO_1,
               'HCOMPRESS': HCOMPRESS_1,
               'HCOMPRESS_1': HCOMPRESS_1,
               NOCOMPRESS:None,
               RICE_1:'RICE_1',
               GZIP_1:'GZIP_1',
               PLIO_1:'PLIO_1',
               HCOMPRESS_1:'HCOMPRESS_1'}

_modeprint_map = {'r':'READONLY','rw':'READWRITE', 0:'READONLY',1:'READWRITE'}
_char_modemap = {'r':'r','rw':'rw', 
                 READONLY:'r',READWRITE:'rw'}
_int_modemap = {'r':READONLY,'rw':READWRITE, READONLY:READONLY, READWRITE:READWRITE}
_hdu_type_map = {IMAGE_HDU:'IMAGE_HDU',
                 ASCII_TBL:'ASCII_TBL',
                 BINARY_TBL:'BINARY_TBL',
                 'IMAGE_HDU':IMAGE_HDU,
                 'ASCII_TBL':ASCII_TBL,
                 'BINARY_TBL':BINARY_TBL}

# no support yet for complex
_table_fits2npy = {#1: 'b1',
                   11: 'u1',
                   12: 'i1',
                   14: 'b1', # logical. Note pyfits uses this for i1, cfitsio casts to char*
                   16: 'S',
                   20: 'u2',
                   21: 'i2',
                   30: 'u4', # 30=TUINT
                   31: 'i4', # 31=TINT
                   40: 'u4', # 40=TULONG
                   41: 'i4', # 41=TLONG
                   42: 'f4',
                   81: 'i8',
                   82: 'f8',
                   83: 'c8',   # TCOMPLEX
                   163: 'c16'} # TDBLCOMPLEX 

# cfitsio returns only types f8, i4 and strings for column types. in order to
# avoid data loss, we always use i8 for integer types
_table_fits2npy_ascii = {16: 'S',
                         31: 'i8', # listed as TINT, reading as i8
                         41: 'i8', # listed as TLONG, reading as i8
                         81: 'i8',
                         21: 'i4', # listed as TSHORT, reading as i4
                         42: 'f8', # listed as TFLOAT, reading as f8
                         82: 'f8'}


# for TFORM
_table_npy2fits_form = {'b1':'L',
                        'u1':'B',
                        'i1':'S', # gets converted to unsigned
                        'S' :'A',
                        'u2':'U', # gets converted to signed
                        'i2':'I',
                        'u4':'V', # gets converted to signed
                        'i4':'J',
                        'i8':'K',
                        'f4':'E',
                        'f8':'D',
                        'c8':'C',
                        'c16':'M'}

_table_npy2fits_form_ascii = {'S' :'A1',       # Need to add max here
                              'i2':'I7',      # I
                              'i4':'I12',     # ??
                              #'i8':'I21',     # K # i8 aren't supported
                              #'f4':'E15.7',   # F
                              'f4':'E26.17',   # F We must write as f8 since we can only read as f8
                              'f8':'E26.17'}  # D 25.16 looks right, but this is recommended
  
# from mrdfits; note G gets turned into E
#      types=  ['A',   'I',   'L',   'B',   'F',    'D',      'C',     'M',     'K']
#      formats=['A1',  'I6',  'I10', 'I4',  'G15.9','G23.17', 'G15.9', 'G23.17','I20']



# remember, you should be using the equivalent image type for this
_image_bitpix2npy = {8: 'u1',
                     10: 'i1',
                     16: 'i2',
                     20: 'u2',
                     32: 'i4',
                     40: 'u4',
                     64: 'i8',
                     -32: 'f4',
                     -64: 'f8'}

# for header keywords
_ftypes = (float,numpy.float32,numpy.float64)

if sys.version_info > (3,0,0):
    _itypes=(int,)
    _stypes = (str,)
else:
    _itypes=(int,long)
    _stypes = (basestring,unicode,)

_itypes += (numpy.uint8,numpy.int8,
            numpy.uint16,numpy.int16,
            numpy.uint32,numpy.int32,
            numpy.uint64,numpy.int64)

_stypes += (numpy.string_,)

