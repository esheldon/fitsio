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
import os
import numpy

from . import _fitsio_wrap
from .util import IS_PY3, mks, array_to_native, isstring
from .header import FITSHDR
from .hdu import (
    ANY_HDU, IMAGE_HDU, BINARY_TBL, ASCII_TBL,
    ImageHDU, AsciiTableHDU, TableHDU,
    _table_npy2fits_form, _npy2fits, _hdu_type_map)

# for python3 compat
if IS_PY3:
    xrange = range


READONLY = 0
READWRITE = 1

NOCOMPRESS = 0
RICE_1 = 11
GZIP_1 = 21
GZIP_2 = 22
PLIO_1 = 31
HCOMPRESS_1 = 41

NO_DITHER = -1
SUBTRACTIVE_DITHER_1 = 1
SUBTRACTIVE_DITHER_2 = 2

# defaults follow fpack
DEFAULT_QLEVEL = 4.0
DEFAULT_QMETHOD = 'SUBTRACTIVE_DITHER_1'
DEFAULT_HCOMP_SCALE = 0.0


def read(filename, ext=None, extver=None, columns=None, rows=None,
         header=False, case_sensitive=False, upper=False, lower=False,
         vstorage='fixed', verbose=False, trim_strings=False, **keys):
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
        If True, force all columns names to lower case in output. Default is
        False.
    upper: bool, optional
        If True, force all columns names to upper case in output. Default is
        False.
    vstorage: string, optional
        Set the default method to store variable length columns.  Can be
        'fixed' or 'object'.  See docs on fitsio.FITS for details. Default is
        'fixed'.
    trim_strings: bool, optional
        If True, trim trailing spaces from strings. Will over-ride the
        trim_strings= keyword from constructor.
    verbose: bool, optional
        If True, print more info when doing various FITS operations.
    """

    if keys:
        import warnings
        warnings.warn(
            "The keyword arguments '%s' are being ignored! This warning "
            "will be an error in a future version of `fitsio`!" % keys,
            DeprecationWarning, stacklevel=2)

    kwargs = {
        'lower': lower,
        'upper': upper,
        'vstorage': vstorage,
        'case_sensitive': case_sensitive,
        'verbose': verbose,
        'trim_strings': trim_strings
    }

    read_kwargs = {}
    if columns is not None:
        read_kwargs['columns'] = columns
    if rows is not None:
        read_kwargs['rows'] = rows

    with FITS(filename, **kwargs) as fits:

        if ext is None:
            for i in xrange(len(fits)):
                if fits[i].has_data():
                    ext = i
                    break
            if ext is None:
                raise IOError("No extensions have data")

        item = _make_item(ext, extver=extver)

        data = fits[item].read(**read_kwargs)
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

    if keys:
        import warnings
        warnings.warn(
            "The keyword arguments '%s' are being ignored! This warning "
            "will be an error in a future version of `fitsio`!" % keys,
            DeprecationWarning, stacklevel=2)

    filename = extract_filename(filename)

    dont_create = 0
    try:
        hdunum = ext+1
    except TypeError:
        hdunum = None

    _fits = _fitsio_wrap.FITS(filename, READONLY, dont_create)

    if hdunum is None:
        extname = mks(ext)
        if extver is None:
            extver_num = 0
        else:
            extver_num = extver

        if not case_sensitive:
            # the builtin movnam_hdu is not case sensitive
            hdunum = _fits.movnam_hdu(ANY_HDU, extname, extver_num)
        else:
            # for case sensitivity we'll need to run through
            # all the hdus
            found = False
            current_ext = 0
            while True:
                hdunum = current_ext+1
                try:
                    hdu_type = _fits.movabs_hdu(hdunum)  # noqa - not used
                    name, vers = _fits.get_hdu_name_version(hdunum)
                    if name == extname:
                        if extver is None:
                            # take the first match
                            found = True
                            break
                        else:
                            if extver_num == vers:
                                found = True
                                break
                except OSError:
                    break

                current_ext += 1

            if not found:
                raise IOError(
                    'hdu not found: %s (extver %s)' % (extname, extver))

    return FITSHDR(_fits.read_header(hdunum))


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
        lines = fobj.readlines()

    lines = [l.strip() for l in lines if l[0:3] != 'END']

    # if header is None an empty FITSHDR is created
    hdr = FITSHDR(header)

    for l in lines:
        hdr.add_record(l)

    return hdr


def _make_item(ext, extver=None):
    if extver is not None:
        # e
        item = (ext, extver)
    else:
        item = ext

    return item


def write(filename, data, extname=None, extver=None, header=None,
          clobber=False, ignore_empty=False, units=None, table_type='binary',
          names=None, write_bitcols=False, compress=None, tile_dims=None,
          qlevel=DEFAULT_QLEVEL,
          qmethod=DEFAULT_QMETHOD,
          hcomp_scale=DEFAULT_HCOMP_SCALE,
          hcomp_smooth=False,
          **keys):
    """
    Convenience function to create a new HDU and write the data.

    Under the hood, a FITS object is constructed.  If you want to append rows
    to an existing HDU, or modify data in an HDU, please construct a FITS
    object.

    parameters
    ----------
    filename: string
        A filename.
    data: numpy.ndarray or recarray
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
    header: FITSHDR, list, dict, optional
        A set of header keys to write. The keys are written before the data
        is written to the table, preventing a resizing of the table area.

        Can be one of these:
            - FITSHDR object
            - list of dictionaries containing 'name','value' and optionally
              a 'comment' field; the order is preserved.
            - a dictionary of keyword-value pairs; no comments are written
              in this case, and the order is arbitrary.
        Note required keywords such as NAXIS, XTENSION, etc are cleaed out.
    clobber: bool, optional
        If True, overwrite any existing file. Default is to append
        a new extension on existing files.
    ignore_empty: bool, optional
        Default False.  Unless set to True, only allow
        empty HDUs in the zero extension.

    table-only keywords
    -------------------
    units: list
        A list of strings representing units for each column.
    table_type: string, optional
        Either 'binary' or 'ascii', default 'binary'
        Matching is case-insensitive
    write_bitcols: bool, optional
        Write boolean arrays in the FITS bitcols format, default False
    names: list, optional
        If data is a list of arrays, you must send `names` as a list
        of names or column numbers.

    image-only keywords
    -------------------
    compress: string, optional
        A string representing the compression algorithm for images,
        default None.
        Can be one of
           'RICE'
           'GZIP'
           'GZIP_2'
           'PLIO' (no unsigned or negative integers)
           'HCOMPRESS'
        (case-insensitive) See the cfitsio manual for details.
    tile_dims: tuple of ints, optional
        The size of the tiles used to compress images.
    qlevel: float, optional
        Quantization level for floating point data.  Lower generally result in
        more compression, we recommend one reads the FITS standard or cfitsio
        manual to fully understand the effects of quantization.  None or 0
        means no quantization, and for gzip also implies lossless.  Default is
        4.0 which follows the fpack defaults
    qmethod: string or int
        The quantization method as string or integer.
            'NO_DITHER' or fitsio.NO_DITHER (-1)
               No dithering is performed
            'SUBTRACTIVE_DITHER_1' or fitsio.SUBTRACTIVE_DITHER_1 (1)
                Standard dithering
            'SUBTRACTIVE_DITHER_2' or fitsio.SUBTRACTIVE_DITHER_2 (2)
                Preserves zeros

        Defaults to 'SUBTRACTIVE_DITHER_1' which follows the fpack defaults

    hcomp_scale: float
        Scale value for HCOMPRESS, 0.0 means lossless compression. Default is 0.0
        following the fpack defaults.
    hcomp_smooth: bool
        If True, apply smoothing when decompressing.  Default False
    """
    if keys:
        import warnings
        warnings.warn(
            "The keyword arguments '%s' are being ignored! This warning "
            "will be an error in a future version of `fitsio`!" % keys,
            DeprecationWarning, stacklevel=2)

    kwargs = {
        'clobber': clobber,
        'ignore_empty': ignore_empty
    }
    with FITS(filename, 'rw', **kwargs) as fits:
        fits.write(
            data,
            table_type=table_type,
            units=units,
            extname=extname,
            extver=extver,
            header=header,
            names=names,
            write_bitcols=write_bitcols,

            compress=compress,
            tile_dims=tile_dims,
            qlevel=qlevel,
            qmethod=qmethod,
            hcomp_scale=hcomp_scale,
            hcomp_smooth=hcomp_smooth,
        )


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
    ignore_empty: bool, optional
        Default False.  Unless set to True, only allow
        empty HDUs in the zero extension.
    verbose: bool, optional
        If True, print more info when doing various FITS operations.

    See the docs at https://github.com/esheldon/fitsio
    """
    def __init__(self, filename, mode='r', lower=False, upper=False,
                 trim_strings=False, vstorage='fixed', case_sensitive=False,
                 iter_row_buffer=1, write_bitcols=False, ignore_empty=False,
                 verbose=False, clobber=False, **keys):

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        self.lower = lower
        self.upper = upper
        self.trim_strings = trim_strings
        self.vstorage = vstorage
        self.case_sensitive = case_sensitive
        self.iter_row_buffer = iter_row_buffer
        self.write_bitcols = write_bitcols
        filename = extract_filename(filename)
        self._filename = filename

        # self.mode=keys.get('mode','r')
        self.mode = mode
        self.ignore_empty = ignore_empty

        self.verbose = verbose

        if self.mode not in _int_modemap:
            raise IOError("mode should be one of 'r', 'rw', "
                          "READONLY,READWRITE")

        self.charmode = _char_modemap[self.mode]
        self.intmode = _int_modemap[self.mode]

        # Will not test existence when reading, let cfitsio
        # do the test and report an error.  This allows opening
        # urls etc.
        create = 0
        if self.mode in [READWRITE, 'rw']:
            if clobber:
                create = 1
                if filename[0] != '!':
                    filename = '!' + filename
            else:
                if os.path.exists(filename):
                    create = 0
                else:
                    create = 1

        self._did_create = (create == 1)
        self._FITS = _fitsio_wrap.FITS(filename, self.intmode, create)

    def close(self):
        """
        Close the fits file and set relevant metadata to None
        """
        if hasattr(self, '_FITS'):
            if self._FITS is not None:
                self._FITS.close()
                self._FITS = None
        self._filename = None
        self.mode = None
        self.charmode = None
        self.intmode = None
        self.hdu_list = None
        self.hdu_map = None

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
        extname = mks(extname)
        hdu = self._FITS.movnam_hdu(hdutype, extname, extver)
        return hdu-1

    def movnam_hdu(self, extname, hdutype=ANY_HDU, extver=0):
        """
        Move to the indicated HDU by name

        In general, it is not necessary to use this method explicitly.

        returns the one-offset extension number
        """
        extname = mks(extname)
        hdu = self._FITS.movnam_hdu(hdutype, extname, extver)
        return hdu

    def reopen(self):
        """
        close and reopen the fits file with the same mode
        """
        self._FITS.close()
        del self._FITS
        self._FITS = _fitsio_wrap.FITS(self._filename, self.intmode, 0)
        self.update_hdu_list()

    def write(self, data, units=None, extname=None, extver=None,
              compress=None,
              tile_dims=None,
              qlevel=DEFAULT_QLEVEL,
              qmethod=DEFAULT_QMETHOD,
              hcomp_scale=DEFAULT_HCOMP_SCALE,
              hcomp_smooth=False,
              header=None, names=None,
              table_type='binary', write_bitcols=False, **keys):
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
                  a 'comment' field; the order is preserved.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary.
            Note required keywords such as NAXIS, XTENSION, etc are cleaed out.

        image-only keywords
        -------------------
        compress: string, optional
            A string representing the compression algorithm for images,
            default None.
            Can be one of
                'RICE'
                'GZIP'
                'GZIP_2'
                'PLIO' (no unsigned or negative integers)
                'HCOMPRESS'
            (case-insensitive) See the cfitsio manual for details.
        tile_dims: tuple of ints, optional
            The size of the tiles used to compress images.
        qlevel: float, optional
            Quantization level for floating point data.  Lower generally result in
            more compression, we recommend one reads the FITS standard or cfitsio
            manual to fully understand the effects of quantization.  None or 0
            means no quantization, and for gzip also implies lossless.  Default is
            4.0 which follows the fpack defaults
        qmethod: string or int
            The quantization method as string or integer.
                'NO_DITHER' or fitsio.NO_DITHER (-1)
                   No dithering is performed
                'SUBTRACTIVE_DITHER_1' or fitsio.SUBTRACTIVE_DITHER_1 (1)
                    Standard dithering
                'SUBTRACTIVE_DITHER_2' or fitsio.SUBTRACTIVE_DITHER_2 (2)
                    Preserves zeros

            Defaults to 'SUBTRACTIVE_DITHER_1' which follows the fpack defaults

        hcomp_scale: float
            Scale value for HCOMPRESS, 0.0 means lossless compression. Default is 0.0
            following the fpack defaults.
        hcomp_smooth: bool
            If True, apply smoothing when decompressing.  Default False

        table-only keywords
        -------------------
        units: list/dec, optional:
            A list of strings with units for each column.
        table_type: string, optional
            Either 'binary' or 'ascii', default 'binary'
            Matching is case-insensitive
        write_bitcols: bool, optional
            Write boolean arrays in the FITS bitcols format, default False
        names: list, optional
            If data is a list of arrays, you must send `names` as a list
            of names or column numbers.

        restrictions
        ------------
        The File must be opened READWRITE
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        isimage = False
        if data is None:
            isimage = True
        elif isinstance(data, numpy.ndarray):
            if data.dtype.fields == None:  # noqa - probably should be is None
                isimage = True

        if isimage:
            self.write_image(data, extname=extname, extver=extver,
                             compress=compress,
                             tile_dims=tile_dims,
                             qlevel=qlevel,
                             qmethod=qmethod,
                             hcomp_scale=hcomp_scale,
                             hcomp_smooth=hcomp_smooth,
                             header=header)
        else:
            self.write_table(data, units=units,
                             extname=extname, extver=extver, header=header,
                             names=names,
                             table_type=table_type,
                             write_bitcols=write_bitcols)

    def write_image(self, img, extname=None, extver=None,
                    compress=None, tile_dims=None,
                    qlevel=DEFAULT_QLEVEL,
                    qmethod=DEFAULT_QMETHOD,
                    hcomp_scale=DEFAULT_HCOMP_SCALE,
                    hcomp_smooth=False,
                    header=None):
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
            A string representing the compression algorithm for images,
            default None.
            Can be one of
                'RICE'
                'GZIP'
                'GZIP_2'
                'PLIO' (no unsigned or negative integers)
                'HCOMPRESS'
            (case-insensitive) See the cfitsio manual for details.
        tile_dims: tuple of ints, optional
            The size of the tiles used to compress images.
        qlevel: float, optional
            Quantization level for floating point data.  Lower generally result in
            more compression, we recommend one reads the FITS standard or cfitsio
            manual to fully understand the effects of quantization.  None or 0
            means no quantization, and for gzip also implies lossless.  Default is
            4.0 which follows the fpack defaults
        qmethod: string or int
            The quantization method as string or integer.
                'NO_DITHER' or fitsio.NO_DITHER (-1)
                   No dithering is performed
                'SUBTRACTIVE_DITHER_1' or fitsio.SUBTRACTIVE_DITHER_1 (1)
                    Standard dithering
                'SUBTRACTIVE_DITHER_2' or fitsio.SUBTRACTIVE_DITHER_2 (2)
                    Preserves zeros

            Defaults to 'SUBTRACTIVE_DITHER_1' which follows the fpack defaults

        hcomp_scale: float
            Scale value for HCOMPRESS, 0.0 means lossless compression. Default is 0.0
            following the fpack defaults.
        hcomp_smooth: bool
            If True, apply smoothing when decompressing.  Default False

        header: FITSHDR, list, dict, optional
            A set of header keys to write. Can be one of these:
                - FITSHDR object
                - list of dictionaries containing 'name','value' and optionally
                  a 'comment' field; the order is preserved.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary.
            Note required keywords such as NAXIS, XTENSION, etc are cleaed out.


        restrictions
        ------------
        The File must be opened READWRITE
        """

        self.create_image_hdu(
            img,
            header=header,
            extname=extname,
            extver=extver,
            compress=compress,
            tile_dims=tile_dims,
            qlevel=qlevel,
            qmethod=qmethod,
            hcomp_scale=hcomp_scale,
            hcomp_smooth=hcomp_smooth,
        )

        if header is not None:
            self[-1].write_keys(header)
            self[-1]._update_info()

        # if img is not None:
        #    self[-1].write(img)

    def create_image_hdu(self,
                         img=None,
                         dims=None,
                         dtype=None,
                         extname=None,
                         extver=None,
                         compress=None,
                         tile_dims=None,
                         qlevel=DEFAULT_QLEVEL,
                         qmethod=DEFAULT_QMETHOD,
                         hcomp_scale=DEFAULT_HCOMP_SCALE,
                         hcomp_smooth=False,
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
            A string representing the compression algorithm for images,
            default None.
            Can be one of
                'RICE'
                'GZIP'
                'GZIP_2'
                'PLIO' (no unsigned or negative integers)
                'HCOMPRESS'
            (case-insensitive) See the cfitsio manual for details.
        tile_dims: tuple of ints, optional
            The size of the tiles used to compress images.
        qlevel: float, optional
            Quantization level for floating point data.  Lower generally result in
            more compression, we recommend one reads the FITS standard or cfitsio
            manual to fully understand the effects of quantization.  None or 0
            means no quantization, and for gzip also implies lossless.  Default is
            4.0 which follows the fpack defaults.
        qmethod: string or int
            The quantization method as string or integer.
                'NO_DITHER' or fitsio.NO_DITHER (-1)
                   No dithering is performed
                'SUBTRACTIVE_DITHER_1' or fitsio.SUBTRACTIVE_DITHER_1 (1)
                    Standard dithering
                'SUBTRACTIVE_DITHER_2' or fitsio.SUBTRACTIVE_DITHER_2 (2)
                    Preserves zeros

            Defaults to 'SUBTRACTIVE_DITHER_1' which follows the fpack defaults

        hcomp_scale: float
            Scale value for HCOMPRESS, 0.0 means lossless compression. Default is 0.0
            following the fpack defaults.
        hcomp_smooth: bool
            If True, apply smoothing when decompressing.  Default False

        header: FITSHDR, list, dict, optional
            This is only used to determine how many slots to reserve for
            header keywords

        restrictions
        ------------
        The File must be opened READWRITE
        """

        if (img is not None) or (img is None and dims is None):
            from_image = True
        elif dims is not None:
            from_image = False

        if from_image:
            img2send = img
            if img is not None:
                dims = img.shape
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

                if IS_PY3 and img2send.dtype.char == 'U':
                    # for python3, we convert unicode to ascii
                    # this will error if the character is not in ascii
                    img2send = img2send.astype('S', copy=False)

            else:
                self._ensure_empty_image_ok()
                compress = None
                tile_dims = None

            # we get dims from the input image
            dims2send = None
        else:
            # img was None and dims was sent
            if dtype is None:
                raise ValueError("send dtype= with dims=")

            # this must work!
            dtype = numpy.dtype(dtype)
            dtstr = dtype.descr[0][1][1:]
            # use the example image to build the type in C
            img2send = numpy.zeros(1, dtype=dtype)

            # sending an array simplifies access
            dims2send = numpy.array(dims, dtype='i8', ndmin=1)

        if img2send is not None:
            if img2send.dtype.fields is not None:
                raise ValueError(
                    "got record data type, expected regular ndarray")

        if extname is None:
            # will be ignored
            extname = ""
        else:
            if not isstring(extname):
                raise ValueError("extension name must be a string")
            extname = mks(extname)

        if extname is not None and extver is not None:
            extver = check_extver(extver)

        if extver is None:
            # will be ignored
            extver = 0

        comptype = get_compress_type(compress)
        qmethod = get_qmethod(qmethod)

        tile_dims = get_tile_dims(tile_dims, dims)
        if qlevel is None:
            # 0.0 is the sentinel value for "no quantization" in cfitsio
            qlevel = 0.0
        else:
            qlevel = float(qlevel)

        if img2send is not None:
            check_comptype_img(comptype, dtstr)

        if header is not None:
            nkeys = len(header)
        else:
            nkeys = 0

        if hcomp_smooth:
            hcomp_smooth = 1
        else:
            hcomp_smooth = 0

        self._FITS.create_image_hdu(
            img2send,
            nkeys,
            dims=dims2send,
            comptype=comptype,
            tile_dims=tile_dims,

            qlevel=qlevel,
            qmethod=qmethod,

            hcomp_scale=hcomp_scale,
            hcomp_smooth=hcomp_smooth,

            extname=extname,
            extver=extver,
        )

        # don't rebuild the whole list unless this is the first hdu
        # to be created
        self.update_hdu_list(rebuild=False)

    def _ensure_empty_image_ok(self):
        """
        If ignore_empty was not set to True, we only allow empty HDU for first
        HDU and if there is no data there already
        """
        if self.ignore_empty:
            return

        if len(self) > 1:
            raise RuntimeError(
                "Cannot write None image at extension %d" % len(self))
        if 'ndims' in self[0]._info:
            raise RuntimeError("Can only write None images to extension zero, "
                               "which already exists")

    def write_table(self, data, table_type='binary',
                    names=None, formats=None, units=None,
                    extname=None, extver=None, header=None,
                    write_bitcols=False):
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
                  a 'comment' field; the order is preserved.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary.
            Note required keywords such as NAXIS, XTENSION, etc are cleaed out.
        write_bitcols: boolean, optional
            Write boolean arrays in the FITS bitcols format, default False

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
                              table_type=table_type,
                              write_bitcols=write_bitcols)

        if header is not None:
            self[-1].write_keys(header)
            self[-1]._update_info()

        self[-1].write(data, names=names)

    def read_raw(self):
        """
        Reads the raw FITS file contents, returning a Python string.
        """
        return self._FITS.read_raw()

    def create_table_hdu(self, data=None, dtype=None,
                         header=None,
                         names=None, formats=None,
                         units=None, dims=None, extname=None, extver=None,
                         table_type='binary', write_bitcols=False):
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
        write_bitcols: bool, optional
            Write boolean arrays in the FITS bitcols format, default False

        header: FITSHDR, list, dict, optional
            This is only used to determine how many slots to reserve for
            header keywords


        restrictions
        ------------
        The File must be opened READWRITE
        """

        # record this for the TableHDU object
        write_bitcols = self.write_bitcols or write_bitcols

        # can leave as turn
        table_type_int = _extract_table_type(table_type)

        if data is not None:
            if isinstance(data, numpy.ndarray):
                names, formats, dims = array2tabledef(
                    data, table_type=table_type, write_bitcols=write_bitcols)
            elif isinstance(data, (list, dict)):
                names, formats, dims = collection2tabledef(
                    data, names=names, table_type=table_type,
                    write_bitcols=write_bitcols)
            else:
                raise ValueError(
                    "data must be an ndarray with fields or a dict")
        elif dtype is not None:
            dtype = numpy.dtype(dtype)
            names, formats, dims = descr2tabledef(
                dtype.
                descr,
                write_bitcols=write_bitcols,
                table_type=table_type,
            )
        else:
            if names is None or formats is None:
                raise ValueError(
                    "send either dtype=, data=, or names= and formats=")

            if not isinstance(names, list) or not isinstance(formats, list):
                raise ValueError("names and formats should be lists")
            if len(names) != len(formats):
                raise ValueError("names and formats must be same length")

            if dims is not None:
                if not isinstance(dims, list):
                    raise ValueError("dims should be a list")
                if len(dims) != len(names):
                    raise ValueError("names and dims must be same length")

        if units is not None:
            if not isinstance(units, list):
                raise ValueError("units should be a list")
            if len(units) != len(names):
                raise ValueError("names and units must be same length")

        if extname is None:
            # will be ignored
            extname = ""
        else:
            if not isstring(extname):
                raise ValueError("extension name must be a string")
            extname = mks(extname)

        if extname is not None and extver is not None:
            extver = check_extver(extver)
        if extver is None:
            # will be ignored
            extver = 0
        if extname is None:
            # will be ignored
            extname = ""

        if header is not None:
            nkeys = len(header)
        else:
            nkeys = 0

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
        if not hasattr(self, 'hdu_list'):
            rebuild = True

        if rebuild:
            self.hdu_list = []
            self.hdu_map = {}

            # we don't know how many hdus there are, so iterate
            # until we can't open any more
            ext_start = 0
        else:
            # start from last
            ext_start = len(self)

        ext = ext_start
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
        hdu_type = self._FITS.movabs_hdu(ext+1)

        if hdu_type == IMAGE_HDU:
            hdu = ImageHDU(self._FITS, ext)
        elif hdu_type == BINARY_TBL:
            hdu = TableHDU(
                self._FITS, ext,
                lower=self.lower, upper=self.upper,
                trim_strings=self.trim_strings,
                vstorage=self.vstorage, case_sensitive=self.case_sensitive,
                iter_row_buffer=self.iter_row_buffer,
                write_bitcols=self.write_bitcols)
        elif hdu_type == ASCII_TBL:
            hdu = AsciiTableHDU(
                self._FITS, ext,
                lower=self.lower, upper=self.upper,
                trim_strings=self.trim_strings,
                vstorage=self.vstorage, case_sensitive=self.case_sensitive,
                iter_row_buffer=self.iter_row_buffer,
                write_bitcols=self.write_bitcols)
        else:
            mess = ("extension %s is of unknown type %s "
                    "this is probably a bug")
            mess = mess % (ext, hdu_type)
            raise IOError(mess)

        self.hdu_list.append(hdu)
        self.hdu_map[ext] = hdu

        extname = hdu.get_extname()
        if not self.case_sensitive:
            extname = extname.lower()
        if extname != '':
            # this will guarantee we default to *first* version,
            # if version is not requested, using __getitem__
            if extname not in self.hdu_map:
                self.hdu_map[extname] = hdu

            ver = hdu.get_extver()
            if ver > 0:
                key = '%s-%s' % (extname, ver)
                self.hdu_map[key] = hdu

    def __iter__(self):
        """
        begin iteration over HDUs
        """
        if not hasattr(self, 'hdu_list'):
            self.update_hdu_list()
        self._iter_index = 0
        return self

    def next(self):
        """
        Move to the next iteration
        """
        if self._iter_index == len(self.hdu_list):
            raise StopIteration
        hdu = self.hdu_list[self._iter_index]
        self._iter_index += 1
        return hdu

    __next__ = next

    def __len__(self):
        """
        get the number of extensions
        """
        if not hasattr(self, 'hdu_list'):
            self.update_hdu_list()
        return len(self.hdu_list)

    def _extract_item(self, item):
        """
        utility function to extract an "item", meaning
        a extension number,name plus version.
        """
        ver = 0
        if isinstance(item, tuple):
            ver_sent = True
            nitem = len(item)
            if nitem == 1:
                ext = item[0]
            elif nitem == 2:
                ext, ver = item
        else:
            ver_sent = False
            ext = item
        return ext, ver, ver_sent

    def __getitem__(self, item):
        """
        Get an hdu by number, name, and possibly version
        """
        if not hasattr(self, 'hdu_list'):
            if self._did_create:
                # we created the file and haven't written anything yet
                raise ValueError("Requested hdu '%s' not present" % item)

            self.update_hdu_list()

        if len(self) == 0:
            raise ValueError("Requested hdu '%s' not present" % item)

        ext, ver, ver_sent = self._extract_item(item)

        try:
            # if it is an int
            hdu = self.hdu_list[ext]
        except Exception:
            # might be a string
            ext = mks(ext)
            if not self.case_sensitive:
                mess = '(case insensitive)'
                ext = ext.lower()
            else:
                mess = '(case sensitive)'

            if ver > 0:
                key = '%s-%s' % (ext, ver)
                if key not in self.hdu_map:
                    raise IOError("extension not found: %s, "
                                  "version %s %s" % (ext, ver, mess))
                hdu = self.hdu_map[key]
            else:
                if ext not in self.hdu_map:
                    raise IOError("extension not found: %s %s" % (ext, mess))
                hdu = self.hdu_map[ext]

        return hdu

    def __contains__(self, item):
        """
        tell whether specified extension exists, possibly
        with version sent as well
        """
        try:
            hdu = self[item]  # noqa
            return True
        except Exception:
            return False

    def __repr__(self):
        """
        Text representation of some fits file metadata
        """
        spacing = ' '*2
        rep = ['']
        rep.append("%sfile: %s" % (spacing, self._filename))
        rep.append("%smode: %s" % (spacing, _modeprint_map[self.intmode]))

        rep.append('%sextnum %-15s %s' % (spacing, "hdutype", "hduname[v]"))

        if not hasattr(self, 'hdu_list'):
            if not self._did_create:
                # we expect some stuff
                self.update_hdu_list()

                for i, hdu in enumerate(self.hdu_list):
                    t = hdu._info['hdutype']
                    name = hdu.get_extname()
                    if name != '':
                        ver = hdu.get_extver()
                        if ver != 0:
                            name = '%s[%s]' % (name, ver)

                    rep.append(
                        "%s%-6d %-15s %s" % (spacing, i, _hdu_type_map[t], name))

        rep = '\n'.join(rep)
        return rep

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def check_extver(extver):
    if extver is None:
        return 0
    extver = int(extver)
    if extver <= 0:
        raise ValueError("extver must be > 0")
    return extver


def extract_filename(filename):
    filename = mks(filename)
    filename = filename.strip()
    if filename[0] == "!":
        filename = filename[1:]
    filename = os.path.expandvars(filename)
    filename = os.path.expanduser(filename)
    return filename


def array2tabledef(data, table_type='binary', write_bitcols=False):
    """
    Similar to descr2tabledef but if there are object columns a type
    and max length will be extracted and used for the tabledef
    """
    is_ascii = (table_type == 'ascii')

    if data.dtype.fields is None:
        raise ValueError("data must have fields")
    names = []
    names_nocase = {}
    formats = []
    dims = []

    descr = data.dtype.descr
    for d in descr:
        # these have the form '<f4' or '|S25', etc.  Extract the pure type
        npy_dtype = d[1][1:]
        if is_ascii:
            if npy_dtype in ['u1', 'i1']:
                raise ValueError(
                    "1-byte integers are not supported for "
                    "ascii tables: '%s'" % npy_dtype)
            if npy_dtype in ['u2']:
                raise ValueError(
                    "unsigned 2-byte integers are not supported for "
                    "ascii tables: '%s'" % npy_dtype)

        if npy_dtype[0] == 'O':
            # this will be a variable length column 1Pt(len) where t is the
            # type and len is max length.  Each element must be convertible to
            # the same type as the first
            name = d[0]
            form, dim = npy_obj2fits(data, name)
        elif npy_dtype[0] == "V":
            continue
        else:
            name, form, dim = _npy2fits(
                d, table_type=table_type, write_bitcols=write_bitcols)

        if name == '':
            raise ValueError("field name is an empty string")

        """
        if is_ascii:
            if dim is not None:
                raise ValueError("array columns are not supported for "
                                 "ascii tables")
        """
        name_nocase = name.upper()
        if name_nocase in names_nocase:
            raise ValueError(
                "duplicate column name found: '%s'.  Note "
                "FITS column names are not case sensitive" % name_nocase)

        names.append(name)
        names_nocase[name_nocase] = name_nocase

        formats.append(form)
        dims.append(dim)

    return names, formats, dims


def collection2tabledef(
        data, names=None, table_type='binary', write_bitcols=False):
    if isinstance(data, dict):
        if names is None:
            names = list(data.keys())
        isdict = True
    elif isinstance(data, list):
        if names is None:
            raise ValueError("For list of array, send names=")
        isdict = False
    else:
        raise ValueError("expected a dict")

    is_ascii = (table_type == 'ascii')
    formats = []
    dims = []

    for i, name in enumerate(names):

        if isdict:
            this_data = data[name]
        else:
            this_data = data[i]

        dt = this_data.dtype.descr[0]
        dname = dt[1][1:]

        if is_ascii:
            if dname in ['u1', 'i1']:
                raise ValueError(
                    "1-byte integers are not supported for "
                    "ascii tables: '%s'" % dname)
            if dname in ['u2']:
                raise ValueError(
                    "unsigned 2-byte integers are not supported for "
                    "ascii tables: '%s'" % dname)

        if dname[0] == 'O':
            # this will be a variable length column 1Pt(len) where t is the
            # type and len is max length.  Each element must be convertible to
            # the same type as the first
            form, dim = npy_obj2fits(this_data)
        else:
            send_dt = dt
            if len(this_data.shape) > 1:
                send_dt = list(dt) + [this_data.shape[1:]]
            _, form, dim = _npy2fits(
                send_dt, table_type=table_type, write_bitcols=write_bitcols)

        formats.append(form)
        dims.append(dim)

    return names, formats, dims


def descr2tabledef(descr, table_type='binary', write_bitcols=False):
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
    names = []
    formats = []
    dims = []

    for d in descr:

        """
        npy_dtype = d[1][1:]
        if is_ascii and npy_dtype in ['u1','i1']:
            raise ValueError("1-byte integers are not supported for "
                             "ascii tables")
        """

        if d[1][1] == 'O':
            raise ValueError(
                'cannot automatically declare a var column without '
                'some data to determine max len')

        name, form, dim = _npy2fits(
            d, table_type=table_type, write_bitcols=write_bitcols)

        if name == '':
            raise ValueError("field name is an empty string")

        """
        if is_ascii:
            if dim is not None:
                raise ValueError("array columns are not supported "
                                 "for ascii tables")
        """

        names.append(name)
        formats.append(form)
        dims.append(dim)

    return names, formats, dims


def npy_obj2fits(data, name=None):
    # this will be a variable length column 1Pt(len) where t is the
    # type and len is max length.  Each element must be convertible to
    # the same type as the first

    if name is None:
        d = data.dtype.descr
        first = data[0]
    else:
        d = data[name].dtype.descr  # noqa - not used
        first = data[name][0]

    # note numpy._string is an instance of str in python2, bytes
    # in python3
    if isinstance(first, str) or (IS_PY3 and isinstance(first, bytes)):
        if IS_PY3:
            if isinstance(first, str):
                fits_dtype = _table_npy2fits_form['U']
            else:
                fits_dtype = _table_npy2fits_form['S']
        else:
            fits_dtype = _table_npy2fits_form['S']
    else:
        arr0 = numpy.array(first, copy=False)
        dtype0 = arr0.dtype
        npy_dtype = dtype0.descr[0][1][1:]
        if npy_dtype[0] == 'S' or npy_dtype[0] == 'U':
            raise ValueError("Field '%s' is an arrays of strings, this is "
                             "not allowed in variable length columns" % name)
        if npy_dtype not in _table_npy2fits_form:
            raise ValueError(
                "Field '%s' has unsupported type '%s'" % (name, npy_dtype))
        fits_dtype = _table_npy2fits_form[npy_dtype]

    # Q uses 64-bit addressing, should try at some point but the cfitsio manual
    # says it is experimental
    # form = '1Q%s' % fits_dtype
    form = '1P%s' % fits_dtype
    dim = None

    return form, dim


def get_tile_dims(tile_dims, imshape):
    """
    Just make sure the tile dims has the appropriate number of dimensions
    """

    if tile_dims is None:
        td = None
    else:
        td = numpy.array(tile_dims, dtype='i8')
        nd = len(imshape)
        if td.size != nd:
            msg = "expected tile_dims to have %d dims, got %d" % (td.size, nd)
            raise ValueError(msg)

    return td


def get_compress_type(compress):
    if compress is not None:
        compress = str(compress).upper()
    if compress not in _compress_map:
        raise ValueError(
            "compress must be one of %s" % list(_compress_map.keys()))
    return _compress_map[compress]


def get_qmethod(qmethod):
    if qmethod not in _qmethod_map:
        if isinstance(qmethod, str):
            qmethod = qmethod.upper()
        elif isinstance(qmethod, bytes):
            # in py27, bytes are str, so we can safely assume
            # py3 here
            qmethod = str(qmethod, 'ascii').upper()

    if qmethod not in _qmethod_map:
        raise ValueError(
            "qmethod must be one of %s" % list(_qmethod_map.keys()))

    return _qmethod_map[qmethod]


def check_comptype_img(comptype, dtype_str):

    if comptype == NOCOMPRESS:
        return

    # if dtype_str == 'i8':
        # no i8 allowed for tile-compressed images
    #    raise ValueError("8-byte integers not supported when "
    #                     "using tile compression")

    if comptype == PLIO_1:
        # no unsigned u4/u8 for plio
        if dtype_str == 'u4' or dtype_str == 'u8':
            raise ValueError("Unsigned 4/8-byte integers currently not "
                             "allowed when writing using PLIO "
                             "tile compression")


def _extract_table_type(type):
    """
    Get the numerical table type
    """
    if isinstance(type, str):
        type = type.lower()
        if type[0:7] == 'binary':
            table_type = BINARY_TBL
        elif type[0:6] == 'ascii':
            table_type = ASCII_TBL
        else:
            raise ValueError(
                "table type string should begin with 'binary' or 'ascii' "
                "(case insensitive)")
    else:
        type = int(type)
        if type not in [BINARY_TBL, ASCII_TBL]:
            raise ValueError(
                "table type num should be BINARY_TBL (%d) or "
                "ASCII_TBL (%d)" % (BINARY_TBL, ASCII_TBL))
        table_type = type

    return table_type


_compress_map = {
    None: NOCOMPRESS,
    'RICE': RICE_1,
    'RICE_1': RICE_1,
    'GZIP': GZIP_1,
    'GZIP_1': GZIP_1,
    'GZIP_2': GZIP_2,
    'PLIO': PLIO_1,
    'PLIO_1': PLIO_1,
    'HCOMPRESS': HCOMPRESS_1,
    'HCOMPRESS_1': HCOMPRESS_1,
    NOCOMPRESS: None,
    RICE_1: 'RICE_1',
    GZIP_1: 'GZIP_1',
    GZIP_2: 'GZIP_2',
    PLIO_1: 'PLIO_1',
    HCOMPRESS_1: 'HCOMPRESS_1',
}

_qmethod_map = {
    None: NO_DITHER,
    'NO_DITHER': NO_DITHER,
    'SUBTRACTIVE_DITHER_1': SUBTRACTIVE_DITHER_1,
    'SUBTRACTIVE_DITHER_2': SUBTRACTIVE_DITHER_2,
    NO_DITHER: NO_DITHER,
    SUBTRACTIVE_DITHER_1: SUBTRACTIVE_DITHER_1,
    SUBTRACTIVE_DITHER_2: SUBTRACTIVE_DITHER_2,
}

_modeprint_map = {
    'r': 'READONLY', 'rw': 'READWRITE', 0: 'READONLY', 1: 'READWRITE'}
_char_modemap = {
    'r': 'r', 'rw': 'rw',
    READONLY: 'r', READWRITE: 'rw'}
_int_modemap = {
    'r': READONLY, 'rw': READWRITE, READONLY: READONLY, READWRITE: READWRITE}
