"""
fitslib

See docs for the package, e.g.
    import fitsio
    help(fitsio)
In ipython:
    fitsio?

Also see docs for the FITS class and its methods.
    help(fitsio.FITS)
In ipython:
    fitsio.FITS?
"""
import os
import numpy
from . import _fitsio_wrap
import copy


def read(filename, ext=None, extver=None, rows=None, columns=None, header=False, case_sensitive=False):
    """
    Convenience function to read data from the specified FITS HDU

    By default, all data are read.  For tables, send columns= and rows= to
    select subsets of the data.  Table data are read into a recarray; use a
    FITS object and read_column() to get a single column as an ordinary array.

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

    """

    with FITS(filename, case_sensitive=case_sensitive) as fits:
        if ext is None:
            for i in xrange(len(fits)):
                if fits[i].has_data():
                    ext=i
                    break
            if ext is None:
                raise ValueError("No extensions have data")

        item=_make_item(ext, extver=extver)

        data = fits[item].read(rows=rows, columns=columns)
        if header:
            h = fits[item].read_header()
            return data, h
        else:
            return data


def read_header(filename, ext, extver=None, case_sensitive=False):
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
    ext: number or string
        The extension.  Either the numerical extension from zero
        or a string extension name.
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

def _make_item(ext, extver=None):
    if extver is not None:
        # e
        item=(ext,extver)
    else:
        item=ext
    return item



def write(filename, data, extname=None, extver=None, units=None, compress=None, header=None, clobber=False):
    """
    Convenience function to Write data to a FITS file.

    Under the hood, a FITS object is constructed.

    parameters
    ----------
    filename: string
        A filename. 
    data:
        Either a normal n-dimensional array or a recarray.  Images are written
        to a new IMAGE_HDU and recarrays are written to BINARY_TBl hdus.
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
    clobber: bool, optional
        If True, overwrite any existing file. Default is to append
        a new extension on existing files.


    table keywords
    --------------
    These keywords are only active when writing tables.

    units: list
        A list of strings representing units for each column.

    """
    with FITS(filename, 'rw', clobber=clobber) as fits:
        fits.write(data, units=units, extname=extname, extver=extver, 
                   compress=compress, header=header)


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

class FITS:
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
    """
    def __init__(self, filename, mode='r', clobber=False, case_sensitive=False):
        filename = extract_filename(filename)
        self.filename = filename
        self.mode=mode
        self.case_sensitive=case_sensitive

        if mode not in _int_modemap:
            raise ValueError("mode should be one of 'r','rw',READONLY,READWRITE")

        self.charmode = _char_modemap[mode]
        self.intmode = _int_modemap[mode]

        create=0
        if mode in [READWRITE,'rw']:
            if clobber:
                create=1
                if os.path.exists(filename):
                    print 'Removing existing file'
                    os.remove(filename)
            else:
                if os.path.exists(filename):
                    create=0
                else:
                    create=1
        else:
            if not os.path.exists(filename):
                raise ValueError("File not found: %s" % filename)

        self._FITS =  _fitsio_wrap.FITS(filename, self.intmode, create)


        if (filename[-3:].lower() == '.gz' 
                or filename[-2:].upper() == '.Z'
                or filename[-4:].lower() == '.zip'):
            self.is_compressed=True
        else:
            self.is_compressed=False

    def close(self):
        """
        Close the fits file and set relevant metadata to None
        """
        if hasattr(self,'_FITS'):
            if self._FITS is not None:
                self._FITS.close()
                self._FITS=None
        self.filename=None
        self.mode=None
        self.charmode=None
        self.intmode=None
        self.hdu_list=None
        self.hdu_map=None

    def movabs_ext(self, ext):
        return self._FITS.movabs_hdu(ext+1)
    def movabs_hdu(self, hdunum):
        return self._FITS.movabs_hdu(hdunum)

    def movnam_ext(self, extname, hdutype=ANY_HDU, extver=0):
        hdu = self._FITS.movnam_hdu(hdutype, extname, extver)
        return hdu-1
    def movnam_hdu(self, extname, hdutype=ANY_HDU, extver=0):
        hdu = self._FITS.movnam_hdu(hdutype, extname, extver)
        return hdu

    def reopen(self):
        """
        close and reopen the fits file with the same mode
        """
        self._FITS.close()
        del self._FITS
        self._FITS =  _fitsio_wrap.FITS(self.filename, self.intmode, 0)
        self.update_hdu_list()

    def cfitsio_version(self):
        """
        Return the cfitsio version
        """
        return '%0.3f' % self._FITS.cfitsio_version()

    def write(self, data, units=None, extname=None, extver=None, compress=None, header=None):   
        """
        Write the data to a new HDU.

        This method is a wrapper.  If the input is an array with fields,
        FITS.write_table is called, otherwise FITS.write_image is called.

        parameters
        ----------
        data: ndarray
            An n-dimensional image or a recarray.
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

        restrictions
        ------------
        The File must be opened READWRITE
        """
        if data.dtype.fields == None:
            self.write_image(data, extname=extname, extver=extver, 
                             compress=compress, header=header)
        else:
            self.write_table(data, units=units, 
                             extname=extname, extver=extver, header=header)



    def write_image(self, img, extname=None, extver=None, compress=None, header=None):
        """
        Create a new image extension and write the data.  

        Unlike tables, the only way to create image extensions and write the
        data is atomically using this function.

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


        restrictions
        ------------
        The File must be opened READWRITE
        """

        self.create_image_hdu(img, extname=extname, extver=extver, compress=compress, header=header)
        self.update_hdu_list()
        self[-1].write_image(img)
        self.update_hdu_list()

    def create_image_hdu(self, img, extname=None, extver=None ,compress=None, header=None):
        """
        Create a new, empty image HDU and reload the hdu list.

        You can write data into the new extension using
            fits[extension].write(image)

        typically you will instead just use 

            fits.write_image(image)

        which will create the new image extension for you with the appropriate
        structure.

        parameters
        ----------
        img: ndarray
            An image with which to determine the properties of the HDU
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


        restrictions
        ------------
        The File must be opened READWRITE
        """

        if img.dtype.fields is not None:
            raise ValueError("got recarray, expected regular ndarray")
        if img.size == 0:
            raise ValueError("data must have at least 1 row")

        if extname is not None and extver is not None:
            extver = check_extver(extver)
        if extver is None:
            # will be ignored
            extver = 0
        if extname is None:
            # will be ignored
            extname=""

        comptype = get_compress_type(compress)
        check_comptype_img(comptype, img)
        self._FITS.create_image_hdu(img, comptype=comptype, 
                                    extname=extname, extver=extver)
        self.update_hdu_list()

        """
        if extname is not None:
            self[-1].write_key("EXTNAME",str(extname))
            if extver is not None:
                self[-1].write_key('EXTVER',extver)
        """
        if header is not None:
            self[-1].write_keys(header)

        if extname is not None or header is not None:
            self.update_hdu_list()



    def write_table(self, data, units=None, extname=None, extver=None, header=None):
        """
        Create a new table extension and write the data.

        The table definition is taken from the input rec array.  If you
        want to append new rows to the table, access the HDU directly
        and use the write() function, e.g.
            fits[extension].write(data, append_rows=True)

        parameters
        ----------
        data: recarray
            A numpy array with fields.  The table definition will be
            determined from this array.
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

        restrictions
        ------------
        The File must be opened READWRITE
        """

        if data.dtype.fields == None:
            raise ValueError("data must have fields")
        if data.size == 0:
            raise ValueError("data must have at least 1 row")

        names, formats, dims = descr2tabledef(data.dtype.descr)
        self.create_table_hdu(names,formats,
                              units=units, dims=dims, extname=extname,extver=extver,
                              header=header)
        
        for colnum,name in enumerate(data.dtype.names):
            self[-1].write_column(colnum, data[name])
        self.update_hdu_list()

    def create_table_hdu(self, names, formats, 
                         units=None, dims=None, extname=None, extver=None, header=None):
        """
        Create a new, empty table extension and reload the hdu list.

        You can write data into the new extension using
            fits[extension].write(array)
            fits[extension].write_column(array)

        typically you will instead just use 

            fits.write_table(recarray)

        which will create the new table extension for you with the appropriate
        fields.

        parameters
        ----------
        names: list of strings
            The list of field names
        formats: list of strings
            The TFORM format strings for each field.
        units: list of strings, optional
            An optional list of unit strings for each field.
        dims: list of strings, optional
            An optional list of dimension strings for each field.  Should
            match the repeat count for the formats fields. Be careful of
            the order since FITS is more like fortran. See the descr2tabledef
            function.
        extname: string, optional
            An optional extension name.
        extver: integer, optional
            FITS allows multiple extensions to have the same name (extname).
            These extensions can optionally specify an EXTVER version number in
            the header.  Send extver= to set a particular version, which will
            be represented in the header with keyname EXTVER.  The extver must
            be an integer > 0.  If extver is not sent, the first one will be
            selected.  If ext is an integer, the extver is ignored.
        header: FITSHDR, list, dict
            A set of header keys to write. Can be one of these:
                - FITSHDR object
                - list of dictionaries containing 'name','value' and optionally
                  a 'comment' field.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary


        restrictions
        ------------
        The File must be opened READWRITE
        """

        if not isinstance(names,list) or not isinstance(formats,list):
            raise ValueError("names and formats should be lists")
        if len(names) != len(formats):
            raise ValueError("names and formats must be same length")
        if units is not None:
            if not isinstance(units,list):
                raise ValueError("units should be a list")
            if len(units) != len(names):
                raise ValueError("names and units must be same length")
        if dims is not None:
            if not isinstance(dims,list):
                raise ValueError("dims should be a list")
            if len(dims) != len(names):
                raise ValueError("names and dims must be same length")
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

        # note we can create extname in the c code for tables, but not images
        self._FITS.create_table_hdu(names, formats, tunit=units, tdim=dims, 
                                    extname=extname, extver=extver)
        self.update_hdu_list()

        if header is not None:
            self[-1].write_keys(header)
            self[-1]._update_info()


    def update_hdu_list(self):
        self.hdu_list = []
        self.hdu_map={}
        for ext in xrange(1000):
            try:
                hdu = FITSHDU(self._FITS, ext, 
                              case_sensitive=self.case_sensitive)
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

            except RuntimeError:
                break



    def __len__(self):
        if not hasattr(self,'hdu_list'):
            self.update_hdu_list()
        return len(self.hdu_list)

    def _extract_item(self,item):
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
                    raise ValueError("extension not found: %s, version %s %s" % (ext,ver,mess))
                hdu = self.hdu_map[key]
            else:
                if ext not in self.hdu_map:
                    raise ValueError("extension not found: %s %s" % (ext,mess))
                hdu = self.hdu_map[ext]

        return hdu

    def __repr__(self):
        spacing = ' '*2
        if not hasattr(self, 'hdu_list'):
            self.update_hdu_list()

        rep = []
        rep.append("%sfile: %s" % (spacing,self.filename))
        rep.append("%smode: %s" % (spacing,_modeprint_map[self.intmode]))

        rep.append('%sextnum %-15s %s' % (spacing,"hdutype","hduname[v]"))
        for i,hdu in enumerate(self.hdu_list):
            t = hdu.info['hdutype']
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



class FITSHDU:
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
    """
    def __init__(self, fits, ext, case_sensitive=False):
        self._FITS = fits
        self.ext = ext
        self.case_sensitive=case_sensitive
        self._update_info()
        self.filename = self._FITS.filename()

        if (self.filename[-3:].lower() == '.gz' 
                or self.filename[-2:].upper() == '.Z'
                or self.filename[-4:].lower() == '.zip'):
            self.is_compressed=True
        else:
            self.is_compressed=False


    def has_data(self):
        """
        Determine if this HDU has any data

        For images, check that the dimensions are not zero.

        For tables, check that the row count is not zero
        """
        if self.info['hdutype'] == IMAGE_HDU:
            if self.info['ndims'] == 0:
                return False
            else:
                return True
        else:
            if self.info['nrows'] > 0:
                return True
            else:
                return False

    def write_checksum(self):
        """
        Write the checksum into the header for this HDU.

        Computes the checksum for the HDU, both the data portion alone (DATASUM
        keyword) and the checksum complement for the entire HDU (CHECKSUM).

        returns
        -------
        A dict with keys 'datasum' and 'hdusum'
        """
        return self._FITS.write_checksum(self.ext+1)

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
        """

        stypes = (str,unicode,numpy.string_)
        ftypes = (float,numpy.float32,numpy.float64)
        itypes = (int,long,
                  numpy.uint8,numpy.int8,
                  numpy.uint16,numpy.int16,
                  numpy.uint32,numpy.int32,
                  numpy.uint64,numpy.int64)


        if isinstance(value, stypes):
            self._FITS.write_string_key(self.ext+1,
                                        str(keyname),
                                        str(value),
                                        str(comment))
        elif isinstance(value, ftypes):
            self._FITS.write_double_key(self.ext+1,
                                        str(keyname),
                                        float(value),
                                        str(comment))
        elif isinstance(value, itypes):
            self._FITS.write_long_key(self.ext+1,
                                      str(keyname),
                                      int(value),
                                      str(comment))

    def write_keys(self, records):
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
        """

        if isinstance(records, dict):
            for k in records:
                self.write_key(k,records[k])
        else:
            if isinstance(records,list):
                h = FITSHDR(records)
            elif isinstance(records,FITSHDR):
                h = records

            for r in h.records():
                name=r['name']
                value=r['value']
                comment = r.get('comment','')
                self.write_key(name,value,comment)



    def write_image(self, img):
        """
        Write the image into this HDU

        parameters
        ----------
        img: ndarray
            A simple numpy ndarray
        """
        if img.dtype.fields is not None:
            raise ValueError("got recarray, expected regular ndarray")
        if img.size == 0:
            raise ValueError("data must have at least 1 row")
        self._FITS.write_image(self.ext+1, img)
        self._update_info()

    def write_column(self, column, data):
        """
        Write data to a column in this HDU

        parameters
        ----------
        column: scalar string/integer
            The column in which to write.  Can be the name or number (0 offset)
        column: ndarray
            Numerical python array to write.  This should match the
            shape of the column.  You are probably better using fits.write_table()
            to be sure.
        """

        colnum = self._extract_colnum(column)

        # need it to be contiguous and native byte order.  For now, make a
        # copy.  but we may be able to avoid this with some care.
        data = numpy.array(data, ndmin=1)
        self._FITS.write_column(self.ext+1, colnum+1, data)
        self._update_info()

    def read_header(self):
        """
        Read the header as a FITSHDR

        The FITSHDR allows access to the values and comments by name and
        number.
        """
        return FITSHDR(self.read_header_list())


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
        return self._FITS.read_header(self.ext+1)

    def read(self, columns=None, rows=None):
        """
        read data from this HDU

        By default, all data are read.  For tables, send columns= and rows= to
        select subsets of the data.  Table data are read into a recarray; use
        read_column() to get a single column as an ordinary array.

        parameters
        ----------
        columns: optional
            An optional set of columns to read from table HDUs.  Default is to
            read all.  Can be string or number.
        rows: optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        """
        if self.info['hdutype'] == IMAGE_HDU:
            return self.read_image()

        if columns is not None and rows is not None:
            return self.read_columns(columns, rows)
        elif columns is not None:
            return self.read_columns(columns)
        elif rows is not None:
            return self.read_rows(rows)
        else:
            return self.read_all()


    def read_image(self):
        """
        Read the image.

        If the HDU is an IMAGE_HDU, read the corresponding image.  Compression
        and scaling are dealt with properly.
        """
        dtype, shape = self._get_image_dtype_and_shape()
        array = numpy.zeros(shape, dtype=dtype)
        self._FITS.read_image(self.ext+1, array)
        return array

    def read_column(self, col, rows=None):
        """
        Read the specified column

        parameters
        ----------
        col: string/int,  required
            The column name or number.
        rows: optional
            An optional set of row numbers to read.

        """
        if self.info['hdutype'] == _hdu_type_map['IMAGE_HDU']:
            raise ValueError("Cannot yet read columns from an image HDU")

        colnum = self._extract_colnum(col)
        rows = self._extract_rows(rows)

        npy_type, shape = self._get_simple_dtype_and_shape(colnum, rows=rows)

        array = numpy.zeros(shape, dtype=npy_type)

        self._FITS.read_column(self.ext+1,colnum+1, array, rows)
        
        self._rescale_array(array, 
                            self.info['colinfo'][colnum]['tscale'], 
                            self.info['colinfo'][colnum]['tzero'])
        return array


    def read_all(self, slow=False):
        """
        Read an entire table.

        parameters
        ----------
        slow: bool, optional
            Read the columns one at a time rather than all at once.
            This will be done automatically for .gz or .Z files.
        """
        """
        if self.is_compressed:
            # we need to use the inernal cfitsio buffers in this case;
            # read_columns always uses buffers
            colnums = self._extract_colnums()
            return self.read_columns(colnums)
        """

        dtype = self.get_rec_dtype()
        nrows = self.info['nrows']
        array = numpy.zeros(nrows, dtype=dtype)

        # read entire thing as a single fread.  This won't work for .gz or
        # .Z files because we have to work with the buffers
        self._FITS.read_as_rec(self.ext+1, array)

        for colnum,name in enumerate(array.dtype.names):
            self._rescale_array(array[name], 
                                self.info['colinfo'][colnum]['tscale'], 
                                self.info['colinfo'][colnum]['tzero'])


        return array

    def __getitem__(self, arg):
        res, isrows, isslice = \
            self.process_args_as_rows_or_columns(arg)

        if isrows:
            # rows were entered: read all columns
            if isslice:
                return self.read_slice(res.start, res.stop, res.step)
            else:
                return self.read(rows=res)
        else:
            raise ValueError("Currently you must send a row slice")
        return self.read_slice(firstrow, lastrow)

    def read_slice(self, firstrow, lastrow, step=1):
        """
        Read the specified row slice.

        Read all rows between firstrow and lastrow (non-inclusive, as per
        python slice notation).

        parameters
        ----------
        firstrow: integer
            The first row to read
        lastrow: integer
            The last row to read, non-inclusive.  This follows the python list
            slice convention that one does not include the last element.
        step: integer, optional
            Step between rows.  If step is 2, skip every other row.
        """

        if self.info['hdutype'] == _hdu_type_map['IMAGE_HDU']:
            raise ValueError("slices currently only supported for tables")

        maxrow = self.info['nrows']
        if firstrow < 0 or lastrow > maxrow:
            raise ValueError("slice must specify a sub-range of [%d,%d]" % (0,maxrow))

        dtype = self.get_rec_dtype()

        if step != 1:
            rows = numpy.arange(firstrow, lastrow, step, dtype='i8')
            return self.read(rows=rows)
        else:
            # no +1 because lastrow is non-inclusive
            nrows=lastrow-firstrow
            array = numpy.zeros(nrows, dtype=dtype)

            # only first needs to be +1
            self._FITS.read_rec_range(self.ext+1, firstrow+1, lastrow, array)

            for colnum,name in enumerate(array.dtype.names):
                self._rescale_array(array[name], 
                                    self.info['colinfo'][colnum]['tscale'], 
                                    self.info['colinfo'][colnum]['tzero'])

            return array


    def read_rows(self, rows):
        """
        Read the specified rows.

        parameters
        ----------
        rows: list,array
            A list or array of row indices.
        """
        if rows is None:
            # we actually want all rows!
            return self.read_all()

        rows = self._extract_rows(rows)
        dtype = self.get_rec_dtype()
        array = numpy.zeros(rows.size, dtype=dtype)
        self._FITS.read_rows_as_rec(self.ext+1, array, rows)

        for colnum,name in enumerate(array.dtype.names):
            self._rescale_array(array[name], 
                                self.info['colinfo'][colnum]['tscale'], 
                                self.info['colinfo'][colnum]['tzero'])

        return array


    def read_columns(self, columns, rows=None):
        """
        read a subset of columns from this binary table HDU

        By default, all rows are read.  Send rows= to select subsets of the
        data.  Table data are read into a recarray; use read_column() to get a
        single column as an ordinary array.

        parameters
        ----------
        columns: list/array
            An optional set of columns to read from table HDUs.  Can be string
            or number.
        rows: list/array, optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        """

        if self.info['hdutype'] == _hdu_type_map['IMAGE_HDU']:
            raise ValueError("Cannot yet read columns from an image HDU")

        # if columns is None, returns all.  Guaranteed to be unique and sorted
        colnums = self._extract_colnums(columns)
        # if rows is None still returns None, and is correctly interpreted
        # by the reader to mean all
        rows = self._extract_rows(rows)

        #if colnums.size == self.ncol and rows is None:
        #    # we are reading everything
        #    return self.read()

        # this is the full dtype for all columns
        dtype = self.get_rec_dtype(colnums)

        if rows is None:
            nrows = self.info['nrows']
        else:
            nrows = rows.size
        array = numpy.zeros(nrows, dtype=dtype)

        colnumsp = colnums[:].copy()
        colnumsp[:] += 1
        self._FITS.read_columns_as_rec(self.ext+1, colnumsp, array, rows)
        
        for i in xrange(colnums.size):
            colnum = int(colnums[i])
            name = array.dtype.names[i]
            self._rescale_array(array[name], 
                                self.info['colinfo'][colnum]['tscale'], 
                                self.info['colinfo'][colnum]['tzero'])
        return array



    def _extract_rows(self, rows):
        if rows is not None:
            rows = numpy.array(rows, ndmin=1, copy=False, dtype='i8')
            # returns unique, sorted
            rows = numpy.unique(rows)

            maxrow = self.info['nrows']-1
            if rows[0] < 0 or rows[-1] > maxrow:
                raise ValueError("rows must be in [%d,%d]" % (0,maxrow))
        return rows

    def process_args_as_rows_or_columns(self, arg, unpack=False):
        """

        args must be a tuple.  Only the first one or two args are used.

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
            isslice=True
            if unpack:
                result = self.slice2rows(arg.start, arg.stop, arg.step)
            else:
                result = self.process_slice(arg)
        else:
            # a single object was entered.  Probably should apply some more 
            # checking on this
            isrows=True

        return result, isrows, isslice

    def process_slice(self, arg):
        start = arg.start
        stop = arg.stop
        step = arg.step

        nrows=self.info['nrows']
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

    def slice2rows(self, start, stop, step=None):
        nrows=self.info['nrows']
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
        return numpy.arange(tstart, tstop, step, dtype='intp')

    def _fix_range(self, num, isslice=True):
        """
        If el=True, then don't treat as a slice element
        """

        nrows = self.info['nrows']
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



    def _rescale_array(self, array, scale, zero):
        if scale != 1.0:
            array *= scale
        if zero != 0.0:
            array += zero

    def get_rec_dtype(self, colnums=None):
        if colnums is None:
            colnums = self._extract_colnums()

        dtype = []
        for colnum in colnums:
            dt = self.get_rec_column_dtype(colnum) 
            dtype.append(dt)
        return dtype

    def get_rec_column_dtype(self, colnum):
        npy_type = self._get_tbl_numpy_dtype(colnum)
        name = self.info['colinfo'][colnum]['name']
        tdim = self.info['colinfo'][colnum]['tdim']

        shape = tdim2shape(tdim, is_string=(npy_type[0] == 'S'))

        if shape is not None:
            return (name,npy_type,shape)
        else:
            return (name,npy_type)

    def _get_image_dtype_and_shape(self):

        if self.info['hdutype'] != _hdu_type_map['IMAGE_HDU']:
            raise ValueError("HDU is not an IMAGE_HDU")

        npy_dtype = self._get_image_numpy_dtype()

        if self.info['ndims'] != 0:
            shape = self.info['dims']
        else:
            raise ValueError("no image present in HDU")

        return npy_dtype, shape

    def _get_simple_dtype_and_shape(self, colnum, rows=None):
        """
        When reading a single column, we want the basic data
        type and the shape of the array.

        for scalar columns, shape is just nrows, otherwise
        it is (nrows, dim1, dim2)

        """

        # basic datatype
        npy_type = self._get_tbl_numpy_dtype(colnum)
        info = self.info['colinfo'][colnum]

        if rows is None:
            nrows = self.info['nrows']
        else:
            nrows = rows.size

        shape = None
        tdim = info['tdim']

        shape = tdim2shape(tdim, is_string=(npy_type[0] == 'S'))
        if shape is not None:
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

    def _get_image_numpy_dtype(self):
        try:
            ftype = self.info['img_equiv_type']
            npy_type = _image_bitpix2npy[ftype]
        except KeyError:
            raise KeyError("unsupported fits data type: %d" % ftype)

        return npy_type

    def _get_tbl_numpy_dtype(self, colnum, include_endianness=True):
        try:
            ftype = self.info['colinfo'][colnum]['eqtype']
            npy_type = _table_fits2npy[ftype]
        except KeyError:
            raise KeyError("unsupported fits data type: %d" % ftype)

        if include_endianness:
            if npy_type not in ['u1','i1','S']:
                npy_type = '>'+npy_type
        if npy_type == 'S':
            width = self.info['colinfo'][colnum]['width']
            npy_type = 'S%d' % width
        return npy_type

    def _extract_colnums(self, columns=None):
        if columns is None:
            return numpy.arange(self.ncol, dtype='i8')
        
        if not isinstance(columns,(tuple,list,numpy.ndarray)):
            columns=[columns]

        colnums = numpy.zeros(len(columns), dtype='i8')
        for i in xrange(colnums.size):
            colnums[i] = self._extract_colnum(columns[i])

        # returns unique sorted
        colnums = numpy.unique(colnums)
        return colnums

    def _extract_colnum(self, col):
        if isinstance(col,(int,long)):
            colnum = col

            if (colnum < 0) or (colnum > (self.ncol-1)):
                raise ValueError("column number should be in [0,%d]" % (0,self.ncol-1))
        else:
            colstr='%s' % col
            try:
                if self.case_sensitive:
                    mess="column name '%s' not found (case sensitive)" % col
                    colnum = self.colnames.index(colstr)
                else:
                    mess="column name '%s' not found (case insensitive)" % col
                    colnum = self.colnames_lower.index(colstr.lower())
            except ValueError:
                raise ValueError(mess)
        return int(colnum)

    def _update_info(self):
        # do this here first so we can catch the error
        try:
            self._FITS.movabs_hdu(self.ext+1)
        except IOError:
            raise RuntimeError("no such hdu")

        self.info = self._FITS.get_hdu_info(self.ext+1)
        # convert to c order
        if 'dims' in self.info:
            self.info['dims'] = list( reversed(self.info['dims']) )
        if 'colinfo' in self.info:
            self.colnames = [i['name'] for i in self.info['colinfo']]
            self.colnames_lower = [i['name'].lower() for i in self.info['colinfo']]
            self.ncol = len(self.colnames)

    def get_extname(self):
        name = self.info['extname']
        if name.strip() == '':
            name = self.info['hduname']
        return name.strip()
    def get_extver(self):
        ver=self.info['extver']
        if ver == 0:
            ver=self.info['hduver']
        return ver

    def __repr__(self):
        spacing = ' '*2
        text = []
        text.append("%sextension: %d" % (spacing,self.info['hdunum']-1))
        text.append("%stype: %s" % (spacing,_hdu_type_map[self.info['hdutype']]))

        extname=self.get_extname()
        if extname != "":
            text.append("%sextname: %s" % (spacing,extname))
        extver=self.get_extver()
        if extver != 0:
            text.append("%sextver: %s" % (spacing,extver))
        
        if self.info['hdutype'] == _hdu_type_map['IMAGE_HDU']:
            text.append("%simage info:" % spacing)
            cspacing = ' '*4

            if self.info['comptype'] is not None:
                text.append("%scompression: %s" % (cspacing,self.info['comptype']))

            if self.info['ndims'] != 0:
                dimstr = [str(d) for d in self.info['dims']]
            else:
                dimstr=''
            dimstr = ",".join(dimstr)

            dt = _image_bitpix2npy[self.info['img_equiv_type']]
            text.append("%sdata type: %s" % (cspacing,dt))
            text.append("%sdims: [%s]" % (cspacing,dimstr))

        else:
            text.append('%srows: %d' % (spacing,self.info['nrows']))
            text.append('%scolumn info:' % spacing)

            cspacing = ' '*4
            nspace = 4
            nname = 15
            ntype = 6
            format = cspacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
            pformat = cspacing + "%-" + str(nname) + "s\n %" + str(nspace+nname+ntype) + "s  %s"

            for colnum,c in enumerate(self.info['colinfo']):
                if len(c['name']) > 15:
                    f = pformat
                else:
                    f = format

                dt = self._get_tbl_numpy_dtype(colnum, include_endianness=False)

                tdim = c['tdim']
                dimstr=''
                if dt[0] == 'S':
                    if len(tdim) > 1:
                        dimstr = [str(d) for d in tdim[1:]]
                else:
                    if len(tdim) > 1 or tdim[0] > 1:
                        dimstr = [str(d) for d in tdim]
                if dimstr != '':
                    dimstr = ','.join(dimstr)
                    dimstr = 'array[%s]' % dimstr

                s = f % (c['name'],dt,dimstr)
                text.append(s)

        text = '\n'.join(text)
        return text


















def check_extver(extver):
    if extver is None:
        return 0
    extver=int(extver)
    if extver <= 0:
        raise ValueError("extver must be > 0")
    return extver

def extract_filename(filename):
    if filename[0] == "!":
        filename=filename[1:]
    filename = os.path.expandvars(filename)
    filename = os.path.expanduser(filename)
    return filename

def tdim2shape(tdim, is_string=False):
    shape=None
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

def descr2tabledef(descr):
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
        # these have the form '<f4' or '|S25', etc.  Extract the pure type
        npy_dtype = d[1][1:]
        if npy_dtype[0] == 'S':
            name, form, dim = npy_string2fits(d)
        else:
            name, form, dim = npy_num2fits(d)

        names.append(name)
        formats.append(form)
        dims.append(dim)

    return names, formats, dims

def npy_num2fits(d):
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
    form = _table_npy2fits_form[npy_dtype]

    # now the dimensions
    if len(d) > 2:
        if isinstance(d[2], tuple):
            # this is an array column.  the form
            # should be total elements followed by A
            #count = 1
            #count = [count*el for el in d[2]]
            count=reduce(lambda x, y: x*y, d[2])
            form = '%d%s' % (count,form)

            # will have to do tests to see if this is the right order
            dim = list(reversed(d[2]))
            #dim = d[2]
            dim = [str(e) for e in dim]
            dim = '(' + ','.join(dim)+')'
        else:
            # this is a vector (1d array) column
            count = d[2]
            form = '%d%s' % (count,form)

    return name, form, dim


def npy_string2fits(d):
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
        form = string_size_str+'A'
    else:
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

class FITSHDR:
    def __init__(self, record_list=None):
        self._record_list = []
        self._record_map = {}

        if record_list is not None:

            if isinstance(record_list, dict):
                for k in record_list:
                    r = {'name':k, 'value':record_list[k]}
                    self.add_record(r)
            elif isinstance(record_list, list):
                for r in record_list:
                    self.add_record(r)
            else:
                raise ValueError("expected a dict or list of dicts")


    def add_record(self, record):   
        self.check_record(record)
        self._record_list.append(record)
        self._add_to_map(record, len(self._record_list)-1)

    def _add_to_map(self, record, num):
        self._record_map[record['name']] = record
        self._record_map[num] = record

    def check_record(self, record):
        if not isinstance(record,dict):
            raise ValueError("each record must be a dictionary")
        if 'name' not in record:
            raise ValueError("each record must have a 'name' field")
        if 'value' not in record:
            raise ValueError("each record must have a 'value' field")

    def get_comment(self, item):
        if item not in self._record_map:
            raise ValueError("unknown record: %s" % item)
        if 'comment' not in self._record_map[item]:
            return None
        else:
            return self._record_map[item]['comment']

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

    def __len__(self):
        return len(self._record_list)

    def __contains__(self, item):
        return item in self._record_map

    def get(self, item):
        if item not in self._record_map:
            raise ValueError("unknown record: %s" % item)

        if item == 'COMMENT':
            # there could be many comments, just return one
            v = self._record_map[item].get('comment','')
            return v

        s = self._record_map[item]['value']
        try:
            val = eval(s)
        except:
            val = s
        return val


    def __getitem__(self, item):
        return self.get(item)

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
            comment = record.get('comment','')
            card = 'COMMENT %s' % comment
        else:
            card = '%-8s= ' % name[0:8]
            # these may be string representations of data, or actual strings
            if isinstance(value,(str,unicode)):
                value = str(value)
                if len(value) > 0:
                    if value[0] != "'":
                        # this is a string representing a string header field
                        # make it look like it will look in the header
                        value = "'" + value + "'"
                        vstr = '%-20s' % value
                    else:
                        # this is a string representing a number
                        vstr = "%20s" % value
            else:
                vstr = '%20s' % value
                    
            card += vstr

            if 'comment' in record:
                f += ' / %s' % record['comment']

        return card[0:80]

    def __repr__(self):
        rep=[]
        for r in self._record_list:
            if 'card' not in r:
                card = self._record2card(r)
            else:
                card = r['card']

            rep.append(card)
        return '\n'.join(rep)

def get_compress_type(compress):
    if compress is not None:
        compress = str(compress).upper()
    if compress not in _compress_map:
        raise ValueError("compress must be one of %s" % list(_compress_map.keys()))
    return _compress_map[compress]
def check_comptype_img(comptype, img):

    if comptype == NOCOMPRESS:
        return

    if img.dtype.descr[0][1][1:] == 'i8':
        # no i8 allowed for tile-compressed images
        raise ValueError("8-byte integers not supported when  using tile compression")

    if comptype == PLIO_1:
        # no unsigned for plio
        if img.dtype.descr[0][1][1] == 'u':
            raise ValueError("unsigned integers not allowed when using PLIO tile compression")

def isstring(arg):
    return isinstance(arg, (str,unicode))

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
_table_fits2npy = {11:'u1',
                   12: 'i1',
                   14: 'i1', # logical. Note pyfits uses this for i1, cfitsio casts to char*
                   16: 'S',
                   20: 'u2',
                   21: 'i2',
                   30: 'u4',
                   31: 'i4',
                   40: 'u4',
                   41: 'i4',
                   42: 'f4',
                   81: 'i8',
                   82: 'f8'}

# for TFORM
# note actually there are no unsigned, they get scaled
# and converted to signed.  When reading, can only do signed.
_table_npy2fits_form = {'u1':'B',
                        'i1':'S', # gets converted to unsigned
                        'S' :'A',
                        'u2':'U', # gets converted to signed
                        'i2':'I',
                        'u4':'V', # gets converted to signed
                        'i4':'J',
                        'i8':'K',
                        'f4':'E',
                        'f8':'D'}

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

