import copy
import warnings

from ..util import _stypes, _itypes, _ftypes, FITSRuntimeWarning
from ..header import FITSHDR

ANY_HDU = -1
IMAGE_HDU = 0
ASCII_TBL = 1
BINARY_TBL = 2

_hdu_type_map = {
    IMAGE_HDU: 'IMAGE_HDU',
    ASCII_TBL: 'ASCII_TBL',
    BINARY_TBL: 'BINARY_TBL',
    'IMAGE_HDU': IMAGE_HDU,
    'ASCII_TBL': ASCII_TBL,
    'BINARY_TBL': BINARY_TBL}


class HDUBase(object):
    """
    A representation of a FITS HDU

    parameters
    ----------
    fits: FITS object
        An instance of a _fistio_wrap.FITS object.  This is the low-level
        python object, not the FITS object defined above.
    ext: integer
        The extension number.
    """
    def __init__(self, fits, ext, **keys):

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        self._FITS = fits
        self._ext = ext
        self._ignore_scaling = False

        self._update_info()
        self._filename = self._FITS.filename()

    @property
    def ignore_scaling(self):
        """
        :return: Flag to indicate whether scaling (BZERO/BSCALE) values should
        be ignored.
        """
        return self._ignore_scaling

    @ignore_scaling.setter
    def ignore_scaling(self, ignore_scaling_flag):
        """
        Set the flag to ignore scaling.
        """
        old_val = self._ignore_scaling
        self._ignore_scaling = ignore_scaling_flag

        # Only endure the overhead of updating the info if the new value is
        # actually different.
        if old_val != self._ignore_scaling:
            self._update_info()

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
        ver = self._info['extver']
        if ver == 0:
            ver = self._info['hduver']
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
            name = _hdu_type_map[self._info['hdutype']]
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

    def _write_continue(self, value):
        """
        Write history text into the header
        """
        self._FITS.write_continue(self._ext+1, str(value))

    def write_key(self, name, value, comment=""):
        """
        Write the input value to the header

        parameters
        ----------
        name: string
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

        if name is None:

            # we write a blank keyword and the rest is a comment
            # string

            if not isinstance(comment, _stypes):
                raise ValueError('when writing blank key the value '
                                 'must be a string')

            # this might be longer than 80 but that's ok, the routine
            # will take care of it
            # card = '         ' + str(comment)
            card = '        ' + str(comment)
            self._FITS.write_record(
                self._ext+1,
                card,
            )

        elif value is None:
            self._FITS.write_undefined_key(self._ext+1,
                                           str(name),
                                           str(comment))

        elif isinstance(value, bool):
            if value:
                v = 1
            else:
                v = 0
            self._FITS.write_logical_key(self._ext+1,
                                         str(name),
                                         v,
                                         str(comment))
        elif isinstance(value, _stypes):
            self._FITS.write_string_key(self._ext+1,
                                        str(name),
                                        str(value),
                                        str(comment))
        elif isinstance(value, _ftypes):
            self._FITS.write_double_key(self._ext+1,
                                        str(name),
                                        float(value),
                                        str(comment))
        elif isinstance(value, _itypes):
            self._FITS.write_long_long_key(self._ext+1,
                                           str(name),
                                           int(value),
                                           str(comment))
        elif isinstance(value, (tuple, list)):
            vl = [str(el) for el in value]
            sval = ','.join(vl)
            self._FITS.write_string_key(self._ext+1,
                                        str(name),
                                        sval,
                                        str(comment))
        else:
            sval = str(value)
            mess = (
                "warning, keyword '%s' has non-standard "
                "value type %s, "
                "Converting to string: '%s'")
            warnings.warn(mess % (name, type(value), sval), FITSRuntimeWarning)
            self._FITS.write_string_key(self._ext+1,
                                        str(name),
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
                  a 'comment' field; the order is preserved.
                - a dictionary of keyword-value pairs; no comments are written
                  in this case, and the order is arbitrary.
        clean: boolean
            If True, trim out the standard fits header keywords that are
            created on HDU creation, such as EXTEND, SIMPLE, STTYPE, TFORM,
            TDIM, XTENSION, BITPIX, NAXIS, etc.

        Notes
        -----
        Input keys named COMMENT and HISTORY are written using the
        write_comment and write_history methods.
        """

        if isinstance(records_in, FITSHDR):
            hdr = records_in
        else:
            hdr = FITSHDR(records_in)

        if clean:
            is_table = hasattr(self, '_table_type_str')
            # is_table = isinstance(self, TableHDU)
            hdr.clean(is_table=is_table)

        for r in hdr.records():
            name = r['name']
            if name is not None:
                name = name.upper()

            value = r['value']

            if name == 'COMMENT':
                self.write_comment(value)
            elif name == 'HISTORY':
                self.write_history(value)
            elif name == 'CONTINUE':
                self._write_continue(value)
            else:
                comment = r.get('comment', '')
                self.write_key(name, value, comment=comment)

    def read_header(self):
        """
        Read the header as a FITSHDR

        The FITSHDR allows access to the values and comments by name and
        number.
        """
        # note converting strings
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
        return self._FITS.read_header(self._ext+1)

    def _update_info(self):
        """
        Update metadata for this HDU
        """
        try:
            self._FITS.movabs_hdu(self._ext+1)
        except IOError:
            raise RuntimeError("no such hdu")

        self._info = self._FITS.get_hdu_info(self._ext+1, self._ignore_scaling)

    def _get_repr_list(self):
        """
        Get some representation data common to all HDU types
        """
        spacing = ' '*2
        text = ['']
        text.append("%sfile: %s" % (spacing, self._filename))
        text.append("%sextension: %d" % (spacing, self._info['hdunum']-1))
        text.append(
            "%stype: %s" % (spacing, _hdu_type_map[self._info['hdutype']]))

        extname = self.get_extname()
        if extname != "":
            text.append("%sextname: %s" % (spacing, extname))
        extver = self.get_extver()
        if extver != 0:
            text.append("%sextver: %s" % (spacing, extver))

        return text, spacing
