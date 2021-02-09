"""
image HDU classes for fitslib, part of the fitsio package.

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
import copy
import warnings
from functools import reduce

import numpy

from ..util import (
    IS_PY3,
    isstring,
    isinteger,
    is_object,
    fields_are_object,
    array_to_native,
    array_to_native_c,
    FITSRuntimeWarning,
    mks
)
from .base import HDUBase, ASCII_TBL, IMAGE_HDU, _hdu_type_map

# for python3 compat
if IS_PY3:
    xrange = range


class TableHDU(HDUBase):
    """
    A table HDU

    parameters
    ----------
    fits: FITS object
        An instance of a _fistio_wrap.FITS object.  This is the low-level
        python object, not the FITS object defined above.
    ext: integer
        The extension number.
    lower: bool, optional
        If True, force all columns names to lower case in output
    upper: bool, optional
        If True, force all columns names to upper case in output
    trim_strings: bool, optional
        If True, trim trailing spaces from strings. Default is False.
    vstorage: string, optional
        Set the default method to store variable length columns.  Can be
        'fixed' or 'object'.  See docs on fitsio.FITS for details.
    case_sensitive: bool, optional
        Match column names and extension names with case-sensitivity.  Default
        is False.
    iter_row_buffer: integer
        Number of rows to buffer when iterating over table HDUs.
        Default is 1.
    write_bitcols: bool, optional
        If True, write logicals a a bit column. Default is False.
    """
    def __init__(self, fits, ext,
                 lower=False, upper=False, trim_strings=False,
                 vstorage='fixed', case_sensitive=False, iter_row_buffer=1,
                 write_bitcols=False, **keys):

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        # NOTE: The defaults of False above cannot be changed since they
        # are or'ed with the method defaults below.
        super(TableHDU, self).__init__(fits, ext)

        self.lower = lower
        self.upper = upper
        self.trim_strings = trim_strings

        self._vstorage = vstorage
        self.case_sensitive = case_sensitive
        self._iter_row_buffer = iter_row_buffer
        self.write_bitcols = write_bitcols

        if self._info['hdutype'] == ASCII_TBL:
            self._table_type_str = 'ascii'
        else:
            self._table_type_str = 'binary'

    def get_nrows(self):
        """
        Get number of rows in the table.
        """
        nrows = self._info.get('nrows', None)
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
            raise ValueError(
                "colnum out of range [0,%s-1]" % (0, len(self._colnames)))
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

    def write(self, data, firstrow=0, columns=None, names=None, slow=False,
              **keys):
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
            of names or column numbers. You can also use the `names` keyword
            argument.
        names: list, optional
            If data is a list of arrays, you must send columns as a list
            of names or column numbers. You can also use the `columns` keyword
            argument.
        slow: bool, optional
            If True, use a slower method to write one column at a time. Useful
            for debugging.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        isrec = False
        if isinstance(data, (list, dict)):
            if isinstance(data, list):
                data_list = data
                if columns is not None:
                    columns_all = columns
                elif names is not None:
                    columns_all = names
                else:
                    raise ValueError(
                        "you must send `columns` or `names` "
                        "with a list of arrays")
            else:
                columns_all = list(data.keys())
                data_list = [data[n] for n in columns_all]

            colnums_all = [self._extract_colnum(c) for c in columns_all]
            names = [self.get_colname(c) for c in colnums_all]

            isobj = numpy.zeros(len(data_list), dtype=bool)
            for i in xrange(len(data_list)):
                isobj[i] = is_object(data_list[i])

        else:
            if data.dtype.fields is None:
                raise ValueError("You are writing to a table, so I expected "
                                 "an array with fields as input. If you want "
                                 "to write a simple array, you should use "
                                 "write_column to write to a single column, "
                                 "or instead write to an image hdu")

            if data.shape == ():
                raise ValueError("cannot write data with shape ()")

            isrec = True
            names = data.dtype.names
            # only write object types (variable-length columns) after
            # writing the main table
            isobj = fields_are_object(data)

            data_list = []
            colnums_all = []
            for i, name in enumerate(names):
                colnum = self._extract_colnum(name)
                data_list.append(data[name])
                colnums_all.append(colnum)

        if slow:
            for i, name in enumerate(names):
                if not isobj[i]:
                    self.write_column(name, data_list[i], firstrow=firstrow)
        else:

            nonobj_colnums = []
            nonobj_arrays = []
            for i in xrange(len(data_list)):
                if not isobj[i]:
                    nonobj_colnums.append(colnums_all[i])
                    if isrec:
                        # this still leaves possibility of f-order sub-arrays..
                        colref = array_to_native(data_list[i], inplace=False)
                    else:
                        colref = array_to_native_c(data_list[i], inplace=False)

                    if IS_PY3 and colref.dtype.char == 'U':
                        # for python3, we convert unicode to ascii
                        # this will error if the character is not in ascii
                        colref = colref.astype('S', copy=False)

                    nonobj_arrays.append(colref)

            for tcolnum, tdata in zip(nonobj_colnums, nonobj_arrays):
                self._verify_column_data(tcolnum, tdata)

            if len(nonobj_arrays) > 0:
                self._FITS.write_columns(
                    self._ext+1, nonobj_colnums, nonobj_arrays,
                    firstrow=firstrow+1, write_bitcols=self.write_bitcols)

        # writing the object arrays always occurs the same way
        # need to make sure this works for array fields
        for i, name in enumerate(names):
            if isobj[i]:
                self.write_var_column(name, data_list[i], firstrow=firstrow)

        self._update_info()

    def write_column(self, column, data, firstrow=0, **keys):
        """
        Write data to a column in this HDU

        This HDU must be a table HDU.

        parameters
        ----------
        column: scalar string/integer
            The column in which to write.  Can be the name or number (0 offset)
        data: ndarray
            Numerical python array to write.  This should match the
            shape of the column.  You are probably better using
            fits.write_table() to be sure.
        firstrow: integer, optional
            At which row you should begin writing.  Be sure you know what you
            are doing!  For appending see the append() method.  Default 0.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

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

        if IS_PY3 and data_send.dtype.char == 'U':
            # for python3, we convert unicode to ascii
            # this will error if the character is not in ascii
            data_send = data_send.astype('S', copy=False)

        self._verify_column_data(colnum, data_send)

        self._FITS.write_columns(
            self._ext+1,
            [colnum],
            [data_send],
            firstrow=firstrow+1,
            write_bitcols=self.write_bitcols,
        )

        del data_send
        self._update_info()

    def _verify_column_data(self, colnum, data):
        """
        verify the input data is of the correct type and shape
        """
        this_dt = data.dtype.descr[0]

        if len(data.shape) > 2:
            this_shape = data.shape[1:]
        elif len(data.shape) == 2 and data.shape[1] > 1:
            this_shape = data.shape[1:]
        else:
            this_shape = ()

        this_npy_type = this_dt[1][1:]

        npy_type, isvar, istbit = self._get_tbl_numpy_dtype(colnum)
        info = self._info['colinfo'][colnum]

        if npy_type[0] in ['>', '<', '|']:
            npy_type = npy_type[1:]

        col_name = info['name']
        col_tdim = info['tdim']
        col_shape = _tdim2shape(
            col_tdim, col_name, is_string=(npy_type[0] == 'S'))

        if col_shape is None:
            if this_shape == ():
                this_shape = None

        if col_shape is not None and not isinstance(col_shape, tuple):
            col_shape = (col_shape,)

        """
        print('column name:',col_name)
        print(data.shape)
        print('col tdim', info['tdim'])
        print('column dtype:',npy_type)
        print('input dtype:',this_npy_type)
        print('column shape:',col_shape)
        print('input shape:',this_shape)
        print()
        """

        # this mismatch is OK
        if npy_type == 'i1' and this_npy_type == 'b1':
            this_npy_type = 'i1'

        if isinstance(self, AsciiTableHDU):
            # we don't enforce types exact for ascii
            if npy_type == 'i8' and this_npy_type in ['i2', 'i4']:
                this_npy_type = 'i8'
            elif npy_type == 'f8' and this_npy_type == 'f4':
                this_npy_type = 'f8'

        if this_npy_type != npy_type:
            raise ValueError(
                "bad input data for column '%s': "
                "expected '%s', got '%s'" % (
                    col_name, npy_type, this_npy_type))

        if this_shape != col_shape:
            raise ValueError(
                "bad input shape for column '%s': "
                "expected '%s', got '%s'" % (col_name, col_shape, this_shape))

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

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if not is_object(data):
            raise ValueError("Only object fields can be written to "
                             "variable-length arrays")
        colnum = self._extract_colnum(column)

        self._FITS.write_var_column(self._ext+1, colnum+1, data,
                                    firstrow=firstrow+1)
        self._update_info()

    def insert_column(self, name, data, colnum=None, write_bitcols=None,
                      **keys):
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
        write_bitcols: bool, optional
            If set, write logical as bit cols. This can over-ride the
            internal class setting. Default of None respects the inner
            class setting.

        Notes
        -----
        This method is used un-modified by ascii tables as well.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if write_bitcols is None:
            write_bitcols = self.write_bitcols

        if name in self._colnames:
            raise ValueError("column '%s' already exists" % name)

        if IS_PY3 and data.dtype.char == 'U':
            # fast dtype conversion using an empty array
            # we could hack at the actual text description, but using
            # the numpy API is probably safer
            # this also avoids doing a dtype conversion on every array
            # element which could b expensive
            descr = numpy.empty(1).astype(data.dtype).astype('S').dtype.descr
        else:
            descr = data.dtype.descr

        if len(descr) > 1:
            raise ValueError("you can only insert a single column, "
                             "requested: %s" % descr)

        this_descr = descr[0]
        this_descr = [name, this_descr[1]]
        if len(data.shape) > 1:
            this_descr += [data.shape[1:]]
        this_descr = tuple(this_descr)

        name, fmt, dims = _npy2fits(
            this_descr,
            table_type=self._table_type_str,
            write_bitcols=write_bitcols,
        )
        if dims is not None:
            dims = [dims]

        if colnum is None:
            new_colnum = len(self._info['colinfo']) + 1
        else:
            new_colnum = colnum+1

        self._FITS.insert_col(self._ext+1, new_colnum, name, fmt, tdim=dims)

        self._update_info()

        self.write_column(name, data)

    def append(self, data, columns=None, names=None, **keys):
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
            If data is a list of arrays, you must send columns as a list
            of names or column numbers. You can also use the `names` keyword
            argument.
        names: list, optional
            If data is a list of arrays, you must send columns as a list
            of names or column numbers. You can also use the `columns` keyword
            argument.
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        firstrow = self._info['nrows']
        self.write(data, firstrow=firstrow, columns=None, names=None)

    def delete_rows(self, rows):
        """
        Delete rows from the table

        parameters
        ----------
        rows: sequence or slice
            The exact rows to delete as a sequence, or a slice.

        examples
        --------
            # delete a range of rows
            with fitsio.FITS(fname,'rw') as fits:
                fits['mytable'].delete_rows(slice(3,20))

            # delete specific rows
            with fitsio.FITS(fname,'rw') as fits:
                rows2delete = [3,88,76]
                fits['mytable'].delete_rows(rows2delete)
        """

        if rows is None:
            return

        # extract and convert to 1-offset for C routine
        if isinstance(rows, slice):
            rows = self._process_slice(rows)
            if rows.step is not None and rows.step != 1:
                rows = numpy.arange(
                    rows.start+1,
                    rows.stop+1,
                    rows.step,
                )
            else:
                # rows must be 1-offset
                rows = slice(rows.start+1, rows.stop+1)
        else:
            rows = self._extract_rows(rows)
            # rows must be 1-offset
            rows += 1

        if isinstance(rows, slice):
            self._FITS.delete_row_range(self._ext+1, rows.start, rows.stop)
        else:
            if rows.size == 0:
                return

            self._FITS.delete_rows(self._ext+1, rows)

        self._update_info()

    def resize(self, nrows, front=False):
        """
        Resize the table to the given size, removing or adding rows as
        necessary.  Note if expanding the table at the end, it is more
        efficient to use the append function than resizing and then
        writing.

        New added rows are zerod, except for 'i1', 'u2' and 'u4' data types
        which get -128,32768,2147483648 respectively

        parameters
        ----------
        nrows: int
            new size of table
        front: bool, optional
            If True, add or remove rows from the front.  Default
            is False
        """

        nrows_current = self.get_nrows()
        if nrows == nrows_current:
            return

        if nrows < nrows_current:
            rowdiff = nrows_current - nrows
            if front:
                # delete from the front
                start = 0
                stop = rowdiff
            else:
                # delete from the back
                start = nrows
                stop = nrows_current

            self.delete_rows(slice(start, stop))
        else:
            rowdiff = nrows - nrows_current
            if front:
                # in this case zero is what we want, since the code inserts
                firstrow = 0
            else:
                firstrow = nrows_current
            self._FITS.insert_rows(self._ext+1, firstrow, rowdiff)

        self._update_info()

    def read(self, columns=None, rows=None, vstorage=None,
             upper=False, lower=False, trim_strings=False, **keys):
        """
        Read data from this HDU

        By default, all data are read. You can set the `columns` and/or
        `rows` keywords to read subsets of the data.

        Table data is read into a numpy recarray. To get a single column as
        a numpy.ndarray, use the `read_column` method.

        Slice notation is also supported for `TableHDU` types.

            >>> fits = fitsio.FITS(filename)
            >>> fits[ext][:]
            >>> fits[ext][2:5]
            >>> fits[ext][200:235:2]
            >>> fits[ext][rows]
            >>> fits[ext][cols][rows]

        parameters
        ----------
        columns: optional
            An optional set of columns to read from table HDUs. Default is to
            read all. Can be string or number. If a sequence, a recarray
            is always returned. If a scalar, an ordinary array is returned.
        rows: optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        vstorage: string, optional
            Over-ride the default method to store variable length columns. Can
            be 'fixed' or 'object'. See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if columns is not None:
            data = self.read_columns(
                columns, rows=rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)
        elif rows is not None:
            # combinations of row and column subsets are covered by
            # read_columns so we pass colnums=None here to get all columns
            data = self.read_rows(
                rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)
        else:
            data = self._read_all(
                vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)

        return data

    def _read_all(self, vstorage=None,
                  upper=False, lower=False, trim_strings=False, colnums=None,
                  **keys):
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
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        colnums: integer array, optional
            The column numbers, 0 offset
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        dtype, offsets, isvar = self.get_rec_dtype(
            colnums=colnums, vstorage=vstorage)

        w, = numpy.where(isvar == True)  # noqa
        has_tbit = self._check_tbit()

        if w.size > 0:
            if vstorage is None:
                _vstorage = self._vstorage
            else:
                _vstorage = vstorage
            colnums = self._extract_colnums()
            rows = None
            array = self._read_rec_with_var(colnums, rows, dtype,
                                            offsets, isvar, _vstorage)
        elif has_tbit:
            # drop down to read_columns since we can't stuff into a
            # contiguous array
            colnums = self._extract_colnums()
            array = self.read_columns(
                colnums,
                rows=None, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)
        else:
            firstrow = 1  # noqa - not used?
            nrows = self._info['nrows']
            array = numpy.zeros(nrows, dtype=dtype)

            self._FITS.read_as_rec(self._ext+1, 1, nrows, array)

            array = self._maybe_decode_fits_ascii_strings_to_unicode_py3(array)

            for colnum, name in enumerate(array.dtype.names):
                self._rescale_and_convert_field_inplace(
                    array,
                    name,
                    self._info['colinfo'][colnum]['tscale'],
                    self._info['colinfo'][colnum]['tzero'])

        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        self._maybe_trim_strings(array, trim_strings=trim_strings)
        return array

    def read_column(self, col, rows=None, vstorage=None,
                    upper=False, lower=False, trim_strings=False, **keys):
        """
        Read the specified column

        Alternatively, you can use slice notation

            >>> fits=fitsio.FITS(filename)
            >>> fits[ext][colname][:]
            >>> fits[ext][colname][2:5]
            >>> fits[ext][colname][200:235:2]
            >>> fits[ext][colname][rows]

        Note, if reading multiple columns, it is more efficient to use
        read(columns=) or slice notation with a list of column names.

        parameters
        ----------
        col: string/int, required
            The column name or number.
        rows: optional
            An optional set of row numbers to read.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        res = self.read_columns(
            [col], rows=rows, vstorage=vstorage,
            upper=upper, lower=lower, trim_strings=trim_strings)
        colname = res.dtype.names[0]
        data = res[colname]

        self._maybe_trim_strings(data, trim_strings=trim_strings)
        return data

    def read_rows(self, rows, vstorage=None,
                  upper=False, lower=False, trim_strings=False, **keys):
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
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if rows is None:
            # we actually want all rows!
            return self._read_all()

        if self._info['hdutype'] == ASCII_TBL:
            return self.read(
                rows=rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)

        rows = self._extract_rows(rows)
        dtype, offsets, isvar = self.get_rec_dtype(vstorage=vstorage)

        w, = numpy.where(isvar == True)  # noqa
        if w.size > 0:
            if vstorage is None:
                _vstorage = self._vstorage
            else:
                _vstorage = vstorage
            colnums = self._extract_colnums()
            return self._read_rec_with_var(
                colnums, rows, dtype, offsets, isvar, _vstorage)
        else:
            array = numpy.zeros(rows.size, dtype=dtype)
            self._FITS.read_rows_as_rec(self._ext+1, array, rows)

            array = self._maybe_decode_fits_ascii_strings_to_unicode_py3(array)

            for colnum, name in enumerate(array.dtype.names):
                self._rescale_and_convert_field_inplace(
                    array,
                    name,
                    self._info['colinfo'][colnum]['tscale'],
                    self._info['colinfo'][colnum]['tzero'])

        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        self._maybe_trim_strings(array, trim_strings=trim_strings)

        return array

    def read_columns(self, columns, rows=None, vstorage=None,
                     upper=False, lower=False, trim_strings=False, **keys):
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
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if self._info['hdutype'] == ASCII_TBL:
            return self.read(
                columns=columns, rows=rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)

        # if columns is None, returns all.  Guaranteed to be unique and sorted
        colnums = self._extract_colnums(columns)
        if isinstance(colnums, int):
            # scalar sent, don't read as a recarray
            return self.read_column(
                columns,
                rows=rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)

        # if rows is None still returns None, and is correctly interpreted
        # by the reader to mean all
        rows = self._extract_rows(rows)

        # this is the full dtype for all columns
        dtype, offsets, isvar = self.get_rec_dtype(
            colnums=colnums, vstorage=vstorage)

        w, = numpy.where(isvar == True)  # noqa
        if w.size > 0:
            if vstorage is None:
                _vstorage = self._vstorage
            else:
                _vstorage = vstorage
            array = self._read_rec_with_var(
                colnums, rows, dtype, offsets, isvar, _vstorage)
        else:

            if rows is None:
                nrows = self._info['nrows']
            else:
                nrows = rows.size
            array = numpy.zeros(nrows, dtype=dtype)

            colnumsp = colnums[:].copy()
            colnumsp[:] += 1
            self._FITS.read_columns_as_rec(self._ext+1, colnumsp, array, rows)

            array = self._maybe_decode_fits_ascii_strings_to_unicode_py3(array)

            for i in xrange(colnums.size):
                colnum = int(colnums[i])
                name = array.dtype.names[i]
                self._rescale_and_convert_field_inplace(
                    array,
                    name,
                    self._info['colinfo'][colnum]['tscale'],
                    self._info['colinfo'][colnum]['tzero'])

        if (self._check_tbit(colnums=colnums)):
            array = self._fix_tbit_dtype(array, colnums)

        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        self._maybe_trim_strings(array, trim_strings=trim_strings)

        return array

    def read_slice(self, firstrow, lastrow, step=1,
                   vstorage=None, lower=False, upper=False,
                   trim_strings=False, **keys):
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
            Step between rows, default 1. e.g., if step is 2, skip every other
            row.
        vstorage: string, optional
            Over-ride the default method to store variable length columns.  Can
            be 'fixed' or 'object'.  See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if self._info['hdutype'] == ASCII_TBL:
            rows = numpy.arange(firstrow, lastrow, step, dtype='i8')
            return self.read_ascii(
                rows=rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)

        if self._info['hdutype'] == IMAGE_HDU:
            raise ValueError("slices currently only supported for tables")

        maxrow = self._info['nrows']
        if firstrow < 0 or lastrow > maxrow:
            raise ValueError(
                "slice must specify a sub-range of [%d,%d]" % (0, maxrow))

        dtype, offsets, isvar = self.get_rec_dtype(vstorage=vstorage)

        w, = numpy.where(isvar == True)  # noqa
        if w.size > 0:
            if vstorage is None:
                _vstorage = self._vstorage
            else:
                _vstorage = vstorage
            rows = numpy.arange(firstrow, lastrow, step, dtype='i8')
            colnums = self._extract_colnums()
            array = self._read_rec_with_var(
                colnums, rows, dtype, offsets, isvar, _vstorage)
        else:
            if step != 1:
                rows = numpy.arange(firstrow, lastrow, step, dtype='i8')
                array = self.read(rows=rows)
            else:
                # no +1 because lastrow is non-inclusive
                nrows = lastrow - firstrow
                array = numpy.zeros(nrows, dtype=dtype)

                # only first needs to be +1.  This is becuase the c code is
                # inclusive
                self._FITS.read_as_rec(self._ext+1, firstrow+1, lastrow, array)

                array = self._maybe_decode_fits_ascii_strings_to_unicode_py3(
                    array)

                for colnum, name in enumerate(array.dtype.names):
                    self._rescale_and_convert_field_inplace(
                        array,
                        name,
                        self._info['colinfo'][colnum]['tscale'],
                        self._info['colinfo'][colnum]['tzero'])

        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        self._maybe_trim_strings(array, trim_strings=trim_strings)

        return array

    def get_rec_dtype(self, colnums=None, vstorage=None, **keys):
        """
        Get the dtype for the specified columns

        parameters
        ----------
        colnums: integer array, optional
            The column numbers, 0 offset
        vstorage: string, optional
            See docs in read_columns
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if vstorage is None:
            _vstorage = self._vstorage
        else:
            _vstorage = vstorage

        if colnums is None:
            colnums = self._extract_colnums()

        descr = []
        isvararray = numpy.zeros(len(colnums), dtype=bool)
        for i, colnum in enumerate(colnums):
            dt, isvar = self.get_rec_column_descr(colnum, _vstorage)
            descr.append(dt)
            isvararray[i] = isvar
        dtype = numpy.dtype(descr)

        offsets = numpy.zeros(len(colnums), dtype='i8')
        for i, n in enumerate(dtype.names):
            offsets[i] = dtype.fields[n][1]
        return dtype, offsets, isvararray

    def _check_tbit(self, colnums=None, **keys):
        """
        Check if one of the columns is a TBIT column

        parameters
        ----------
        colnums: integer array, optional
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if colnums is None:
            colnums = self._extract_colnums()

        has_tbit = False
        for i, colnum in enumerate(colnums):
            npy_type, isvar, istbit = self._get_tbl_numpy_dtype(colnum)
            if (istbit):
                has_tbit = True
                break

        return has_tbit

    def _fix_tbit_dtype(self, array, colnums):
        """
        If necessary, patch up the TBIT to convert to bool array

        parameters
        ----------
        array: record array
        colnums: column numbers for lookup
        """
        descr = array.dtype.descr
        for i, colnum in enumerate(colnums):
            npy_type, isvar, istbit = self._get_tbl_numpy_dtype(colnum)
            if (istbit):
                coldescr = list(descr[i])
                coldescr[1] = '?'
                descr[i] = tuple(coldescr)

        return array.view(descr)

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
        npy_type, isvar, istbit = self._get_tbl_numpy_dtype(colnum)
        info = self._info['colinfo'][colnum]
        name = info['name']

        if rows is None:
            nrows = self._info['nrows']
        else:
            nrows = rows.size

        shape = None
        tdim = info['tdim']

        shape = _tdim2shape(tdim, name, is_string=(npy_type[0] == 'S'))
        if shape is not None:
            if nrows > 1:
                if not isinstance(shape, tuple):
                    # vector
                    shape = (nrows, shape)
                else:
                    # multi-dimensional
                    shape = tuple([nrows] + list(shape))
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
        npy_type, isvar, istbit = self._get_tbl_numpy_dtype(colnum)
        name = self._info['colinfo'][colnum]['name']

        if isvar:
            if vstorage == 'object':
                descr = (name, 'O')
            else:
                tform = self._info['colinfo'][colnum]['tform']
                max_size = _extract_vararray_max(tform)

                if max_size <= 0:
                    name = self._info['colinfo'][colnum]['name']
                    mess = 'Will read as an object field'
                    if max_size < 0:
                        mess = "Column '%s': No maximum size: '%s'. %s"
                        mess = mess % (name, tform, mess)
                        warnings.warn(mess, FITSRuntimeWarning)
                    else:
                        mess = "Column '%s': Max size is zero: '%s'. %s"
                        mess = mess % (name, tform, mess)
                        warnings.warn(mess, FITSRuntimeWarning)

                    # we are forced to read this as an object array
                    return self.get_rec_column_descr(colnum, 'object')

                if npy_type[0] == 'S':
                    # variable length string columns cannot
                    # themselves be arrays I don't think
                    npy_type = 'S%d' % max_size
                    descr = (name, npy_type)
                elif npy_type[0] == 'U':
                    # variable length string columns cannot
                    # themselves be arrays I don't think
                    npy_type = 'U%d' % max_size
                    descr = (name, npy_type)
                else:
                    descr = (name, npy_type, max_size)
        else:
            tdim = self._info['colinfo'][colnum]['tdim']
            shape = _tdim2shape(
                tdim, name,
                is_string=(npy_type[0] == 'S' or npy_type[0] == 'U'))
            if shape is not None:
                descr = (name, npy_type, shape)
            else:
                descr = (name, npy_type)
        return descr, isvar

    def _read_rec_with_var(
            self, colnums, rows, dtype, offsets, isvar, vstorage):
        """
        Read columns from a table into a rec array, including variable length
        columns.  This is special because, for efficiency, it involves reading
        from the main table as normal but skipping the columns in the array
        that are variable.  Then reading the variable length columns, with
        accounting for strides appropriately.

        row and column numbers should be checked before calling this function
        """

        colnumsp = colnums+1
        if rows is None:
            nrows = self._info['nrows']
        else:
            nrows = rows.size
        array = numpy.zeros(nrows, dtype=dtype)

        # read from the main table first
        wnotvar, = numpy.where(isvar == False)  # noqa
        if wnotvar.size > 0:
            # this will be contiguous (not true for slices)
            thesecol = colnumsp[wnotvar]
            theseoff = offsets[wnotvar]
            self._FITS.read_columns_as_rec_byoffset(self._ext+1,
                                                    thesecol,
                                                    theseoff,
                                                    array,
                                                    rows)
            for i in xrange(thesecol.size):

                name = array.dtype.names[wnotvar[i]]
                colnum = thesecol[i]-1
                self._rescale_and_convert_field_inplace(
                    array,
                    name,
                    self._info['colinfo'][colnum]['tscale'],
                    self._info['colinfo'][colnum]['tzero'])

        array = self._maybe_decode_fits_ascii_strings_to_unicode_py3(array)

        # now read the variable length arrays we may be able to speed this up
        # by storing directly instead of reading first into a list
        wvar, = numpy.where(isvar == True)  # noqa
        if wvar.size > 0:
            # this will be contiguous (not true for slices)
            thesecol = colnumsp[wvar]
            for i in xrange(thesecol.size):
                colnump = thesecol[i]
                name = array.dtype.names[wvar[i]]
                dlist = self._FITS.read_var_column_as_list(
                    self._ext+1, colnump, rows)

                if (isinstance(dlist[0], str) or
                        (IS_PY3 and isinstance(dlist[0], bytes))):
                    is_string = True
                else:
                    is_string = False

                if array[name].dtype.descr[0][1][1] == 'O':
                    # storing in object array
                    # get references to each, no copy made
                    for irow, item in enumerate(dlist):
                        if IS_PY3 and isinstance(item, bytes):
                            item = item.decode('ascii')
                        array[name][irow] = item
                else:
                    for irow, item in enumerate(dlist):
                        if IS_PY3 and isinstance(item, bytes):
                            item = item.decode('ascii')

                        if is_string:
                            array[name][irow] = item
                        else:
                            ncopy = len(item)

                            if IS_PY3:
                                ts = array[name].dtype.descr[0][1][1]
                                if ts != 'S' and ts != 'U':
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
            if len(rows) > 0 and (rows[0] < 0 or rows[-1] > maxrow):
                raise ValueError("rows must be in [%d,%d]" % (0, maxrow))
        return rows

    def _process_slice(self, arg):
        """
        process the input slice for use calling the C code
        """
        start = arg.start
        stop = arg.stop
        step = arg.step

        nrows = self._info['nrows']
        if step is None:
            step = 1
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
            stop = nrows
        return slice(start, stop, step)

    def _slice2rows(self, start, stop, step=None):
        """
        Convert a slice to an explicit array of rows
        """
        nrows = self._info['nrows']
        if start is None:
            start = 0
        if stop is None:
            stop = nrows
        if step is None:
            step = 1

        tstart = self._fix_range(start)
        tstop = self._fix_range(stop)
        if tstart == 0 and tstop == nrows and step is None:
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
                num = nrows + (1+num)
            elif num > nrows:
                num = nrows
        else:
            # single element
            if num < 0:
                num = nrows + num
            elif num > (nrows-1):
                num = nrows-1

        return num

    def _rescale_and_convert_field_inplace(self, array, name, scale, zero):
        """
        Apply fits scalings.  Also, convert bool to proper
        numpy boolean values
        """
        self._rescale_array(array[name], scale, zero)
        if array[name].dtype == bool:
            array[name] = self._convert_bool_array(array[name])
        return array

    def _rescale_and_convert(self, array, scale, zero, name=None):
        """
        Apply fits scalings.  Also, convert bool to proper
        numpy boolean values
        """
        self._rescale_array(array, scale, zero)
        if array.dtype == bool:
            array = self._convert_bool_array(array)

        return array

    def _rescale_array(self, array, scale, zero):
        """
        Scale the input array
        """
        if scale != 1.0:
            sval = numpy.array(scale, dtype=array.dtype)
            array *= sval
        if zero != 0.0:
            zval = numpy.array(zero, dtype=array.dtype)
            array += zval

    def _maybe_trim_strings(self, array, trim_strings=False, **keys):
        """
        if requested, trim trailing white space from
        all string fields in the input array
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if self.trim_strings or trim_strings:
            _trim_strings(array)

    def _maybe_decode_fits_ascii_strings_to_unicode_py3(self, array):
        if IS_PY3:
            do_conversion = False
            new_dt = []
            for _dt in array.dtype.descr:
                if 'S' in _dt[1]:
                    do_conversion = True
                    if len(_dt) == 3:
                        new_dt.append((
                            _dt[0],
                            _dt[1].replace('S', 'U').replace('|', ''),
                            _dt[2]))
                    else:
                        new_dt.append((
                            _dt[0],
                            _dt[1].replace('S', 'U').replace('|', '')))
                else:
                    new_dt.append(_dt)
            if do_conversion:
                array = array.astype(new_dt, copy=False)
        return array

    def _convert_bool_array(self, array):
        """
        cfitsio reads as characters 'T' and 'F' -- convert to real boolean
        If input is a fits bool, convert to numpy boolean
        """

        output = (array.view(numpy.int8) == ord('T')).astype(bool)
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

        istbit = False
        if (ftype == 1):
            istbit = True

        isvar = False
        if ftype < 0:
            isvar = True
        if include_endianness:
            # if binary we will read the big endian bytes directly,
            # if ascii we read into native byte order
            if table_type == ASCII_TBL:
                addstr = ''
            else:
                addstr = '>'
            if npy_type not in ['u1', 'i1', 'S', 'U']:
                npy_type = addstr+npy_type

        if npy_type == 'S':
            width = self._info['colinfo'][colnum]['width']
            npy_type = 'S%d' % width
        elif npy_type == 'U':
            width = self._info['colinfo'][colnum]['width']
            npy_type = 'U%d' % width

        return npy_type, isvar, istbit

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

        flags = set()
        #
        if isinstance(arg, (tuple, list, numpy.ndarray)):
            # a sequence was entered
            if isstring(arg[0]):
                result = arg
            else:
                result = arg
                flags.add('isrows')
        elif isstring(arg):
            # a single string was entered
            result = arg
        elif isinstance(arg, slice):
            if unpack:
                flags.add('isrows')
                result = self._slice2rows(arg.start, arg.stop, arg.step)
            else:
                flags.add('isrows')
                flags.add('isslice')
                result = self._process_slice(arg)
        else:
            # a single object was entered.
            # Probably should apply some more checking on this
            result = arg
            flags.add('isrows')
            if numpy.ndim(arg) == 0:
                flags.add('isscalar')

        return result, flags

    def _read_var_column(self, colnum, rows, vstorage):
        """

        first read as a list of arrays, then copy into either a fixed length
        array or an array of objects, depending on vstorage.

        """

        if IS_PY3:
            stype = bytes
        else:
            stype = str

        dlist = self._FITS.read_var_column_as_list(self._ext+1, colnum+1, rows)

        if vstorage == 'fixed':
            tform = self._info['colinfo'][colnum]['tform']
            max_size = _extract_vararray_max(tform)

            if max_size <= 0:
                name = self._info['colinfo'][colnum]['name']
                mess = 'Will read as an object field'
                if max_size < 0:
                    mess = "Column '%s': No maximum size: '%s'. %s"
                    mess = mess % (name, tform, mess)
                    warnings.warn(mess, FITSRuntimeWarning)
                else:
                    mess = "Column '%s': Max size is zero: '%s'. %s"
                    mess = mess % (name, tform, mess)
                    warnings.warn(mess, FITSRuntimeWarning)

                # we are forced to read this as an object array
                return self._read_var_column(colnum, rows, 'object')

            if isinstance(dlist[0], stype):
                descr = 'S%d' % max_size
                array = numpy.fromiter(dlist, descr)
                if IS_PY3:
                    array = array.astype('U', copy=False)
            else:
                descr = dlist[0].dtype.str
                array = numpy.zeros((len(dlist), max_size), dtype=descr)

                for irow, item in enumerate(dlist):
                    ncopy = len(item)
                    array[irow, 0:ncopy] = item[:]
        else:
            array = numpy.zeros(len(dlist), dtype='O')
            for irow, item in enumerate(dlist):
                if IS_PY3 and isinstance(item, bytes):
                    item = item.decode('ascii')
                array[irow] = item

        return array

    def _extract_colnums(self, columns=None):
        """
        Extract an array of columns from the input
        """
        if columns is None:
            return numpy.arange(self._ncol, dtype='i8')

        if not isinstance(columns, (tuple, list, numpy.ndarray)):
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
                raise ValueError(
                    "column number should be in [0,%d]" % (0, self._ncol-1))
        else:
            colstr = mks(col)
            try:
                if self.case_sensitive:
                    mess = "column name '%s' not found (case sensitive)" % col
                    colnum = self._colnames.index(colstr)
                else:
                    mess \
                        = "column name '%s' not found (case insensitive)" % col
                    colnum = self._colnames_lower.index(colstr.lower())
            except ValueError:
                raise ValueError(mess)
        return int(colnum)

    def _update_info(self):
        """
        Call parent method and make sure this is in fact a
        table HDU.  Set some convenience data.
        """
        super(TableHDU, self)._update_info()
        if self._info['hdutype'] == IMAGE_HDU:
            mess = "Extension %s is not a Table HDU" % self.ext
            raise ValueError(mess)
        if 'colinfo' in self._info:
            self._colnames = [i['name'] for i in self._info['colinfo']]
            self._colnames_lower = [
                i['name'].lower() for i in self._info['colinfo']]
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
            fits[ext][2]   # returns a scalar
            fits[ext][2:5]
            fits[ext][200:235:2]
            fits[ext][rows]
            fits[ext][cols][rows]

        Note data are only read once the rows are specified.

        Note you can only read variable length arrays the default way,
        using this function, so set it as you want on construction.

        This function is used for ascii tables as well
        """

        res, flags = \
            self._process_args_as_rows_or_columns(arg)

        if 'isrows' in flags:
            # rows were entered: read all columns
            if 'isslice' in flags:
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

        self._maybe_trim_strings(array)

        if 'isscalar' in flags:
            assert array.shape[0] == 1
            array = array[0]
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
        self._iter_row = 0

        # for iterating we must assume the number of rows will not change
        self._iter_nrows = self.get_nrows()

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

    __next__ = next

    def _get_next_buffered_row(self):
        """
        Get the next row for iteration.
        """
        if self._iter_row == self._iter_nrows:
            raise StopIteration

        if self._row_buffer_index >= self._iter_row_buffer:
            self._buffer_iter_rows(self._iter_row)

        data = self._row_buffer[self._row_buffer_index]
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

        text.append('%srows: %d' % (spacing, self._info['nrows']))
        text.append('%scolumn info:' % spacing)

        cspacing = ' '*4
        nspace = 4
        nname = 15
        ntype = 6
        format = cspacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
        pformat = (
            cspacing + "%-" +
            str(nname) + "s\n %" +
            str(nspace+nname+ntype) + "s  %s")

        for colnum, c in enumerate(self._info['colinfo']):
            if len(c['name']) > nname:
                f = pformat
            else:
                f = format

            dt, isvar, istbit = self._get_tbl_numpy_dtype(
                colnum, include_endianness=False)
            if isvar:
                tform = self._info['colinfo'][colnum]['tform']
                if dt[0] == 'S':
                    dt = 'S0'
                    dimstr = 'vstring[%d]' % _extract_vararray_max(tform)
                else:
                    dimstr = 'varray[%s]' % _extract_vararray_max(tform)
            else:
                if dt[0] == 'S':
                    is_string = True
                else:
                    is_string = False
                dimstr = _get_col_dimstr(c['tdim'], is_string=is_string)

            s = f % (c['name'], dt, dimstr)
            text.append(s)

        text = '\n'.join(text)
        return text


class AsciiTableHDU(TableHDU):
    def read(self, rows=None, columns=None, vstorage=None,
             upper=False, lower=False, trim_strings=False, **keys):
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
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        # if columns is None, returns all.  Guaranteed to be unique and sorted
        colnums = self._extract_colnums(columns)
        if isinstance(colnums, int):
            # scalar sent, don't read as a recarray
            return self.read_column(
                columns, rows=rows, vstorage=vstorage,
                upper=upper, lower=lower, trim_strings=trim_strings)

        rows = self._extract_rows(rows)
        if rows is None:
            nrows = self._info['nrows']
        else:
            nrows = rows.size

        # if rows is None still returns None, and is correctly interpreted
        # by the reader to mean all
        rows = self._extract_rows(rows)

        # this is the full dtype for all columns
        dtype, offsets, isvar = self.get_rec_dtype(
            colnums=colnums, vstorage=vstorage)
        array = numpy.zeros(nrows, dtype=dtype)

        # note reading into existing data
        wnotvar, = numpy.where(isvar == False)  # noqa
        if wnotvar.size > 0:
            for i in wnotvar:
                colnum = colnums[i]
                name = array.dtype.names[i]
                a = array[name].copy()
                self._FITS.read_column(self._ext+1, colnum+1, a, rows)
                array[name] = a
                del a

        array = self._maybe_decode_fits_ascii_strings_to_unicode_py3(array)

        wvar, = numpy.where(isvar == True)  # noqa
        if wvar.size > 0:
            for i in wvar:
                colnum = colnums[i]
                name = array.dtype.names[i]
                dlist = self._FITS.read_var_column_as_list(
                    self._ext+1, colnum+1, rows)
                if (isinstance(dlist[0], str) or
                        (IS_PY3 and isinstance(dlist[0], bytes))):
                    is_string = True
                else:
                    is_string = False

                if array[name].dtype.descr[0][1][1] == 'O':
                    # storing in object array
                    # get references to each, no copy made
                    for irow, item in enumerate(dlist):
                        if IS_PY3 and isinstance(item, bytes):
                            item = item.decode('ascii')
                        array[name][irow] = item
                else:
                    for irow, item in enumerate(dlist):
                        if IS_PY3 and isinstance(item, bytes):
                            item = item.decode('ascii')
                        if is_string:
                            array[name][irow] = item
                        else:
                            ncopy = len(item)
                            array[name][irow][0:ncopy] = item[:]

        if self.lower or lower:
            _names_to_lower_if_recarray(array)
        elif self.upper or upper:
            _names_to_upper_if_recarray(array)

        self._maybe_trim_strings(array, trim_strings=trim_strings)

        return array

    read_ascii = read


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
        Input is the FITS instance and a list of column names.
        """

        self.columns = columns
        if isstring(columns) or isinteger(columns):
            # this is to check if it exists
            self.colnums = [fitshdu._extract_colnum(columns)]

            self.is_scalar = True
            self.columns_list = [columns]
        else:
            # this is to check if it exists
            self.colnums = fitshdu._extract_colnums(columns)

            self.is_scalar = False
            self.columns_list = columns

        self.fitshdu = fitshdu

    def read(self, columns=None, rows=None, vstorage=None, lower=False,
             upper=False, trim_strings=False, **keys):
        """
        Read the data from disk and return as a numpy array

        parameters
        ----------
        columns: list/array, optional
            An optional set of columns to read from table HDUs.  Can be string
            or number. If a sequence, a recarray is always returned.  If a
            scalar, an ordinary array is returned.
        rows: optional
            An optional list of rows to read from table HDUS.  Default is to
            read all.
        vstorage: string, optional
            Over-ride the default method to store variable length columns. Can
            be 'fixed' or 'object'. See docs on fitsio.FITS for details.
        lower: bool, optional
            If True, force all columns names to lower case in output. Will over
            ride the lower= keyword from construction.
        upper: bool, optional
            If True, force all columns names to upper case in output. Will over
            ride the lower= keyword from construction.
        trim_strings: bool, optional
            If True, trim trailing spaces from strings. Will over-ride the
            trim_strings= keyword from constructor.
        """
        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if self.is_scalar:
            data = self.fitshdu.read_column(
                self.columns,
                rows=rows, vstorage=vstorage, lower=lower, upper=upper,
                trim_strings=trim_strings)
        else:
            if columns is None:
                c = self.columns
            else:
                c = columns
            data = self.fitshdu.read(
                columns=c,
                rows=rows, vstorage=vstorage, lower=lower, upper=upper,
                trim_strings=trim_strings)

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

        res, flags = \
            self.fitshdu._process_args_as_rows_or_columns(arg, unpack=True)
        if 'isrows' in flags:
            # rows was entered: read all current column subset
            array = self.read(rows=res)
            if 'isscalar' in flags:
                assert array.shape[0] == 1
                array = array[0]
            return array
        else:
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
        text.append("%sfile: %s" % (spacing, hdu._filename))
        text.append("%sextension: %d" % (spacing, info['hdunum']-1))
        text.append("%stype: %s" % (spacing, _hdu_type_map[info['hdutype']]))
        text.append('%srows: %d' % (spacing, info['nrows']))
        text.append("%scolumn subset:" % spacing)

        cspacing = ' '*4
        nspace = 4
        nname = 15
        ntype = 6
        format = cspacing + "%-" + str(nname) + "s %" + str(ntype) + "s  %s"
        pformat = (
            cspacing + "%-" + str(nname) + "s\n %" +
            str(nspace+nname+ntype) + "s  %s")

        for colnum in self.colnums:
            cinfo = colinfo[colnum]

            if len(cinfo['name']) > nname:
                f = pformat
            else:
                f = format

            dt, isvar, istbit = hdu._get_tbl_numpy_dtype(
                colnum, include_endianness=False)
            if isvar:
                tform = cinfo['tform']
                if dt[0] == 'S':
                    dt = 'S0'
                    dimstr = 'vstring[%d]' % _extract_vararray_max(tform)
                else:
                    dimstr = 'varray[%s]' % _extract_vararray_max(tform)
            else:
                dimstr = _get_col_dimstr(cinfo['tdim'])

            s = f % (cinfo['name'], dt, dimstr)
            text.append(s)

        s = "\n".join(text)
        return s


def _tdim2shape(tdim, name, is_string=False):
    shape = None
    if tdim is None:
        raise ValueError("field '%s' has malformed TDIM" % name)

    if len(tdim) > 1 or tdim[0] > 1:
        if is_string:
            shape = list(reversed(tdim[1:]))
        else:
            shape = list(reversed(tdim))

        if len(shape) == 1:
            shape = shape[0]
        else:
            shape = tuple(shape)

    return shape


def _names_to_lower_if_recarray(data):
    if data.dtype.names is not None:
        data.dtype.names = [n.lower() for n in data.dtype.names]


def _names_to_upper_if_recarray(data):
    if data.dtype.names is not None:
        data.dtype.names = [n.upper() for n in data.dtype.names]


def _trim_strings(data):
    names = data.dtype.names
    if names is not None:
        # run through each field separately
        for n in names:
            if data[n].dtype.descr[0][1][1] in ['S', 'U']:
                data[n] = numpy.char.rstrip(data[n])
    else:
        if data.dtype.descr[0][1][1] in ['S', 'U']:
            data[:] = numpy.char.rstrip(data[:])


def _extract_vararray_max(tform):
    """
    Extract number from PX(number)
    """
    first = tform.find('(')
    last = tform.rfind(')')

    if first == -1 or last == -1:
        # no max length specified
        return -1

    maxnum = int(tform[first+1:last])
    return maxnum


def _get_col_dimstr(tdim, is_string=False):
    """
    not for variable length
    """
    dimstr = ''
    if tdim is None:
        dimstr = 'array[bad TDIM]'
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


# no support yet for complex
# all strings are read as bytes for python3 and then decoded to unicode
_table_fits2npy = {1: 'i1',
                   11: 'u1',
                   12: 'i1',
                   # logical. Note pyfits uses this for i1,
                   # cfitsio casts to char*
                   14: 'b1',
                   16: 'S',
                   20: 'u2',
                   21: 'i2',
                   30: 'u4',  # 30=TUINT
                   31: 'i4',  # 31=TINT
                   40: 'u4',  # 40=TULONG
                   41: 'i4',  # 41=TLONG
                   42: 'f4',
                   81: 'i8',
                   82: 'f8',
                   83: 'c8',   # TCOMPLEX
                   163: 'c16'}  # TDBLCOMPLEX

# cfitsio returns only types f8, i4 and strings for column types. in order to
# avoid data loss, we always use i8 for integer types
# all strings are read as bytes for python3 and then decoded to unicode
_table_fits2npy_ascii = {16: 'S',
                         31: 'i8',  # listed as TINT, reading as i8
                         41: 'i8',  # listed as TLONG, reading as i8
                         81: 'i8',
                         21: 'i4',  # listed as TSHORT, reading as i4
                         42: 'f8',  # listed as TFLOAT, reading as f8
                         82: 'f8'}

# for TFORM
_table_npy2fits_form = {'b1': 'L',
                        'u1': 'B',
                        'i1': 'S',  # gets converted to unsigned
                        'S': 'A',
                        'U': 'A',
                        'u2': 'U',  # gets converted to signed
                        'i2': 'I',
                        'u4': 'V',  # gets converted to signed
                        'i4': 'J',
                        'i8': 'K',
                        'f4': 'E',
                        'f8': 'D',
                        'c8': 'C',
                        'c16': 'M'}

# from mrdfits; note G gets turned into E
# types=  ['A',   'I',   'L',   'B',   'F',    'D',      'C',     'M',     'K']
# formats=['A1',  'I6',  'I10', 'I4',  'G15.9','G23.17', 'G15.9', 'G23.17',
#          'I20']

_table_npy2fits_form_ascii = {'S': 'A1',       # Need to add max here
                              'U': 'A1',       # Need to add max here
                              'i2': 'I7',      # I
                              'i4': 'I12',     # ??
                              # 'i8':'I21',     # K # i8 aren't supported
                              # 'f4':'E15.7',   # F
                              # F We must write as f8 since we can only
                              # read as f8
                              'f4': 'E26.17',
                              # D 25.16 looks right, but this is recommended
                              'f8': 'E26.17'}


def _npy2fits(d, table_type='binary', write_bitcols=False):
    """
    d is the full element from the descr
    """
    npy_dtype = d[1][1:]
    if npy_dtype[0] == 'S' or npy_dtype[0] == 'U':
        name, form, dim = _npy_string2fits(d, table_type=table_type)
    else:
        name, form, dim = _npy_num2fits(
            d, table_type=table_type, write_bitcols=write_bitcols)

    return name, form, dim


def _npy_num2fits(d, table_type='binary', write_bitcols=False):
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
    if npy_dtype[0] == 'S' or npy_dtype[0] == 'U':
        raise ValueError("got S or U type: use _npy_string2fits")

    if npy_dtype not in _table_npy2fits_form:
        raise ValueError("unsupported type '%s'" % npy_dtype)

    if table_type == 'binary':
        form = _table_npy2fits_form[npy_dtype]
    else:
        form = _table_npy2fits_form_ascii[npy_dtype]

    # now the dimensions
    if len(d) > 2:
        if table_type == 'ascii':
            raise ValueError(
                "Ascii table columns must be scalar, got %s" % str(d))

        if write_bitcols and npy_dtype == 'b1':
            # multi-dimensional boolean
            form = 'X'

        # Note, depending on numpy version, even 1-d can be a tuple
        if isinstance(d[2], tuple):
            count = reduce(lambda x, y: x*y, d[2])
            form = '%d%s' % (count, form)

            if len(d[2]) > 1:
                # this is multi-dimensional array column.  the form
                # should be total elements followed by A
                dim = list(reversed(d[2]))
                dim = [str(e) for e in dim]
                dim = '(' + ','.join(dim)+')'
        else:
            # this is a vector (1d array) column
            count = d[2]
            form = '%d%s' % (count, form)

    return name, form, dim


def _npy_string2fits(d, table_type='binary'):
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
    if npy_dtype[0] != 'S' and npy_dtype[0] != 'U':
        raise ValueError("expected S or U type, got %s" % npy_dtype[0])

    # get the size of each string
    string_size_str = npy_dtype[1:]
    string_size = int(string_size_str)

    if string_size <= 0:
        raise ValueError('string sizes must be > 0, '
                         'got %s for field %s' % (npy_dtype, name))

    # now the dimensions
    if len(d) == 2:
        if table_type == 'ascii':
            form = 'A'+string_size_str
        else:
            form = string_size_str+'A'
    else:
        if table_type == 'ascii':
            raise ValueError(
                "Ascii table columns must be scalar, got %s" % str(d))
        if isinstance(d[2], tuple):
            # this is an array column.  the form
            # should be total elements followed by A
            # count = 1
            # count = [count*el for el in d[2]]
            count = reduce(lambda x, y: x*y, d[2])
            count = string_size*count
            form = '%dA' % count

            # will have to do tests to see if this is the right order
            dim = list(reversed(d[2]))
            # dim = d[2]
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
