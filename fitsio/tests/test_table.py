import pytest
import numpy as np
import os
import tempfile
from .checks import (
    compare_names,
    compare_array,
    compare_array_tol,
    compare_object_array,
    compare_rec,
    compare_headerlist_header,
    compare_rec_with_var,
    compare_rec_subrows,
)
from .makedata import make_data
from ..fitslib import FITS, write, read
from .. import util

DTYPES = ['u1', 'i1', 'u2', 'i2', '<u4', 'i4', 'i8', '>f4', 'f8']


def test_table_read_write():

    adata = make_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(
                adata['data'], header=adata['keys'], extname='mytable'
            )

            d = fits[1].read()
            compare_rec(adata['data'], d, "table read/write")

            h = fits[1].read_header()
            compare_headerlist_header(adata['keys'], h)

        # see if our convenience functions are working
        write(
            fname,
            adata['data2'],
            extname="newext",
            header={'ra': 335.2, 'dec': -25.2},
        )
        d = read(fname, ext='newext')
        compare_rec(adata['data2'], d, "table data2")

        # now test read_column
        with FITS(fname) as fits:

            for f in adata['data'].dtype.names:
                d = fits[1].read_column(f)
                compare_array(
                    adata['data'][f], d, "table 1 single field read '%s'" % f
                )

            for f in adata['data2'].dtype.names:
                d = fits['newext'].read_column(f)
                compare_array(
                    adata['data2'][f], d, "table 2 single field read '%s'" % f
                )

            # now list of columns
            for cols in [['u2scalar', 'f4vec', 'Sarr'],
                         ['f8scalar', 'u2arr', 'Sscalar']]:
                d = fits[1].read(columns=cols)
                for f in d.dtype.names:
                    compare_array(
                        adata['data'][f][:], d[f], "test column list %s" % f
                    )

                for rows in [[1, 3], [3, 1], [2, 2, 1]]:
                    d = fits[1].read(columns=cols, rows=rows)
                    for col in d.dtype.names:
                        compare_array(
                            adata['data'][col][rows], d[col],
                            "test column list %s row subset" % col
                        )
                    for col in cols:
                        d = fits[1].read_column(col, rows=rows)
                        compare_array(
                            adata['data'][col][rows], d,
                            "test column list %s row subset" % col
                        )


@pytest.mark.parametrize('nvec', [2, 1])
def test_table_read_write_vec1(nvec):
    """
    ensure the data for vec length 1 gets round-tripped, even though
    the shape is not preserved
    """
    dtype = [('x', 'f4', (nvec,))]
    num = 10
    data = np.zeros(num, dtype=dtype)
    data['x'] = np.arange(num * nvec).reshape(num, nvec)
    assert data['x'].shape == (num, nvec)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(data)

            d = fits[1].read()
            if nvec == 1:
                assert d['x'].shape == (num,)
            compare_array(
                data['x'].ravel(), d['x'].ravel(),
                "table single field read 'x'"
            )

        # see if our convenience functions are working
        write(
            fname,
            data,
            extname="newext",
        )
        d = read(fname, ext='newext')
        if nvec == 1:
            assert d['x'].shape == (num,)
        compare_array(data['x'].ravel(), d['x'].ravel(), "table data2")

        # now test read_column
        with FITS(fname) as fits:

            d = fits[1].read_column('x')
            if nvec == 1:
                assert d.shape == (num,)
            compare_array(
                data['x'].ravel(), d.ravel(),
                "table single field read 'x'"
            )


@pytest.mark.parametrize('nvec', [2, 1])
def test_table_read_write_uvec1(nvec):
    """
    ensure the data for U string vec length 1 gets round-tripped, even though
    the shape is not preserved.  Also test 2 for consistency
    """

    dtype = [('string', 'U10', (nvec,))]
    num = 10
    data = np.zeros(num, dtype=dtype)
    sravel = data['string'].ravel()
    sravel[:] = ['%-10s' % i for i in range(num * nvec)]
    assert data['string'].shape == (num, nvec)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(data)

            d = fits[1].read()

            if nvec == 1:
                assert d['string'].shape == (num,)

            compare_array(
                data['string'].ravel(), d['string'].ravel(),
                "table single field read 'string'"
            )

        # see if our convenience functions are working
        write(
            fname,
            data,
            extname="newext",
        )
        d = read(fname, ext='newext')

        if nvec == 1:
            assert d['string'].shape == (num,)
        compare_array(
            data['string'].ravel(), d['string'].ravel(), "table data2",
        )

        # now test read_column
        with FITS(fname) as fits:

            d = fits[1].read_column('string')

            if nvec == 1:
                assert d.shape == (num,)
            compare_array(
                data['string'].ravel(), d.ravel(),
                "table single field read 'string'"
            )


def test_table_column_index_scalar():
    """
    Test a basic table write, data and a header, then reading back in to
    check the values
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.empty(1, dtype=[('Z', 'f8')])
            data['Z'][:] = 1.0
            fits.write_table(data)
            fits.write_table(data)

        with FITS(fname, 'r') as fits:
            assert fits[1]['Z'][0].ndim == 0
            assert fits[1][0].ndim == 0


def test_table_read_empty_rows():
    """
    test reading empty list of rows from an table.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.empty(1, dtype=[('Z', 'f8')])
            data['Z'][:] = 1.0
            fits.write_table(data)
            fits.write_table(data)

        with FITS(fname, 'r') as fits:
            assert len(fits[1].read(rows=[])) == 0
            assert len(fits[1].read(rows=range(0, 0))) == 0
            assert len(fits[1].read(rows=np.arange(0, 0))) == 0


def test_table_format_column_subset():
    """
    Test a basic table write, data and a header, then reading back in to
    check the values
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.empty(1, dtype=[('Z', 'f8'), ('Z_PERSON', 'f8')])
            data['Z'][:] = 1.0
            data['Z_PERSON'][:] = 1.0
            fits.write_table(data)
            fits.write_table(data)
            fits.write_table(data)

        with FITS(fname, 'r') as fits:
            # assert we do not have an extra row of 'Z'
            sz = str(fits[2]['Z_PERSON']).split('\n')
            s = str(fits[2][('Z_PERSON', 'Z')]).split('\n')
            assert len(sz) == len(s) - 1


def test_table_write_dict_of_arrays_scratch():

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            d = {}
            for n in data.dtype.names:
                d[n] = data[n]

            fits.write(d)

        d = read(fname)
        compare_rec(data, d, "list of dicts, scratch")


def test_table_write_dict_of_arrays():

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.create_table_hdu(data, extname='mytable')

            d = {}
            for n in data.dtype.names:
                d[n] = data[n]

            fits[-1].write(d)

        d = read(fname)
        compare_rec(data, d, "list of dicts")


def test_table_write_dict_of_arrays_var():
    """
    This version creating the table from a dict of arrays, variable
    lenght columns
    """

    adata = make_data()
    vardata = adata['vardata']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            d = {}
            for n in vardata.dtype.names:
                d[n] = vardata[n]

            fits.write(d)

        d = read(fname)
        compare_rec_with_var(vardata, d, "dict of arrays, var")


def test_table_write_list_of_arrays_scratch():
    """
    This version creating the table from the names and list, creating
    table first
    """

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            names = [n for n in data.dtype.names]
            dlist = [data[n] for n in data.dtype.names]
            fits.write(dlist, names=names)

        d = read(fname)
        compare_rec(data, d, "list of arrays, scratch")


def test_table_write_list_of_arrays():

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.create_table_hdu(data, extname='mytable')

            columns = [n for n in data.dtype.names]
            dlist = [data[n] for n in data.dtype.names]
            fits[-1].write(dlist, columns=columns)

        d = read(fname, ext='mytable')
        compare_rec(data, d, "list of arrays")


def test_table_write_list_of_arrays_var():
    """
    This version creating the table from the names and list, variable
    lenght cols
    """
    adata = make_data()
    vardata = adata['vardata']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            names = [n for n in vardata.dtype.names]
            dlist = [vardata[n] for n in vardata.dtype.names]
            fits.write(dlist, names=names)

        d = read(fname)
        compare_rec_with_var(vardata, d, "list of arrays, var")


def test_table_write_bad_string():

    for d in ['S0', 'U0']:
        dt = [('s', d)]

        # old numpy didn't allow this dtype, so will throw
        # a TypeError for empty dtype
        try:
            data = np.zeros(1, dtype=dt)
            supported = True
        except TypeError:
            supported = False

        if supported:
            with pytest.raises(ValueError):
                with tempfile.TemporaryDirectory() as tmpdir:
                    fname = os.path.join(tmpdir, 'test.fits')
                    with FITS(fname, 'rw') as fits:
                        fits.write(data)


def test_variable_length_columns():

    adata = make_data()
    vardata = adata['vardata']

    for vstorage in ['fixed', 'object']:
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.fits')

            with FITS(fname, 'rw', vstorage=vstorage) as fits:
                fits.write(vardata)

                # reading multiple columns
                d = fits[1].read()
                compare_rec_with_var(
                    vardata, d, "read all test '%s'" % vstorage
                )

                cols = ['u2scalar', 'Sobj']
                d = fits[1].read(columns=cols)
                compare_rec_with_var(
                    vardata, d, "read all test subcols '%s'" % vstorage
                )

                # one at a time
                for f in vardata.dtype.names:
                    d = fits[1].read_column(f)
                    if util.is_object(vardata[f]):
                        compare_object_array(
                            vardata[f], d,
                            "read all field '%s'" % f
                        )

                # same as above with slices
                # reading multiple columns
                d = fits[1][:]
                compare_rec_with_var(
                    vardata, d, "read all test '%s'" % vstorage
                )

                d = fits[1][cols][:]
                compare_rec_with_var(
                    vardata, d, "read all test subcols '%s'" % vstorage
                )

                # one at a time
                for f in vardata.dtype.names:
                    d = fits[1][f][:]
                    if util.is_object(vardata[f]):
                        compare_object_array(
                            vardata[f], d,
                            "read all field '%s'" % f
                        )

                #
                # now same with sub rows
                #

                # reading multiple columns, sorted and unsorted
                for rows in [[0, 2], [2, 0]]:
                    d = fits[1].read(rows=rows)
                    compare_rec_with_var(
                        vardata, d, "read subrows test '%s'" % vstorage,
                        rows=rows,
                    )

                    d = fits[1].read(columns=cols, rows=rows)
                    compare_rec_with_var(
                        vardata,
                        d,
                        "read subrows test subcols '%s'" % vstorage,
                        rows=rows,
                    )

                    # one at a time
                    for f in vardata.dtype.names:
                        d = fits[1].read_column(f, rows=rows)
                        if util.is_object(vardata[f]):
                            compare_object_array(
                                vardata[f], d,
                                "read subrows field '%s'" % f,
                                rows=rows,
                            )

                    # same as above with slices
                    # reading multiple columns
                    d = fits[1][rows]
                    compare_rec_with_var(
                        vardata, d, "read subrows slice test '%s'" % vstorage,
                        rows=rows,
                    )
                    d = fits[1][2:4]
                    compare_rec_with_var(
                        vardata,
                        d,
                        "read slice test '%s'" % vstorage,
                        rows=[2, 3],
                    )

                    d = fits[1][cols][rows]
                    compare_rec_with_var(
                        vardata,
                        d,
                        "read subcols subrows slice test '%s'" % vstorage,
                        rows=rows,
                    )

                    d = fits[1][cols][2:4]

                    compare_rec_with_var(
                        vardata,
                        d,
                        "read subcols slice test '%s'" % vstorage,
                        rows=[2, 3],
                    )

                    # one at a time
                    for f in vardata.dtype.names:
                        d = fits[1][f][rows]
                        if util.is_object(vardata[f]):
                            compare_object_array(
                                vardata[f], d,
                                "read subrows field '%s'" % f,
                                rows=rows,
                            )
                        d = fits[1][f][2:4]
                        if util.is_object(vardata[f]):
                            compare_object_array(
                                vardata[f], d,
                                "read slice field '%s'" % f,
                                rows=[2, 3],
                            )


def test_table_iter():
    """
    Test iterating over rows of a table
    """

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(
                data,
                header=adata['keys'],
                extname='mytable'
            )

        # one row at a time
        with FITS(fname) as fits:
            hdu = fits["mytable"]
            i = 0
            for row_data in hdu:
                compare_rec(data[i], row_data, "table data")
                i += 1


def test_ascii_table_write_read():
    """
    Test write and read for an ascii table
    """

    adata = make_data()
    ascii_data = adata['ascii_data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.write_table(
                ascii_data,
                table_type='ascii',
                header=adata['keys'],
                extname='mytable',
            )

            # cfitsio always reports type as i4 and f8, period, even if if
            # written with higher precision.  Need to fix that somehow
            for f in ascii_data.dtype.names:
                d = fits[1].read_column(f)
                if d.dtype == np.float64:
                    # note we should be able to do 1.11e-16 in principle, but
                    # in practice we get more like 2.15e-16
                    compare_array_tol(
                        ascii_data[f], d, 2.15e-16, "table field read '%s'" % f
                    )
                else:
                    compare_array(
                        ascii_data[f], d, "table field read '%s'" % f
                    )

            for rows in [[1, 3], [3, 1]]:
                for f in ascii_data.dtype.names:
                    d = fits[1].read_column(f, rows=rows)
                    if d.dtype == np.float64:
                        compare_array_tol(ascii_data[f][rows], d, 2.15e-16,
                                          "table field read subrows '%s'" % f)
                    else:
                        compare_array(ascii_data[f][rows], d,
                                      "table field read subrows '%s'" % f)

            beg = 1
            end = 3
            for f in ascii_data.dtype.names:
                d = fits[1][f][beg:end]
                if d.dtype == np.float64:
                    compare_array_tol(ascii_data[f][beg:end], d, 2.15e-16,
                                      "table field read slice '%s'" % f)
                else:
                    compare_array(ascii_data[f][beg:end], d,
                                  "table field read slice '%s'" % f)

            cols = ['i2scalar', 'f4scalar']
            for f in ascii_data.dtype.names:
                data = fits[1].read(columns=cols)
                for f in data.dtype.names:
                    d = data[f]
                    if d.dtype == np.float64:
                        compare_array_tol(
                            ascii_data[f],
                            d,
                            2.15e-16,
                            "table subcol, '%s'" % f
                        )
                    else:
                        compare_array(
                            ascii_data[f], d, "table subcol, '%s'" % f
                        )

                data = fits[1][cols][:]
                for f in data.dtype.names:
                    d = data[f]
                    if d.dtype == np.float64:
                        compare_array_tol(
                            ascii_data[f],
                            d,
                            2.15e-16,
                            "table subcol, '%s'" % f
                        )
                    else:
                        compare_array(
                            ascii_data[f], d, "table subcol, '%s'" % f
                        )

            for rows in [[1, 3], [3, 1]]:
                for f in ascii_data.dtype.names:
                    data = fits[1].read(columns=cols, rows=rows)
                    for f in data.dtype.names:
                        d = data[f]
                        if d.dtype == np.float64:
                            compare_array_tol(ascii_data[f][rows], d, 2.15e-16,
                                              "table subcol, '%s'" % f)
                        else:
                            compare_array(ascii_data[f][rows], d,
                                          "table subcol, '%s'" % f)

                    data = fits[1][cols][rows]
                    for f in data.dtype.names:
                        d = data[f]
                        if d.dtype == np.float64:
                            compare_array_tol(ascii_data[f][rows], d, 2.15e-16,
                                              "table subcol/row, '%s'" % f)
                        else:
                            compare_array(ascii_data[f][rows], d,
                                          "table subcol/row, '%s'" % f)

            for f in ascii_data.dtype.names:

                data = fits[1][cols][beg:end]
                for f in data.dtype.names:
                    d = data[f]
                    if d.dtype == np.float64:
                        compare_array_tol(ascii_data[f][beg:end], d, 2.15e-16,
                                          "table subcol/slice, '%s'" % f)
                    else:
                        compare_array(ascii_data[f][beg:end], d,
                                      "table subcol/slice, '%s'" % f)


def test_table_insert_column():
    """
    Insert a new column
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.write_table(data, header=adata['keys'], extname='mytable')

            d = fits[1].read()

            for n in d.dtype.names:
                newname = n+'_insert'

                fits[1].insert_column(newname, d[n])

                newdata = fits[1][newname][:]

                compare_array(
                    d[n],
                    newdata,
                    "table single field insert and read '%s'" % n
                )


def test_table_delete_row_range():
    """
    Test deleting a range of rows using the delete_rows method
    """

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(data)

        rowslice = slice(1, 3)
        with FITS(fname, 'rw') as fits:
            fits[1].delete_rows(rowslice)

        with FITS(fname) as fits:
            d = fits[1].read()

        compare_data = data[[0, 3]]
        compare_rec(compare_data, d, "delete row range")


def test_table_delete_rows():
    """
    Test deleting specific set of rows using the delete_rows method
    """

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(data)

        rows2delete = [1, 3]
        with FITS(fname, 'rw') as fits:
            fits[1].delete_rows(rows2delete)

        with FITS(fname) as fits:
            d = fits[1].read()

        compare_data = data[[0, 2]]
        compare_rec(compare_data, d, "delete rows")


def test_table_where():
    """
    Use the where method to get indices for a row filter expression
    """

    adata = make_data()
    data2 = adata['data2']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(data2)

        #
        # get all indices
        #
        with FITS(fname) as fits:
            a = fits[1].where('x > 3 && y < 8')
        b = np.where((data2['x'] > 3) & (data2['y'] < 8))[0]
        np.testing.assert_array_equal(a, b)

        #
        # get slice of indices
        #
        with FITS(fname) as fits:
            a = fits[1].where('x > 3 && y < 8', 2, 8)
        b = np.where((data2['x'][2:8] > 3) & (data2['y'][2:8] < 8))[0]
        np.testing.assert_array_equal(a, b)


def test_table_resize():
    """
    Use the resize method to change the size of a table

    default values get filled in and these are tested
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        #
        # shrink from back
        #
        with FITS(fname, 'rw', clobber=True) as fits:
            fits.write_table(data)

        nrows = 2
        with FITS(fname, 'rw') as fits:
            fits[1].resize(nrows)

        with FITS(fname) as fits:
            d = fits[1].read()

        compare_data = data[0:nrows]
        compare_rec(compare_data, d, "shrink from back")

        #
        # shrink from front
        #
        with FITS(fname, 'rw', clobber=True) as fits:
            fits.write_table(data)

        with FITS(fname, 'rw') as fits:
            fits[1].resize(nrows, front=True)

        with FITS(fname) as fits:
            d = fits[1].read()

        compare_data = data[nrows-data.size:]
        compare_rec(compare_data, d, "shrink from front")

        # These don't get zerod

        nrows = 10
        add_data = np.zeros(nrows-data.size, dtype=data.dtype)
        add_data['i1scalar'] = -128
        add_data['i1vec'] = -128
        add_data['i1arr'] = -128
        add_data['u2scalar'] = 32768
        add_data['u2vec'] = 32768
        add_data['u2arr'] = 32768
        add_data['u4scalar'] = 2147483648
        add_data['u4vec'] = 2147483648
        add_data['u4arr'] = 2147483648

        #
        # expand at the back
        #
        with FITS(fname, 'rw', clobber=True) as fits:
            fits.write_table(data)
        with FITS(fname, 'rw') as fits:
            fits[1].resize(nrows)

        with FITS(fname) as fits:
            d = fits[1].read()

        compare_data = np.hstack((data, add_data))
        compare_rec(compare_data, d, "expand at the back")

        #
        # expand at the front
        #
        with FITS(fname, 'rw', clobber=True) as fits:
            fits.write_table(data)
        with FITS(fname, 'rw') as fits:
            fits[1].resize(nrows, front=True)

        with FITS(fname) as fits:
            d = fits[1].read()

        compare_data = np.hstack((add_data, data))
        # These don't get zerod
        compare_rec(compare_data, d, "expand at the front")


def test_slice():
    """
    Test reading by slice
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            # initial write
            fits.write_table(data)

            # test reading single columns
            for f in data.dtype.names:
                d = fits[1][f][:]
                compare_array(
                    data[f], d, "test read all rows %s column subset" % f
                )

            # test reading row subsets
            rows = [1, 3]
            for f in data.dtype.names:
                d = fits[1][f][rows]
                compare_array(data[f][rows], d, "test %s row subset" % f)
            for f in data.dtype.names:
                d = fits[1][f][1:3]
                compare_array(data[f][1:3], d, "test %s row slice" % f)
            for f in data.dtype.names:
                d = fits[1][f][1:4:2]
                compare_array(
                    data[f][1:4:2], d, "test %s row slice with step" % f
                )
            for f in data.dtype.names:
                d = fits[1][f][::2]
                compare_array(
                    data[f][::2], d, "test %s row slice with only setp" % f
                )

            # now list of columns
            cols = ['u2scalar', 'f4vec', 'Sarr']
            d = fits[1][cols][:]
            for f in d.dtype.names:
                compare_array(data[f][:], d[f], "test column list %s" % f)

            cols = ['u2scalar', 'f4vec', 'Sarr']
            d = fits[1][cols][rows]
            for f in d.dtype.names:
                compare_array(
                    data[f][rows], d[f], "test column list %s row subset" % f
                )

            cols = ['u2scalar', 'f4vec', 'Sarr']
            d = fits[1][cols][1:3]
            for f in d.dtype.names:
                compare_array(
                    data[f][1:3], d[f], "test column list %s row slice" % f
                )


def test_table_append():
    """
    Test creating a table and appending new rows.
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            # initial write
            fits.write_table(data, header=adata['keys'], extname='mytable')
            # now append
            data2 = data.copy()
            data2['f4scalar'] = 3
            fits[1].append(data2)

            d = fits[1].read()
            assert d.size == data.size*2

            compare_rec(data, d[0:data.size], "Comparing initial write")
            compare_rec(data2, d[data.size:], "Comparing appended data")

            h = fits[1].read_header()
            compare_headerlist_header(adata['keys'], h)

            # append with list of arrays and names
            names = data.dtype.names
            data3 = [np.array(data[name]) for name in names]
            fits[1].append(data3, names=names)

            d = fits[1].read()
            assert d.size == data.size*3
            compare_rec(data, d[2*data.size:], "Comparing appended data")

            # append with list of arrays and columns
            fits[1].append(data3, columns=names)

            d = fits[1].read()
            assert d.size == data.size*4
            compare_rec(data, d[3*data.size:], "Comparing appended data")


def test_table_subsets():
    """
    testing reading subsets
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.write_table(data, header=adata['keys'], extname='mytable')

            for rows in [[1, 3], [3, 1]]:
                d = fits[1].read(rows=rows)
                compare_rec_subrows(data, d, rows, "table subset")
                columns = ['i1scalar', 'f4arr']
                d = fits[1].read(columns=columns, rows=rows)

                for f in columns:
                    d = fits[1].read_column(f, rows=rows)
                    compare_array(
                        data[f][rows], d, "row subset, multi-column '%s'" % f
                    )
                for f in data.dtype.names:
                    d = fits[1].read_column(f, rows=rows)
                    compare_array(
                        data[f][rows], d, "row subset, column '%s'" % f
                    )


def test_gz_write_read():
    """
    Test a basic table write, data and a header, then reading back in to
    check the values

    this code all works, but the file is zere size when done!
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.write_table(data, header=adata['keys'], extname='mytable')

            d = fits[1].read()
            compare_rec(data, d, "gzip write/read")

            h = fits[1].read_header()
            for entry in adata['keys']:
                name = entry['name'].upper()
                value = entry['value']
                hvalue = h[name]
                if isinstance(hvalue, str):
                    hvalue = hvalue.strip()
                assert value == hvalue, "testing header key '%s'" % name

                if 'comment' in entry:
                    assert (
                        entry['comment'].strip()
                        == h.get_comment(name).strip()
                    ), (
                        "testing comment for header key '%s'" % name
                    )

        stat = os.stat(fname)
        assert stat.st_size != 0, "Making sure the data was flushed to disk"


@pytest.mark.skipif('SKIP_BZIP_TEST' in os.environ,
                    reason='SKIP_BZIP_TEST set')
def test_bz2_read():
    '''
    Write a normal .fits file, run bzip2 on it, then read the bz2
    file and verify that it's the same as what we put in; we don't
    [currently support or] test *writing* bzip2.
    '''

    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        bzfname = fname + '.bz2'

        try:
            fits = FITS(fname, 'rw')
            fits.write_table(data, header=adata['keys'], extname='mytable')
            fits.close()

            os.system('bzip2 %s' % fname)
            f2 = FITS(bzfname)
            d = f2[1].read()
            compare_rec(data, d, "bzip2 read")

            h = f2[1].read_header()
            for entry in adata['keys']:
                name = entry['name'].upper()
                value = entry['value']
                hvalue = h[name]
                if isinstance(hvalue, str):
                    hvalue = hvalue.strip()

                assert value == hvalue, "testing header key '%s'" % name

                if 'comment' in entry:
                    assert (
                        entry['comment'].strip()
                        == h.get_comment(name).strip()
                    ), (
                        "testing comment for header key '%s'" % name
                    )
        except Exception:
            import traceback
            traceback.print_exc()

            assert False, 'Exception in testing bzip2 reading'


def test_checksum():
    """
    test that checksumming works
    """
    adata = make_data()
    data = adata['data']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            fits.write_table(data, header=adata['keys'], extname='mytable')
            fits[1].write_checksum()
            fits[1].verify_checksum()


def test_trim_strings():
    """
    test mode where we strim strings on read
    """

    dt = [('fval', 'f8'), ('name', 'S15'), ('vec', 'f4', 2)]
    n = 3
    data = np.zeros(n, dtype=dt)
    data['fval'] = np.random.random(n)
    data['vec'] = np.random.random(n*2).reshape(n, 2)

    data['name'] = ['mike', 'really_long_name_to_fill', 'jan']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write(data)

        for onconstruct in [True, False]:
            if onconstruct:
                ctrim = True
                otrim = False
            else:
                ctrim = False
                otrim = True

            with FITS(fname, 'rw', trim_strings=ctrim) as fits:

                if ctrim:
                    dread = fits[1][:]
                    compare_rec(
                        data,
                        dread,
                        "trimmed strings constructor",
                    )

                    dname = fits[1]['name'][:]
                    compare_array(
                        data['name'],
                        dname,
                        "trimmed strings col read, constructor",
                    )
                    dread = fits[1][['name']][:]
                    compare_array(
                        data['name'],
                        dread['name'],
                        "trimmed strings col read, constructor",
                    )

                dread = fits[1].read(trim_strings=otrim)
                compare_rec(
                    data,
                    dread,
                    "trimmed strings keyword",
                )
                dname = fits[1].read(columns='name', trim_strings=otrim)
                compare_array(
                    data['name'],
                    dname,
                    "trimmed strings col keyword",
                )
                dread = fits[1].read(columns=['name'], trim_strings=otrim)
                compare_array(
                    data['name'],
                    dread['name'],
                    "trimmed strings col keyword",
                )

        # convenience function
        dread = read(fname, trim_strings=True)
        compare_rec(
            data,
            dread,
            "trimmed strings convenience function",
        )
        dname = read(fname, columns='name', trim_strings=True)
        compare_array(
            data['name'],
            dname,
            "trimmed strings col convenience function",
        )
        dread = read(fname, columns=['name'], trim_strings=True)
        compare_array(
            data['name'],
            dread['name'],
            "trimmed strings col convenience function",
        )


def test_lower_upper():
    """
    test forcing names to upper and lower
    """

    rng = np.random.RandomState(8908)

    dt = [('MyName', 'f8'), ('StuffThings', 'i4'), ('Blah', 'f4')]
    data = np.zeros(3, dtype=dt)
    data['MyName'] = rng.uniform(data.size)
    data['StuffThings'] = rng.uniform(data.size)
    data['Blah'] = rng.uniform(data.size)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write(data)

        for i in [1, 2]:
            if i == 1:
                lower = True
                upper = False
            else:
                lower = False
                upper = True

            with FITS(fname, 'rw', lower=lower, upper=upper) as fits:
                for rows in [None, [1, 2]]:

                    d = fits[1].read(rows=rows)
                    compare_names(d.dtype.names, data.dtype.names,
                                  lower=lower, upper=upper)

                    d = fits[1].read(
                        rows=rows, columns=['MyName', 'stuffthings']
                    )
                    compare_names(d.dtype.names, data.dtype.names[0:2],
                                  lower=lower, upper=upper)

                    d = fits[1][1:2]
                    compare_names(d.dtype.names, data.dtype.names,
                                  lower=lower, upper=upper)

                    if rows is not None:
                        d = fits[1][rows]
                    else:
                        d = fits[1][:]

                    compare_names(d.dtype.names, data.dtype.names,
                                  lower=lower, upper=upper)

                    if rows is not None:
                        d = fits[1][['myname', 'stuffthings']][rows]
                    else:
                        d = fits[1][['myname', 'stuffthings']][:]

                    compare_names(d.dtype.names, data.dtype.names[0:2],
                                  lower=lower, upper=upper)

            # using overrides
            with FITS(fname, 'rw') as fits:
                for rows in [None, [1, 2]]:

                    d = fits[1].read(rows=rows, lower=lower, upper=upper)
                    compare_names(d.dtype.names, data.dtype.names,
                                  lower=lower, upper=upper)

                    d = fits[1].read(
                        rows=rows, columns=['MyName', 'stuffthings'],
                        lower=lower, upper=upper
                    )
                    compare_names(d.dtype.names, data.dtype.names[0:2],
                                  lower=lower, upper=upper)

            for rows in [None, [1, 2]]:
                d = read(fname, rows=rows, lower=lower, upper=upper)
                compare_names(d.dtype.names, data.dtype.names,
                              lower=lower, upper=upper)

                d = read(fname, rows=rows, columns=['MyName', 'stuffthings'],
                         lower=lower, upper=upper)
                compare_names(d.dtype.names, data.dtype.names[0:2],
                              lower=lower, upper=upper)


def test_read_raw():
    """
    testing reading the file as raw bytes
    """
    rng = np.random.RandomState(8908)

    dt = [('MyName', 'f8'), ('StuffThings', 'i4'), ('Blah', 'f4')]
    data = np.zeros(3, dtype=dt)
    data['MyName'] = rng.uniform(data.size)
    data['StuffThings'] = rng.uniform(data.size)
    data['Blah'] = rng.uniform(data.size)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        try:
            with FITS(fname, 'rw') as fits:
                fits.write(data)
                raw1 = fits.read_raw()

            with FITS('mem://', 'rw') as fits:
                fits.write(data)
                raw2 = fits.read_raw()

            with open(fname, 'rb') as fobj:
                raw3 = fobj.read()

            assert raw1 == raw2
            assert raw1 == raw3
        except Exception:
            import traceback
            traceback.print_exc()
            assert False, 'Exception in testing read_raw'


def test_table_bitcol_read_write():
    """
    Test basic write/read with bitcols
    """

    adata = make_data()
    bdata = adata['bdata']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            fits.write_table(bdata, extname='mytable', write_bitcols=True)

            d = fits[1].read()
            compare_rec(bdata, d, "table read/write")

            rows = [0, 2]
            d = fits[1].read(rows=rows)
            compare_rec(bdata[rows], d, "table read/write rows")

            d = fits[1][:2]
            compare_rec(bdata[:2], d, "table read/write slice")

        # now test read_column
        with FITS(fname) as fits:

            for f in bdata.dtype.names:
                d = fits[1].read_column(f)
                compare_array(
                    bdata[f], d, "table 1 single field read '%s'" % f
                )

            # now list of columns
            for cols in [['b1vec', 'b1arr']]:
                d = fits[1].read(columns=cols)
                for f in d.dtype.names:
                    compare_array(bdata[f][:], d[f], "test column list %s" % f)

                for rows in [[1, 3], [3, 1]]:
                    d = fits[1].read(columns=cols, rows=rows)
                    for f in d.dtype.names:
                        compare_array(
                            bdata[f][rows],
                            d[f],
                            "test column list %s row subset" % f
                        )


def test_table_bitcol_append():
    """
    Test creating a table with bitcol support and appending new rows.
    """
    adata = make_data()
    bdata = adata['bdata']

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            # initial write
            fits.write_table(bdata, extname='mytable', write_bitcols=True)

        with FITS(fname, 'rw') as fits:
            # now append
            bdata2 = bdata.copy()
            fits[1].append(bdata2)

            d = fits[1].read()
            assert d.size == bdata.size*2

            compare_rec(bdata, d[0:bdata.size], "Comparing initial write")
            compare_rec(bdata2, d[bdata.size:], "Comparing appended data")


def test_table_bitcol_insert():
    """
    Test creating a table with bitcol support and appending new rows.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            # initial write
            nrows = 3
            d = np.zeros(nrows, dtype=[('ra', 'f8')])
            d['ra'] = range(d.size)
            fits.write(d)

        with FITS(fname, 'rw') as fits:
            bcol = np.array([True, False, True])

            # now append
            fits[-1].insert_column(
                'bscalar_inserted', bcol, write_bitcols=True
            )

            d = fits[-1].read()
            assert d.size == nrows, 'read size equals'
            compare_array(bcol, d['bscalar_inserted'], "inserted bitcol")

            bvec = np.array(
                [[True, False],
                 [False, True],
                 [True, True]]
            )

            # now append
            fits[-1].insert_column('bvec_inserted', bvec, write_bitcols=True)

            d = fits[-1].read()
            assert d.size == nrows, 'read size equals'
            compare_array(bvec, d['bvec_inserted'], "inserted bitcol")
