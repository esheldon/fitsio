import numpy as np
import os
import tempfile
from .checks import (
    # check_header,
    compare_array,
    compare_object_array,
    compare_rec,
    compare_headerlist_header,
    compare_rec_with_var,
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

                rows = [1, 3]
                d = fits[1].read(columns=cols, rows=rows)
                for f in d.dtype.names:
                    compare_array(
                        adata['data'][f][rows], d[f],
                        "test column list %s row subset" % f
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

                # reading multiple columns
                rows = [0, 2]
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
