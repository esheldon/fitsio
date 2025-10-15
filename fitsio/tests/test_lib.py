import os
import tempfile
import numpy as np
from ..fitslib import FITS, read_header
from .checks import compare_array, compare_rec


def test_move_by_name():
    """
    test moving hdus by name
    """

    nrows = 3

    seed = 1234
    rng = np.random.RandomState(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            data1 = np.zeros(nrows, dtype=[('ra', 'f8'), ('dec', 'f8')])
            data1['ra'] = rng.uniform(nrows)
            data1['dec'] = rng.uniform(nrows)
            fits.write_table(data1, extname='mytable')

            fits[-1].write_key('EXTVER', 1)

            data2 = np.zeros(nrows, dtype=[('ra', 'f8'), ('dec', 'f8')])
            data2['ra'] = rng.uniform(nrows)
            data2['dec'] = rng.uniform(nrows)

            fits.write_table(data2, extname='mytable')
            fits[-1].write_key('EXTVER', 2)

            hdunum1 = fits.movnam_hdu('mytable', extver=1)
            assert hdunum1 == 2
            hdunum2 = fits.movnam_hdu('mytable', extver=2)
            assert hdunum2 == 3


def test_ext_ver():
    """
    Test using extname and extver, all combinations I can think of
    """

    seed = 9889
    rng = np.random.RandomState(seed)

    dtype = [('num', 'i4'), ('ra', 'f8'), ('dec', 'f8')]

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:

            img1 = np.arange(2*3, dtype='i4').reshape(2, 3) + 5
            img2 = np.arange(2*3, dtype='i4').reshape(2, 3) + 6
            img3 = np.arange(2*3, dtype='i4').reshape(2, 3) + 7

            nrows = 3
            data1 = np.zeros(nrows, dtype=dtype)

            data1['num'] = 1
            data1['ra'] = rng.uniform(nrows)
            data1['dec'] = rng.uniform(nrows)

            data2 = np.zeros(nrows, dtype=dtype)

            data2['num'] = 2
            data2['ra'] = rng.uniform(nrows)
            data2['dec'] = rng.uniform(nrows)

            data3 = np.zeros(nrows, dtype=dtype)
            data3['num'] = 3
            data3['ra'] = rng.uniform(nrows)
            data3['dec'] = rng.uniform(nrows)

            hdr1 = {'k1': 'key1'}
            hdr2 = {'k2': 'key2'}

            fits.write_image(img1, extname='myimage', header=hdr1, extver=1)
            fits.write_table(data1)
            fits.write_table(data2, extname='mytable', extver=1)
            fits.write_image(img2, extname='myimage', header=hdr2, extver=2)
            fits.write_table(data3, extname='mytable', extver=2)
            fits.write_image(img3)

            d1 = fits[1].read()
            d2 = fits['mytable'].read()
            d2b = fits['mytable', 1].read()
            d3 = fits['mytable', 2].read()

            for f in data1.dtype.names:
                compare_rec(data1, d1, "data1")
                compare_rec(data2, d2, "data2")
                compare_rec(data2, d2b, "data2b")
                compare_rec(data3, d3, "data3")

            dimg1 = fits[0].read()
            dimg1b = fits['myimage', 1].read()
            dimg2 = fits['myimage', 2].read()
            dimg3 = fits[5].read()

            compare_array(img1, dimg1, "img1")
            compare_array(img1, dimg1b, "img1b")
            compare_array(img2, dimg2, "img2")
            compare_array(img3, dimg3, "img3")

        rhdr1 = read_header(fname, ext='myimage', extver=1)
        rhdr2 = read_header(fname, ext='myimage', extver=2)
        assert 'k1' in rhdr1, 'testing k1 in header version 1'
        assert 'k2' in rhdr2, 'testing k2 in header version 2'
