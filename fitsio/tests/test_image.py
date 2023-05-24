import os
import tempfile
# import warnings
from .checks import check_header, compare_array
import numpy as np
from ..fitslib import FITS

DTYPES = ['u1', 'i1', 'u2', 'i2', '<u4', 'i4', 'i8', '>f4', 'f8']


def test_image_write_read():
    """
    Test a basic image write, data and a header, then reading back in to
    check the values
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        with FITS(fname, 'rw') as fits:

            # note mixing up byte orders a bit
            for dtype in DTYPES:
                data = np.arange(5*20, dtype=dtype).reshape(5, 20)
                header = {'DTYPE': dtype, 'NBYTES': data.dtype.itemsize}
                fits.write_image(data, header=header)
                rdata = fits[-1].read()

                compare_array(data, rdata, "images")

                rh = fits[-1].read_header()
                check_header(header, rh)

        with FITS(fname) as fits:
            for i in range(len(DTYPES)):
                assert not fits[i].is_compressed(), 'not compressed'


def test_image_write_empty():
    """
    Test a basic image write, with no data and just a header, then reading
    back in to check the values
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        data = None

        header = {
            'EXPTIME': 120,
            'OBSERVER': 'Beatrice Tinsley',
            'INSTRUME': 'DECam',
            'FILTER': 'r',
        }
        ccds = ['CCD1', 'CCD2', 'CCD3', 'CCD4', 'CCD5', 'CCD6', 'CCD7', 'CCD8']
        with FITS(fname, 'rw', ignore_empty=True) as fits:
            for extname in ccds:
                fits.write_image(data, header=header)
                _ = fits[-1].read()
                rh = fits[-1].read_header()
                check_header(header, rh)


def test_image_write_read_from_dims():
    """
    Test creating an image from dims and writing in place
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            # note mixing up byte orders a bit
            for dtype in DTYPES:
                data = np.arange(5*20, dtype=dtype).reshape(5, 20)

                fits.create_image_hdu(dims=data.shape, dtype=data.dtype)

                fits[-1].write(data)
                rdata = fits[-1].read()

                compare_array(data, rdata, "images")

        with FITS(fname) as fits:
            for i in range(len(DTYPES)):
                assert not fits[i].is_compressed(), "not compressed"


def test_image_write_read_from_dims_chunks():
    """
    Test creating an image and reading/writing chunks
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            # note mixing up byte orders a bit
            for dtype in DTYPES:
                data = np.arange(5*3, dtype=dtype).reshape(5, 3)

                fits.create_image_hdu(dims=data.shape, dtype=data.dtype)

                chunk1 = data[0:2, :]
                chunk2 = data[2:, :]

                #
                # first using scalar pixel offset
                #

                fits[-1].write(chunk1)

                start = chunk1.size
                fits[-1].write(chunk2, start=start)

                rdata = fits[-1].read()

                compare_array(data, rdata, "images")

                #
                # now using sequence, easier to calculate
                #

                fits.create_image_hdu(dims=data.shape,
                                      dtype=data.dtype)

                # first using pixel offset
                fits[-1].write(chunk1)

                start = [2, 0]
                fits[-1].write(chunk2, start=start)

                rdata2 = fits[-1].read()

                compare_array(data, rdata2, "images")

        with FITS(fname) as fits:
            for i in range(len(DTYPES)):
                assert not fits[i].is_compressed(), "not compressed"


def test_image_slice():
    """
    test reading an image slice
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            # note mixing up byte orders a bit
            for dtype in DTYPES:
                data = np.arange(16*20, dtype=dtype).reshape(16, 20)
                header = {'DTYPE': dtype, 'NBYTES': data.dtype.itemsize}

                fits.write_image(data, header=header)
                rdata = fits[-1][4:12, 9:17]

                compare_array(data[4:12, 9:17], rdata, "images")

                rh = fits[-1].read_header()
                check_header(header, rh)


def _check_shape(expected_data, rdata):
    mess = (
        'Data are not the same (Expected shape: %s, '
        'actual shape: %s.' % (expected_data.shape, rdata.shape)
    )
    np.testing.assert_array_equal(expected_data, rdata, mess)


def test_read_flip_axis_slice():
    """
    Test reading a slice when the slice's start is less than the slice's stop.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            dtype = np.int16
            data = np.arange(100 * 200, dtype=dtype).reshape(100, 200)
            fits.write_image(data)
            hdu = fits[-1]
            rdata = hdu[:, 130:70]

            # Expanded by two to emulate adding one to the start value, and
            # adding one to the calculated dimension.
            expected_data = data[:, 130:70:-1]

            _check_shape(expected_data, rdata)

            rdata = hdu[:, 130:70:-6]
            expected_data = data[:, 130:70:-6]
            _check_shape(expected_data, rdata)

            # Expanded by two to emulate adding one to the start value, and
            # adding one to the calculated dimension.
            expected_data = data[:, 130:70:-6]
            _check_shape(expected_data, rdata)

            # Positive step integer with start > stop will return an empty
            # array
            rdata = hdu[:, 90:60:4]
            expected_data = np.empty(0, dtype=dtype)
            _check_shape(expected_data, rdata)

            # Negative step integer with start < stop will return an empty
            # array.
            rdata = hdu[:, 60:90:-4]
            expected_data = np.empty(0, dtype=dtype)
            _check_shape(expected_data, rdata)


def test_image_slice_striding():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            # note mixing up byte orders a bit
            for dtype in DTYPES:
                data = np.arange(16*20, dtype=dtype).reshape(16, 20)
                header = {'DTYPE': dtype, 'NBYTES': data.dtype.itemsize}
                fits.write_image(data, header=header)

                rdata = fits[-1][4:16:4, 2:20:2]
                expected_data = data[4:16:4, 2:20:2]
                assert rdata.shape == expected_data.shape, (
                    "Shapes differ with dtype %s" % dtype
                )
                compare_array(
                    expected_data, rdata, "images with dtype %s" % dtype
                )


def test_read_ignore_scaling():
    """
    Test the flag to ignore scaling when reading an HDU.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            dtype = 'i2'
            data = np.arange(10 * 20, dtype=dtype).reshape(10, 20)
            header = {
                'DTYPE': dtype,
                'BITPIX': 16,
                'NBYTES': data.dtype.itemsize,
                'BZERO': 9.33,
                'BSCALE': 3.281
            }

            fits.write_image(data, header=header)
            hdu = fits[-1]

            rdata = hdu.read()
            assert rdata.dtype == np.float32, 'Wrong dtype.'

            hdu.ignore_scaling = True
            rdata = hdu[:, :]
            assert rdata.dtype == dtype, 'Wrong dtype when ignoring.'
            np.testing.assert_array_equal(
                data, rdata, err_msg='Wrong unscaled data.'
            )

            rh = fits[-1].read_header()
            check_header(header, rh)

            hdu.ignore_scaling = False
            rdata = hdu[:, :]
            assert rdata.dtype == np.float32, (
                'Wrong dtype when not ignoring.'
            )
            np.testing.assert_array_equal(
                data.astype(np.float32), rdata,
                err_msg='Wrong scaled data returned.'
            )
