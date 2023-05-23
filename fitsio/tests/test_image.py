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
