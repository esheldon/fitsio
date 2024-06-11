import tempfile
import os
import numpy as np
from ..fitslib import write, FITS


def test_empty_image_slice():
    shape = (10, 10)
    data = np.arange(shape[0] * shape[1]).reshape(shape)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        write(fname, data, clobber=True)

        # third, test slices using fitsio
        with FITS(fname) as fits:
            # first, passing the slices directly
            # overlap = f[0][xslice, yslice]
            overlap = fits[0][0:8, 0:0]
            assert overlap.size == 0
