from multiprocessing.pool import ThreadPool
import tempfile

import numpy as np
import fitsio


def test_threading():
    """
    Test a basic image write, data and a header, then reading back in to
    check the values
    """
    filenames = [
        tempfile.mktemp(prefix='fitsio-ImageWrite-', suffix='.fits')
        for i in range(32)
    ]

    # create files for reading in serial
    def create_file(i):
        fname = filenames[i]
        data = np.zeros((32, 32), dtype='f8')
        data[:] = i
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write_image(data)

    def read_file(i):
        fname = filenames[i]
        with fitsio.FITS(fname, 'r') as fits:
            assert (fits[0].read() == i).all()

    with ThreadPool(32) as pool:
        pool.map(create_file, range(32))
        pool.map(read_file, range(32))
