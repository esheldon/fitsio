from multiprocessing.pool import ThreadPool
import os
import tempfile
import time

import numpy as np
import fitsio


def test_threading():
    """
    Test a basic image write, data and a header, then reading back in to
    check the values
    """

    size = 3000

    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = [
            os.path.join(tmpdir, "fname%d.fits" % i) for i in range(32)
        ]

        # create files for reading in serial
        def create_file(i):
            fname = filenames[i]
            data = np.zeros((size, size), dtype='f8')
            data[:] = i
            with fitsio.FITS(fname, 'rw', clobber=True) as fits:
                fits.write_image(data)

        def read_file(i):
            fname = filenames[i]
            with fitsio.FITS(fname, 'r') as fits:
                assert (fits[0].read() == i).all()

        t0 = time.time()
        with ThreadPool(32) as pool:
            pool.map(create_file, range(32))
            pool.map(read_file, range(32))
        t0_threads = time.time() - t0
        print("threaded time:", t0_threads, flush=True)

        t0 = time.time()
        for i in range(32):
            create_file(i)
            read_file(i)
        t0_serial = time.time() - t0
        print("serial time:", t0_serial, flush=True)

        assert t0_threads < t0_serial, (
            "Threading should be faster than serial! ( %f < %f)"
            % (t0_threads, t0_serial)
        )
