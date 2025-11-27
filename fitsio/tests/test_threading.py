from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import tempfile
import time

import numpy as np
import fitsio

import pytest

SIZE = 10000


def create_file(fname):
    val = int(
        os.path.basename(fname).replace(".fits", "").replace("fname", "")
    )
    data = np.zeros((SIZE, SIZE), dtype='f8')
    data[:] = val
    with fitsio.FITS(fname, 'rw') as fits:
        fits.write_image(data)


def read_file(fname):
    val = int(
        os.path.basename(fname).replace(".fits", "").replace("fname", "")
    )
    with fitsio.FITS(fname, 'r') as fits:
        assert (fits[0].read() == val).all()


@pytest.mark.xfail(reason="Releasing the GIL doesn't help much so far!")
@pytest.mark.parametrize("klass", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_threading(klass):
    """
    Test a basic image write, data and a header, then reading back in to
    check the values
    """
    nt = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = [
            os.path.join(tmpdir, "fname%d.fits" % i) for i in range(nt)
        ]

        def _remove_files():
            for fname in filenames:
                try:
                    os.remove(fname)
                except Exception:
                    pass

        t0 = time.time()
        create_file(filenames[0])
        read_file(filenames[0])
        t0_one = time.time() - t0
        print("one file time:", t0_one, flush=True)
        _remove_files()

        t0 = time.time()
        with klass(max_workers=nt) as pool:
            for _ in pool.map(create_file, filenames):
                pass
            for _ in pool.map(read_file, filenames):
                pass
        t0_threads = time.time() - t0
        print("parallel time / one file time", t0_threads / t0_one, flush=True)
        _remove_files()

        t0 = time.time()
        for fname in filenames:
            create_file(fname)
        for fname in filenames:
            read_file(fname)
        t0_serial = time.time() - t0
        print("serial time / one file time:", t0_serial / t0_one, flush=True)

        assert t0_threads < t0_serial, (
            "Threading should be faster than serial! ( %f < %f)"
            % (t0_threads, t0_serial)
        )
