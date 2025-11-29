from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import time

import numpy as np
import fitsio

import pytest

SIZE = 5000
DATA = np.zeros((SIZE, SIZE), dtype='f8')
DATA[:] = -1


def create_file(fname):
    with fitsio.FITS(fname, 'rw') as fits:
        fits.write_image(DATA)


def read_file(fname):
    with fitsio.FITS(fname, 'r') as fits:
        fits[0].read()


def test_threading_works():
    """
    Test a basic image write, data and a header, then reading back in to
    check the values
    """
    nt = 32

    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = [
            os.path.join(tmpdir, "fname%d.fits" % i) for i in range(nt)
        ]

        def _create_file(i):
            fname = filenames[i]
            data = np.zeros((32, 32), dtype='f8')
            data[:] = i
            with fitsio.FITS(fname, 'rw', clobber=True) as fits:
                fits.write_image(data)

        def _read_file(i):
            fname = filenames[i]
            with fitsio.FITS(fname, 'r') as fits:
                assert (fits[0].read() == i).all()

        with ThreadPoolExecutor(max_workers=nt) as pool:
            for _ in pool.map(_create_file, range(nt)):
                pass
            for _ in pool.map(_read_file, range(nt)):
                pass


@pytest.mark.xfail(reason="Threading performance might be flaky!")
@pytest.mark.parametrize(
    "write_only,read_only",
    [
        (False, True),
        (True, False),
        (False, False),
    ],
)
@pytest.mark.parametrize("klass", [ThreadPoolExecutor])
def test_threading_timing(klass, write_only, read_only):
    """
    Test a basic image write, data and a header, then reading back in to
    check the values
    """
    nt = 2
    fac = 8

    if read_only:
        print("\nread only", flush=True)
    elif write_only:
        print("\nwrite only", flush=True)
    else:
        print("\nread and write", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = [
            os.path.join(tmpdir, "fname%d.fits" % i) for i in range(nt * fac)
        ]

        def _remove_files():
            for fname in filenames:
                try:
                    os.remove(fname)
                except Exception:
                    pass

        if read_only:
            for fname in filenames:
                create_file(fname)

        t0 = time.time()
        if not read_only:
            create_file(filenames[0])
        if not write_only:
            read_file(filenames[0])
        t0_one = time.time() - t0
        print("one file time:", t0_one, flush=True)
        if not read_only:
            _remove_files()

        t0 = time.time()
        with klass(max_workers=nt) as pool:
            if not read_only:
                futs = [
                    pool.submit(create_file, filenames[i])
                    for i in range(nt * fac)
                ]
                for fut in as_completed(futs):
                    fut.result()
            if not write_only:
                futs = [
                    pool.submit(read_file, filenames[i])
                    for i in range(nt * fac)
                ]
                for fut in as_completed(futs):
                    fut.result()
        t0_threads = time.time() - t0
        print(
            "parallel time / one file time",
            t0_threads / t0_one,
            "(perfect is %d)" % fac,
            flush=True,
        )
        if not read_only:
            _remove_files()

        t0 = time.time()
        if not read_only:
            for fname in filenames:
                create_file(fname)
        if not write_only:
            for fname in filenames:
                read_file(fname)
        t0_serial = time.time() - t0
        print(
            "serial time / one file time:",
            t0_serial / t0_one,
            f"(should be about {nt * fac})",
            flush=True,
        )

        if read_only:
            assert t0_threads < t0_serial, (
                "Threading should be faster than serial! ( %f < %f)"
                % (t0_threads, t0_serial)
            )
