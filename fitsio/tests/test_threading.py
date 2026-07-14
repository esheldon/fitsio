from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
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
        fits[0].write_checksum()
        fits.read_raw()


def read_file(fname):
    with fitsio.FITS(fname, 'r') as fits:
        fits[0].verify_checksum()
        fits[0].read()


@pytest.mark.parallel_threads_limit(1)
@pytest.mark.iterations(1)
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
                fits[0].write_checksum()

        def _read_file(i):
            fname = filenames[i]
            with fitsio.FITS(fname, 'r') as fits:
                fits[0].verify_checksum()
                assert (fits[0].read() == i).all()

        with ThreadPoolExecutor(max_workers=nt) as pool:
            for _ in pool.map(_create_file, range(nt)):
                pass
            for _ in pool.map(_read_file, range(nt)):
                pass


@pytest.mark.xfail(
    reason="threading performance might be flaky",
    condition=sys.version_info < (3, 13) or not fitsio.backend_is_reentrant(),
)
@pytest.mark.parallel_threads_limit(1)
@pytest.mark.iterations(1)
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
    fac = 2

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

        n_trials = 3
        t0_threads = 0
        t0_one = 0
        t0_serial = 0

        for _ in range(n_trials):
            t0 = time.time()
            if not read_only:
                create_file(filenames[0])
            if not write_only:
                read_file(filenames[0])
            t0_one += time.time() - t0
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
            t0_threads += time.time() - t0
            if not read_only:
                _remove_files()

            t0 = time.time()
            if not read_only:
                for fname in filenames:
                    create_file(fname)
            if not write_only:
                for fname in filenames:
                    read_file(fname)
            t0_serial += time.time() - t0

        t0_one /= n_trials
        t0_serial /= n_trials
        t0_threads /= n_trials

        print("one file time:", t0_one, flush=True)
        print(
            "parallel time / one file time",
            t0_threads / t0_one,
            "(perfect is %d)" % fac,
            flush=True,
        )
        print(
            "serial time / one file time:",
            t0_serial / t0_one,
            f"(should be about {nt * fac})",
            flush=True,
        )

        assert t0_threads < t0_serial, (
            "Threading should be faster than serial! (%f < %f)"
            % (t0_threads, t0_serial)
        )


@pytest.mark.xfail(
    reason="threading performance might be flaky",
    condition=sys.version_info < (3, 13) or not fitsio.backend_is_reentrant(),
)
@pytest.mark.parallel_threads_limit(1)
@pytest.mark.iterations(1)
def test_threading_read_one_file():
    nt = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "fname.fits")
        with fitsio.FITS(fname, 'rw', clobber=True) as fits:
            fits.write_image(np.concatenate([DATA]), compress="GZIP", qlevel=0)
            fits[0].write_checksum()

        def _read_file(fname):
            with fitsio.FITS(fname, 'r') as fits:
                fits[0].verify_checksum()
                assert (fits[1].read() == -1).all()
            return True

        n_trials = 3
        t0_threads = 0
        t0_one = 0
        t0_serial = 0

        for _ in range(n_trials):
            t0 = time.time()
            _read_file(fname)
            t0_one += time.time() - t0

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=nt) as pool:
                futs = [pool.submit(_read_file, fname) for _ in range(nt)]

                assert all([fut.result() for fut in as_completed(futs)])
            t0_threads += time.time() - t0

            t0 = time.time()
            for _ in range(nt):
                _read_file(fname)
            t0_serial += time.time() - t0

        t0_one /= n_trials
        t0_serial /= n_trials
        t0_threads /= n_trials

        print("\none file time:", t0_one, flush=True)
        print(
            "parallel time / one file time",
            t0_threads / t0_one,
            "(perfect is 1)",
            flush=True,
        )
        print(
            "serial time / one file time:",
            t0_serial / t0_one,
            f"(should be about {nt})",
            flush=True,
        )

        assert t0_threads < t0_serial, (
            "Threading should be faster than serial! (%f < %f)"
            % (t0_threads, t0_serial)
        )
