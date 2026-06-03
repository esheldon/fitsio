import os
import platform
import sys
import tempfile

import numpy as np

import fitsio


def env_banner():
    print(
        f"python:   {sys.version.split()[0]} "
        f"({platform.python_implementation()})"
    )
    print(f"platform: {platform.platform()}")
    print(f"machine:  {platform.machine()}")
    print(f"fitsio:   {fitsio.__version__}")
    print(f"numpy:    {np.__version__}")
    try:
        print(f"cfitsio:  {fitsio.cfitsio_version()}")
    except Exception:
        pass
    print()


def write_plio1(tmpdir, i):
    """The crashing call, isolated."""
    fname = os.path.join(tmpdir, f"plio_{i:04d}.fits.fz")
    img = np.arange(16, dtype="i4").reshape(4, 4)
    with fitsio.FITS(fname, "rw") as f:
        f.write(img, compress="PLIO_1", tile_dims=(2, 2))


def write_rice1(tmpdir, i):
    """A 'control' compressed write that we know does NOT crash."""
    fname = os.path.join(tmpdir, f"rice_{i:04d}.fits.fz")
    img = np.arange(100, dtype="i4").reshape(10, 10)
    with fitsio.FITS(fname, "rw") as f:
        f.write(img, compress="RICE_1", tile_dims=(5, 5))


def run_alone(n):
    """N back-to-back PLIO_1 writes, fresh tempdir per call."""
    print(f"variant: PLIO_1 alone, {n} iterations")
    for i in range(n):
        with tempfile.TemporaryDirectory() as tmp:
            write_plio1(tmp, i)
        if (i + 1) % 10 == 0:
            print(f"  ok through {i + 1}")
    print("completed without abort")


def run_mixed(n):
    """
    Mirrors the test ordering that exposed the bug: many
    RICE_1 fixtures then a PLIO_1 write.  Loop the whole sequence.
    """
    print(f"variant: 25x RICE_1 then 1x PLIO_1, {n} outer iterations")
    for i in range(n):
        with tempfile.TemporaryDirectory() as tmp:
            for k in range(25):
                write_rice1(tmp, k)
            write_plio1(tmp, i)
        if (i + 1) % 5 == 0:
            print(f"  ok through {i + 1}")
    print("completed without abort")


def test_segfault_osx():
    env_banner()
    n = 10
    run_mixed(n)
    # run_alone(n)
