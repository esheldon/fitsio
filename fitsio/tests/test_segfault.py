import os
import tempfile

import numpy as np

import fitsio
from .. import fitsio_backend, backend_is_bundled

import pytest


def _write_plio1(tmpdir, i):
    """The crashing call, isolated."""
    fname = os.path.join(tmpdir, f"plio_{i:04d}.fits.fz")
    img = np.arange(16, dtype="i4").reshape(4, 4)
    with fitsio.FITS(fname, "rw") as f:
        f.write(img, compress="PLIO_1", tile_dims=(2, 2))


def _write_rice1(tmpdir, i):
    """A 'control' compressed write that we know does NOT crash."""
    fname = os.path.join(tmpdir, f"rice_{i:04d}.fits.fz")
    img = np.arange(100, dtype="i4").reshape(10, 10)
    with fitsio.FITS(fname, "rw") as f:
        f.write(img, compress="RICE_1", tile_dims=(5, 5))


def _run_alone(n):
    """N back-to-back PLIO_1 writes, fresh tempdir per call."""
    print(f"variant: PLIO_1 alone, {n} iterations")
    for i in range(n):
        with tempfile.TemporaryDirectory() as tmp:
            _write_plio1(tmp, i)
        if (i + 1) % 10 == 0:
            print(f"  ok through {i + 1}")
    print("completed without abort")


def _run_mixed(n):
    """
    Mirrors the test ordering that exposed the bug: many
    RICE_1 fixtures then a PLIO_1 write.  Loop the whole sequence.
    """
    print(f"variant: 25x RICE_1 then 1x PLIO_1, {n} outer iterations")
    for i in range(n):
        with tempfile.TemporaryDirectory() as tmp:
            for k in range(25):
                _write_rice1(tmp, k)
            _write_plio1(tmp, i)
        if (i + 1) % 5 == 0:
            print(f"  ok through {i + 1}", flush=True)
    print("completed without abort")


@pytest.mark.slow
@pytest.mark.skipif(
    (fitsio_backend() == "cfitsio" and not backend_is_bundled()),
    reason=(
        "small images cause a memory corruption w/ PLIO "
        "compression (see https://github.com/heasarc/cfitsio/issues/136)"
    ),
)
def test_segfault_osx():
    n = 50
    _run_mixed(n)
    _run_alone(n)
