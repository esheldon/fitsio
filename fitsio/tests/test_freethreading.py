from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile

import numpy as np

import pytest

import fitsio


def test_concurrent_shared_usage_raises():
    def _read_data(fp):
        return fp[0].read()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "fname.fits")

        with fitsio.FITS(fname, "rw", clobber=True) as fp:
            fp.write_image(np.ones((1000, 100)))

        with fitsio.FITS(fname, "r") as fp:
            res = fp[0].read()
            assert (res == 1).all()

            with ThreadPoolExecutor(max_workers=2) as exc:
                futs = [
                    exc.submit(_read_data, fp)
                    for _ in range(2)
                ]

                for fut in as_completed(futs):
                    with pytest.raises(RuntimeError) as e:
                        fut.result()
                    assert "Concurrent" in str(e.value)
