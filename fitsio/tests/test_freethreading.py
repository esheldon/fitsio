from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile

import numpy as np

import fitsio


def test_locking_works():
    rng = np.random.RandomState(seed=10)
    img = rng.normal(size=(1000, 1000))
    max_workers = 2

    def _read_data(fp):
        return fp[0].read()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "fname.fits")

        with fitsio.FITS(fname, "rw", clobber=True) as fp:
            fp.write_image(img)

        with fitsio.FITS(fname, "r") as fp:
            res = fp[0].read()
            assert (res == img).all()

            with ThreadPoolExecutor(max_workers=max_workers) as exc:
                futs = [
                    exc.submit(_read_data, fp) for _ in range(10 * max_workers)
                ]
                for fut in as_completed(futs):
                    res = fut.result()
                    assert (res == img).all()
