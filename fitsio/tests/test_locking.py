from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import tempfile
import threading
import time

import fitsio


def test_locking_read():
    nt = 10
    rng = np.random.RandomState(seed=10)
    data = rng.normal(size=(100, 100))
    lock = threading.RLock()

    def _read_file(fp):
        with lock:
            time.sleep(0.1)
            return fp[0].read()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "fname.fits")

        with fitsio.FITS(fname, "rw", clobber=True) as fp:
            fp.write_image(data)

        with fitsio.FITS(fname) as fp:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=nt) as exc:
                futs = [
                    exc.submit(_read_file, fp) for _ in range(nt)
                ]
                for fut in futs:
                    res = fut.result()
                    np.testing.assert_array_equal(res, data)
            t0 = time.time() - t0
            assert t0 > 1.0
