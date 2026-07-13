from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import threading
import tempfile
import uuid

import fitsio


def test_locking_io():
    nt = 10
    rng = np.random.RandomState(seed=10)
    data = rng.normal(size=(100, 100))

    lock = threading.RLock()

    def _read_file(fp):
        with lock:
            fp.reopen()
            fp.write_image(data)
            return fp[0].read()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(
            tmpdir,
            "fname" + str(uuid.uuid4().hex) + ".fits",
        )

        with fitsio.FITS(fname, "rw", clobber=True) as fp:
            fp.write_image(data)

        with fitsio.FITS(fname, "rw") as fp:
            with ThreadPoolExecutor(max_workers=nt) as exc:
                futs = [exc.submit(_read_file, fp) for _ in range(nt)]
                for fut in futs:
                    res = fut.result()
                    np.testing.assert_array_equal(res, data)

        with fitsio.FITS(fname) as fp:
            n_ext = len(fp)

        assert n_ext == nt + 1
