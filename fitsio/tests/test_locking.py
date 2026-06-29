from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import sys
import threading
import tempfile
import time
import uuid

import fitsio


def test_locking_read():
    nt = 10
    rng = np.random.RandomState(seed=10)
    data = rng.normal(size=(100, 100))

    lock = threading.RLock()

    def _read_file(fp):
        # older pythons need a lock
        if sys.version_info.major < 3 or sys.version_info.minor < 13:
            with lock:
                time.sleep(0.1)
                fp.write_image(data)
                return fp[0].read()
        else:
            time.sleep(0.1)
            fp.write_image(data)
            return fp[0].read()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "fname" + str(uuid.uuid4().hex) + ".fits")

        with fitsio.FITS(fname, "rw", clobber=True) as fp:
            fp.write_image(data)

        with fitsio.FITS(fname, "rw") as fp:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=nt) as exc:
                futs = [exc.submit(_read_file, fp) for _ in range(nt)]
                for fut in futs:
                    res = fut.result()
                    np.testing.assert_array_equal(res, data)
            t0 = time.time() - t0
            assert len(fp) == nt + 1
            if sys.version_info.major < 3 or sys.version_info.minor < 13:
                assert t0 > 1.0
            else:
                # locking in the C layer is much more efficient
                assert t0 > 0.1
