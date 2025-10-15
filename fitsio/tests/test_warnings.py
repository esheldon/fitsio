import os
import tempfile
import warnings
import numpy as np
from ..fitslib import FITS
from ..util import FITSRuntimeWarning


def test_non_standard_key_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        im = np.zeros((3, 3))
        with warnings.catch_warnings(record=True) as w:
            with FITS(fname, 'rw') as fits:
                fits.write(im)

                # now write a key with a non-standard value
                value = {'test': 3}
                fits[-1].write_key('odd', value)

            # DeprecationWarnings have crept into the Warning list.  This will
            # filter the list to be just
            # FITSRuntimeWarning instances.
            # @at88mph  2019.10.09
            filtered_warnings = list(
                filter(lambda x: 'FITSRuntimeWarning' in '{}'.format(x.category), w)  # noqa
            )

            assert len(filtered_warnings) == 1, (
                'Wrong length of output (Expected {} but got {}.)'.format(
                    1, len(filtered_warnings),
                )
            )
            assert issubclass(
                filtered_warnings[-1].category, FITSRuntimeWarning,
            )
