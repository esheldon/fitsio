import numpy as np

import pytest

from ..util import _nans_as_cfitsio_null_value, cfitsio_version
from .. import DOUBLE_NULL_VALUE, FLOAT_NULL_VALUE

CFITSIO_VERSION = cfitsio_version(asfloat=True)
DTYPES = ['u1', 'i1', 'u2', 'i2', '<u4', 'i4', 'i8', '>f4', 'f8']
if CFITSIO_VERSION > 3.44:
    DTYPES += ["u8"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_nan", [True, False])
@pytest.mark.parametrize("hdu_is_compressed", [True, False])
def test_nans_as_cfitsio_null_value(dtype, with_nan, hdu_is_compressed):
    data = np.arange(5 * 20, dtype=dtype).reshape(5, 20)
    if "f" in dtype and with_nan:
        data[3, 13] = np.nan

    with _nans_as_cfitsio_null_value(data, hdu_is_compressed) as (
        nan_data,
        any_nan,
    ):
        if with_nan and "f" in dtype and hdu_is_compressed:
            assert any_nan

            msk_nan = np.isnan(data)

            if "4" in dtype:
                np.testing.assert_array_equal(data[msk_nan], FLOAT_NULL_VALUE)
            else:
                np.testing.assert_array_equal(data[msk_nan], DOUBLE_NULL_VALUE)

            np.testing.assert_array_equal(data[~msk_nan], nan_data[~msk_nan])
        else:
            assert not any_nan
            np.testing.assert_array_equal(nan_data, data)
