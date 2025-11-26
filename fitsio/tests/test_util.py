import math
import numpy as np

import pytest

from ..util import (
    _nonfinite_as_cfitsio_floating_null_value,
    cfitsio_version,
    _FLOATING_NULL_VALUE,
)

CFITSIO_VERSION = cfitsio_version(asfloat=True)
DTYPES = ['u1', 'i1', 'u2', 'i2', '<u4', 'i4', 'i8', '>f4', 'f8']
if CFITSIO_VERSION > 3.44:
    DTYPES += ["u8"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_nan", [True, False])
@pytest.mark.parametrize("hdu_is_compressed", [True, False])
def test_nonfinite_as_cfitsio_floating_null_value(
    dtype, with_nan, hdu_is_compressed
):
    data = np.arange(5 * 20, dtype=dtype).reshape(5, 20)
    if "f" in dtype and with_nan:
        data[3, 13] = np.nan
        data[1, 7] = np.inf
        data[1, 9] = -np.inf

    with _nonfinite_as_cfitsio_floating_null_value(
        data, hdu_is_compressed
    ) as (
        nan_data,
        any_nan,
    ):
        if with_nan and "f" in dtype and hdu_is_compressed:
            assert any_nan
            assert not np.any(np.isnan(nan_data))
            msk = ~np.isfinite(data)
            np.testing.assert_array_equal(data[msk], _FLOATING_NULL_VALUE)
            np.testing.assert_array_equal(data[~msk], nan_data[~msk])
        else:
            assert not any_nan
            np.testing.assert_array_equal(nan_data, data)

    if with_nan and "f" in dtype:
        np.testing.assert_array_equal(data[3, 13], np.nan)
        np.testing.assert_array_equal(data[1, 7], np.inf)
        np.testing.assert_array_equal(data[1, 9], -np.inf)


def test_cfitsio_floating_null_value_equal_inf():
    assert np.float64(np.inf) == _FLOATING_NULL_VALUE
    assert np.float32(np.inf) == _FLOATING_NULL_VALUE
    assert np.inf == _FLOATING_NULL_VALUE
    assert math.inf == _FLOATING_NULL_VALUE


def test_nonfinite_as_cfitsio_floating_null_value_with_exception():
    data = np.arange(5 * 20, dtype=np.float32).reshape(5, 20)
    data[3, 13] = np.nan
    data[1, 7] = np.inf
    data[1, 9] = -np.inf

    with pytest.raises(RuntimeError):
        with _nonfinite_as_cfitsio_floating_null_value(data, True) as (
            nan_data,
            any_nan,
        ):
            assert any_nan
            assert not np.any(np.isnan(nan_data))
            msk = ~np.isfinite(data)
            np.testing.assert_array_equal(data[msk], _FLOATING_NULL_VALUE)
            np.testing.assert_array_equal(data[~msk], nan_data[~msk])
            raise RuntimeError("Exception raised while data is modified!")

    assert data[1, 7] == np.inf
    assert data[1, 9] == -np.inf
