"""
utilities for the fits library
"""

from contextlib import contextmanager
import sys
import numpy

from . import _fitsio_wrap

if sys.version_info >= (3, 0, 0):
    IS_PY3 = True
else:
    IS_PY3 = False

FLOAT_NULL_VALUE = _fitsio_wrap.cfitsio_float_null_value()
DOUBLE_NULL_VALUE = _fitsio_wrap.cfitsio_double_null_value()


class FITSRuntimeWarning(RuntimeWarning):
    pass


def cfitsio_version(asfloat=False):
    """
    Return the cfitsio version as a string.
    """
    # use string version to avoid roundoffs
    ver = '%0.3f' % _fitsio_wrap.cfitsio_version()
    if asfloat:
        return float(ver)
    else:
        return ver


def cfitsio_is_bundled():
    """Return True if library was built with a
    bundled copy of cfitsio.
    """
    return _fitsio_wrap.cfitsio_is_bundled()


if sys.version_info > (3, 0, 0):
    _itypes = (int,)
    _stypes = (str, bytes)
else:
    _itypes = (int, long)  # noqa - only for py2
    _stypes = (
        basestring,  # noqa - only for py2
        unicode,  # noqa - only for py2
    )  # noqa - only for py2

_itypes += (
    numpy.uint8,
    numpy.int8,
    numpy.uint16,
    numpy.int16,
    numpy.uint32,
    numpy.int32,
    numpy.uint64,
    numpy.int64,
)

# different for py3
if numpy.lib.NumpyVersion(numpy.__version__) < "1.28.0":
    _stypes += (
        numpy.string_,
        numpy.str_,
    )
else:
    _stypes += (
        numpy.bytes_,
        numpy.str_,
    )

# for header keywords
_ftypes = (float, numpy.float32, numpy.float64)


def isstring(arg):
    return isinstance(arg, _stypes)


def isinteger(arg):
    return isinstance(arg, _itypes)


def is_object(arr):
    if arr.dtype.descr[0][1][1] == 'O':
        return True
    else:
        return False


def fields_are_object(arr):
    isobj = numpy.zeros(len(arr.dtype.names), dtype=bool)
    for i, name in enumerate(arr.dtype.names):
        if is_object(arr[name]):
            isobj[i] = True
    return isobj


def is_little_endian(array):
    """
    Return True if array is little endian, False otherwise.

    Parameters
    ----------
    array: numpy array
        A numerical python array.

    Returns
    -------
    Truth value:
        True for little-endian

    Notes
    -----
    Strings are neither big or little endian.  The input must be a simple numpy
    array, not an array with fields.
    """
    if numpy.little_endian:
        machine_little = True
    else:
        machine_little = False

    byteorder = array.dtype.base.byteorder
    return (byteorder == '<') or (machine_little and byteorder == '=')


def array_to_native(array, inplace=False):
    """
    Convert an array to the native byte order.

    NOTE: the inplace keyword argument is not currently used.
    """
    if numpy.little_endian:
        machine_little = True
    else:
        machine_little = False

    data_little = False
    if array.dtype.names is None:
        if array.dtype.base.byteorder == '|':
            # strings and 1 byte integers
            return array

        data_little = is_little_endian(array)
    else:
        # assume all are same byte order: we only need to find one with
        # little endian
        for fname in array.dtype.names:
            if is_little_endian(array[fname]):
                data_little = True
                break

    if (machine_little and not data_little) or (
        not machine_little and data_little
    ):
        output = array.byteswap(inplace)
    else:
        output = array

    return numpy.require(output, requirements=['ALIGNED'])


if numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0":
    copy_if_needed = None
elif numpy.lib.NumpyVersion(numpy.__version__) < "1.28.0":
    copy_if_needed = False
else:
    # 2.0.0 dev versions, handle cases where copy may or may not exist
    try:
        numpy.array([1]).__array__(copy=None)
        copy_if_needed = None
    except TypeError:
        copy_if_needed = False


def array_to_native_c(array_in, inplace=False):
    # copy only made if not C order
    arr = numpy.require(
        array_in,
        requirements=['C_CONTIGUOUS', 'ALIGNED'],
    )
    return array_to_native(arr, inplace=inplace)


def mks(val):
    """
    make sure the value is a string, paying mind to python3 vs 2
    """
    if sys.version_info > (3, 0, 0):
        if isinstance(val, bytes):
            sval = str(val, 'utf-8')
        else:
            sval = str(val)
    else:
        sval = str(val)

    return sval


@contextmanager
def _nans_as_cfitsio_null_value(data):
    has_nan = False
    if data is not None and data.dtype.kind == "f":
        msk_nan = numpy.isnan(data)
        if numpy.any(msk_nan):
            has_nan = True
            if data.dtype.itemsize == 8:
                if numpy.any(data == DOUBLE_NULL_VALUE):
                    raise RuntimeError(
                        "Array has both NaNs and values equal to the "
                        "cfitsio sentinel value for nulls (%.27e). "
                        "Thus the data cannot be correctly written "
                        "to a FITS file." % DOUBLE_NULL_VALUE
                    )
                data[msk_nan] = DOUBLE_NULL_VALUE
            else:
                if numpy.any(data == FLOAT_NULL_VALUE):
                    raise RuntimeError(
                        "Array has both NaNs and values equal to the "
                        "cfitsio sentinel value for nulls (%.27e). "
                        "Thus the data cannot be correctly written "
                        "to a FITS file." % FLOAT_NULL_VALUE
                    )
                data[msk_nan] = FLOAT_NULL_VALUE
    else:
        msk_nan = None

    yield data, has_nan

    if msk_nan is not None:
        data[msk_nan] = numpy.nan
