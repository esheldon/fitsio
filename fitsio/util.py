"""
utilities for the fits library
"""
import sys
import numpy

from . import _fitsio_wrap

if sys.version_info >= (3, 0, 0):
    IS_PY3 = True
else:
    IS_PY3 = False


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


if sys.version_info > (3, 0, 0):
    _itypes = (int,)
    _stypes = (str, bytes)
else:
    _itypes = (int, long)  # noqa - only for py2
    _stypes = (basestring, unicode,)  # noqa - only for py2

_itypes += (numpy.uint8, numpy.int8,
            numpy.uint16, numpy.int16,
            numpy.uint32, numpy.int32,
            numpy.uint64, numpy.int64)

# different for py3
_stypes += (numpy.string_, numpy.str_)

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

    if ((machine_little and not data_little)
            or (not machine_little and data_little)):
        output = array.byteswap(inplace)
    else:
        output = array

    return output


def array_to_native_c(array_in, inplace=False):
    # copy only made if not C order
    arr = numpy.array(array_in, order='C', copy=False)
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
