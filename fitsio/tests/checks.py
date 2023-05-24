import sys
import numpy as np
from .. import util


def check_header(header, rh):
    for k in header:
        v = header[k]
        rv = rh[k]

        if isinstance(rv, str):
            v = v.strip()
            rv = rv.strip()

        assert v == rv, "testing equal key '%s'" % k


def compare_headerlist_header(header_list, header):
    """
    The first is a list of dicts, second a FITSHDR
    """
    for entry in header_list:
        name = entry['name'].upper()
        value = entry['value']
        hvalue = header[name]

        if isinstance(hvalue, str):
            hvalue = hvalue.strip()

        assert value == hvalue, (
            "testing header key '%s'" % name
        )

        if 'comment' in entry:
            assert (
                entry['comment'].strip() ==
                header.get_comment(name).strip()
            ), "testing comment for header key '%s'" % name


def cast_shape(shape):
    if len(shape) == 2 and shape[1] == 1:
        return (shape[0], )
    elif shape == (1, ):
        return tuple()
    else:
        return shape


def compare_array(arr1, arr2, name):
    arr1_shape = cast_shape(arr1.shape)
    arr2_shape = cast_shape(arr2.shape)

    assert arr1_shape == arr2_shape, (
        "testing arrays '%s' shapes are equal: "
        "input %s, read: %s" % (name, arr1_shape, arr2_shape)
    )

    if sys.version_info >= (3, 0, 0) and arr1.dtype.char == 'S':
        _arr1 = arr1.astype('U')
    else:
        _arr1 = arr1

    res = np.where(_arr1 != arr2)
    for i, w in enumerate(res):
        assert w.size == 0, "testing array '%s' dim %d are equal" % (name, i)


def compare_array_tol(arr1, arr2, tol, name):
    assert arr1.shape == arr2.shape, (
        "testing arrays '%s' shapes are equal: "
        "input %s, read: %s" % (name, arr1.shape, arr2.shape)
    )

    adiff = np.abs((arr1 - arr2)/arr1)
    maxdiff = adiff.max()
    res = np.where(adiff > tol)
    for i, w in enumerate(res):
        assert w.size == 0, (
            "testing array '%s' dim %d are "
            "equal within tolerance %e, found "
            "max diff %e" % (name, i, tol, maxdiff)
        )


def compare_array_abstol(arr1, arr2, tol, name):
    assert arr1.shape == arr2.shape, (
        "testing arrays '%s' shapes are equal: "
        "input %s, read: %s" % (name, arr1.shape, arr2.shape)
    )

    adiff = np.abs(arr1-arr2)
    maxdiff = adiff.max()
    res = np.where(adiff > tol)
    for i, w in enumerate(res):
        assert w.size == 0, (
            "testing array '%s' dim %d are "
            "equal within tolerance %e, found "
            "max diff %e" % (name, i, tol, maxdiff)
        )


def compare_object_array(arr1, arr2, name, rows=None):
    """
    The first must be object
    """
    if rows is None:
        rows = np.arange(arr1.size)

    for i, row in enumerate(rows):
        if ((sys.version_info >= (3, 0, 0) and isinstance(arr2[i], bytes))
                or isinstance(arr2[i], str)):

            if sys.version_info >= (3, 0, 0) and isinstance(arr1[row], bytes):
                _arr1row = arr1[row].decode('ascii')
            else:
                _arr1row = arr1[row]

            assert _arr1row == arr2[i], (
                "%s str el %d equal" % (name, i)
            )
        else:
            delement = arr2[i]
            orig = arr1[row]
            s = len(orig)
            compare_array(
                orig, delement[0:s], "%s num el %d equal" % (name, i)
            )


def compare_rec(rec1, rec2, name):
    for f in rec1.dtype.names:
        rec1_shape = cast_shape(rec1[f].shape)
        rec2_shape = cast_shape(rec2[f].shape)

        assert rec1_shape == rec2_shape, (
            "testing '%s' field '%s' shapes are equal: "
            "input %s, read: %s" % (
                name, f, rec1_shape, rec2_shape)
        )

        if sys.version_info >= (3, 0, 0) and rec1[f].dtype.char == 'S':
            # for python 3, we get back unicode always
            _rec1f = rec1[f].astype('U')
        else:
            _rec1f = rec1[f]

        assert np.all(_rec1f == rec2[f])
        # res = np.where(_rec1f != rec2[f])
        # for w in res:
        #     assert w.size == 0, "testing column %s" % f


def compare_rec_subrows(rec1, rec2, rows, name):
    for f in rec1.dtype.names:
        rec1_shape = cast_shape(rec1[f][rows].shape)
        rec2_shape = cast_shape(rec2[f].shape)

        assert rec1_shape == rec2_shape, (
            "testing '%s' field '%s' shapes are equal: "
            "input %s, read: %s" % (
                name, f, rec1_shape, rec2_shape)
        )

        if sys.version_info >= (3, 0, 0) and rec1[f].dtype.char == 'S':
            # for python 3, we get back unicode always
            _rec1frows = rec1[f][rows].astype('U')
        else:
            _rec1frows = rec1[f][rows]

        res = np.where(_rec1frows != rec2[f])
        for w in res:
            assert w.size == 0, "testing column %s" % f


def compare_rec_with_var(rec1, rec2, name, rows=None):
    """

    First one *must* be the one with object arrays

    Second can have fixed length

    both should be same number of rows

    """

    if rows is None:
        rows = np.arange(rec2.size)
        assert rec1.size == rec2.size, (
            "testing '%s' same number of rows" % name
        )

    # rec2 may have fewer fields
    for f in rec2.dtype.names:

        # f1 will have the objects
        if util.is_object(rec1[f]):
            compare_object_array(
                rec1[f], rec2[f],
                "testing '%s' field '%s'" % (name, f),
                rows=rows
            )
        else:
            compare_array(
                rec1[f][rows], rec2[f],
                "testing '%s' num field '%s' equal" % (name, f)
            )


def compare_names(read_names, true_names, lower=False, upper=False):
    for nread, ntrue in zip(read_names, true_names):
        if lower:
            tname = ntrue.lower()
            mess = "lower: '%s' vs '%s'" % (nread, tname)
        else:
            tname = ntrue.upper()
            mess = "upper: '%s' vs '%s'" % (nread, tname)

        assert nread == tname, mess
