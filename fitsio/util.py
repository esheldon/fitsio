"""
utilities for the fits library
"""

from . import _fitsio_wrap

class FITSRuntimeWarning(RuntimeWarning):
    pass

def cfitsio_version(asfloat=False):
    """
    Return the cfitsio version as a string.
    """
    # use string version to avoid roundoffs
    ver= '%0.3f' % _fitsio_wrap.cfitsio_version()
    if asfloat:
        return float(ver)
    else:
        return ver



