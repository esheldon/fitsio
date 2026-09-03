# flake8: noqa
"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = None

from . import fitslib

from .fitslib import (
    FITS,
    read,
    read_header,
    read_scamp_head,
    write,
    READONLY,
    READWRITE,
    NOCOMPRESS,
    RICE_1,
    GZIP_1,
    GZIP_2,
    PLIO_1,
    HCOMPRESS_1,
    NO_DITHER,
    SUBTRACTIVE_DITHER_1,
    SUBTRACTIVE_DITHER_2,
    NOT_SET,
)

from .header import FITSHDR, FITSRecord, FITSCard
from .hdu import BINARY_TBL, ASCII_TBL, IMAGE_HDU

from . import util
from .util import (
    cfitsio_version,
    FITSRuntimeWarning,
    cfitsio_is_bundled,
)

backend_version = cfitsio_version
backend_is_bundled = cfitsio_is_bundled
from ._fitsio_wrap import (
    cfitsio_has_bzip2_support,
    cfitsio_has_curl_support,
    cfitsio_is_reentrant,
    fitsio_backend,
)

backend_has_bzip2_support = cfitsio_has_bzip2_support
backend_has_curl_support = cfitsio_has_curl_support
backend_is_reentrant = cfitsio_is_reentrant

# return values of fitsio_backend, here to help make code
# clearer and avoid mispelling errors in testing strings
CFITSIO_BACKEND = "cfitsio"
RSFITSIO_BACKEND = "rsfitsio"

from .fits_exceptions import FITSFormatError
