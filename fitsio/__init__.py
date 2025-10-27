"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = None

from . import fitslib as fitslib

from .fitslib import (
    FITS as FITS,
    read as read,
    read_header as read_header,
    read_scamp_head as read_scamp_head,
    write as write,
    READONLY as READONLY,
    READWRITE as READWRITE,
    NOCOMPRESS as NOCOMPRESS,
    RICE_1 as RICE_1,
    GZIP_1 as GZIP_1,
    GZIP_2 as GZIP_2,
    PLIO_1 as PLIO_1,
    HCOMPRESS_1 as HCOMPRESS_1,
    NO_DITHER as NO_DITHER,
    SUBTRACTIVE_DITHER_1 as SUBTRACTIVE_DITHER_1,
    SUBTRACTIVE_DITHER_2 as SUBTRACTIVE_DITHER_2,
    NOT_SET as NOT_SET,
)

from .header import (
    FITSHDR as FITSHDR,
    FITSRecord as FITSRecord,
    FITSCard as FITSCard
)
from .hdu import (
    BINARY_TBL as BINARY_TBL,
    ASCII_TBL as ASCII_TBL,
    IMAGE_HDU as IMAGE_HDU,
)

from . import util as util
from .util import (
    cfitsio_version as cfitsio_version,
    FITSRuntimeWarning as FITSRuntimeWarning,
    cfitsio_is_bundled as cfitsio_is_bundled,
    FLOAT_NULL_VALUE as FLOAT_NULL_VALUE,
    DOUBLE_NULL_VALUE as DOUBLE_NULL_VALUE,
)
from ._fitsio_wrap import (
    cfitsio_has_bzip2_support as cfitsio_has_bzip2_support,
    cfitsio_has_curl_support as cfitsio_has_curl_support,
)
from .fits_exceptions import FITSFormatError as FITSFormatError
