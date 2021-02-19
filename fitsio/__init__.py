# flake8: noqa
"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

__version__ = '1.1.4'

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
)

from .header import FITSHDR, FITSRecord, FITSCard
from .hdu import BINARY_TBL, ASCII_TBL, IMAGE_HDU

from . import util
from .util import cfitsio_version, FITSRuntimeWarning

from . import test
