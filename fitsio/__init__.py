"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

__version__='1.0.1'

from . import fitslib

from .fitslib import (
    FITS,
    FITSHDR,
    FITSRecord,
    FITSCard,
    read,
    read_header,
    read_scamp_head,
    write,
    READONLY,
    READWRITE,
    BINARY_TBL, ASCII_TBL, IMAGE_HDU,
    FITSRuntimeWarning,
)

from . import util
from .util import cfitsio_version

from . import test
