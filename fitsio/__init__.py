"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

__version__ = '1.0.1'

from . import fitslib   # noqa

from .fitslib import (    # noqa
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

from . import util  # noqa
from .util import cfitsio_version  # noqa

from . import test  # noqa
