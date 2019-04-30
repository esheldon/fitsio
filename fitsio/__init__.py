"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

__version__ = '1.0.3'

from . import fitslib   # noqa

from .fitslib import (    # noqa
    FITS,
    read,
    read_header,
    read_scamp_head,
    write,
    READONLY,
    READWRITE,
)

from .header import FITSHDR, FITSRecord, FITSCard  # noqa
from .hdu import BINARY_TBL, ASCII_TBL, IMAGE_HDU  # noqa

from . import util  # noqa
from .util import cfitsio_version, FITSRuntimeWarning  # noqa

from . import test  # noqa
