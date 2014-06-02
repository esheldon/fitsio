"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

__version__='0.9.5'

from . import fitslib
from .fitslib import FITS
from .fitslib import FITSHDR
from .fitslib import read
from .fitslib import read_header
from .fitslib import write
from .fitslib import READONLY
from .fitslib import READWRITE
from .fitslib import cfitsio_version

from .fitslib import BINARY_TBL, ASCII_TBL, IMAGE_HDU


from . import test
