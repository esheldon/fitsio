"""
A python library to read and write data to FITS files using cfitsio.
See the docs at https://github.com/esheldon/fitsio for example
usage.
"""

__version__='0.9.10'

from . import fitslib
from . import util

from .fitslib import FITS
from .fitslib import FITSHDR
from .fitslib import FITSRecord
from .fitslib import FITSCard

from .fitslib import read
from .fitslib import read_header
from .fitslib import read_scamp_head
from .fitslib import write
from .fitslib import READONLY
from .fitslib import READWRITE

from .fitslib import BINARY_TBL, ASCII_TBL, IMAGE_HDU

from .fitslib import FITSRuntimeWarning

from .util import cfitsio_version

from . import test
