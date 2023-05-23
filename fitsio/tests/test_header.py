import os
import tempfile
import numpy as np
from .makedata import lorem_ipsum
from .checks import check_header
from ..fitslib import FITS
from ..header import FITSHDR


def test_add_delete_and_update_records():
    # Build a FITSHDR from a few records (no need to write on disk)
    # Record names have to be in upper case to match with FITSHDR.add_record
    recs = [
        {'name': "First_record".upper(), 'value': 1,
         'comment': "number 1"},
        {'name': "Second_record".upper(), 'value': "2"},
        {'name': "Third_record".upper(), 'value': "3"},
        {'name': "Last_record".upper(), 'value': 4,
         'comment': "number 4"}
    ]
    hdr = FITSHDR(recs)

    # Add a new record
    hdr.add_record({'name': 'New_record'.upper(), 'value': 5})

    # Delete number 2 and 4
    hdr.delete('Second_record'.upper())
    hdr.delete('Last_record'.upper())

    # Update records : first and new one
    hdr['First_record'] = 11
    hdr['New_record'] = 3

    # Do some checks : len and get value/comment
    assert len(hdr) == 3
    assert hdr['First_record'] == 11
    assert hdr['New_record'] == 3
    assert hdr['Third_record'] == '3'
    assert hdr.get_comment('First_record') == 'number 1'
    assert not hdr.get_comment('New_record')


def test_header_write_read():
    """
    Test a basic header write and read

    Note the other read/write tests also are checking header writing with
    a list of dicts
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.zeros(10)
            header = {
                'x': 35,
                'y': 88.215,
                'eval': 1.384123233e+43,
                'empty': '',
                'funky': '35-8',  # test old bug when strings look
                                  # like expressions
                'name': 'J. Smith',
                'what': '89113e6',  # test bug where converted to float
                'und': None,
                'binop': '25-3',  # test string with binary operation in it
                'unders': '1_000_000',  # test string with underscore
                'longs': lorem_ipsum,
            }
            fits.write_image(data, header=header)

            rh = fits[0].read_header()
            check_header(header, rh)

        with FITS(fname) as fits:
            rh = fits[0].read_header()
            check_header(header, rh)
