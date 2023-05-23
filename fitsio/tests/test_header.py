import os
import tempfile
import numpy as np
from .makedata import make_data, lorem_ipsum
from .checks import check_header, compare_headerlist_header
from ..fitslib import FITS, read_header
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


def testHeaderCommentPreserved():
    """
    Test that the comment is preserved after resetting the value
    """

    l1 = 'KEY1    =                   77 / My comment1'
    l2 = 'KEY2    =                   88 / My comment2'
    hdr = FITSHDR()
    hdr.add_record(l1)
    hdr.add_record(l2)

    hdr['key1'] = 99
    assert hdr.get_comment('key1') == 'My comment1', (
        'comment not preserved'
    )


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


def test_header_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.zeros(10)
            header1 = {
                'SCARD': 'one',
                'ICARD': 1,
                'FCARD': 1.0,
                'LCARD': True
            }
            header2 = {
                'SCARD': 'two',
                'ICARD': 2,
                'FCARD': 2.0,
                'LCARD': False,

                'SNEW': 'two',
                'INEW': 2,
                'FNEW': 2.0,
                'LNEW': False
            }
            fits.write_image(data, header=header1)
            rh = fits[0].read_header()
            check_header(header1, rh)

            # Update header
            fits[0].write_keys(header2)

        with FITS(fname) as fits:
            rh = fits[0].read_header()
            check_header(header2, rh)


def test_read_header_case():
    """
    Test read_header with and without case sensitivity

    The reason we need a special test for this is because
    the read_header code is optimized for speed and has
    a different code path
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.zeros(10)
            _, keys, _, _, _, _ = make_data()
            fits.write_image(data, header=keys, extname='First')
            fits.write_image(data, header=keys, extname='second')

        cases = [
            ('First', True),
            ('FIRST', False),
            ('second', True),
            ('seConD', False),
        ]
        for ext, ci in cases:
            h = read_header(fname, ext=ext, case_sensitive=ci)
            compare_headerlist_header(keys, h)


def test_blank_key_comments():
    """
    test a few different comments
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            records = [
                # empty should return empty
                {'name': None, 'value': '', 'comment': ''},
                # this will also return empty
                {'name': None, 'value': '', 'comment': ' '},
                # this will return exactly
                {'name': None, 'value': '', 'comment': ' h'},
                # this will return exactly
                {'name': None, 'value': '', 'comment': '--- test comment ---'},
            ]
            header = FITSHDR(records)

            fits.write(None, header=header)

            rh = fits[0].read_header()

            rrecords = rh.records()

            for i, ri in ((0, 6), (1, 7), (2, 8)):
                rec = records[i]
                rrec = rrecords[ri]

                assert rec['name'] is None, (
                    'checking name is None'
                )

                comment = rec['comment']
                rcomment = rrec['comment']
                if '' == comment.strip():
                    comment = ''

                assert comment == rcomment, (
                    "check empty key comment"
                )


def test_blank_key_comments_from_cards():
    """
    test a few different comments
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            records = [
                '                                                                                ',  # noqa
                '         --- testing comment ---                                                ',  # noqa
                '        --- testing comment ---                                                 ',  # noqa
                "COMMENT testing                                                                 ",  # noqa
            ]
            header = FITSHDR(records)

            fits.write(None, header=header)

            rh = fits[0].read_header()

            rrecords = rh.records()

            assert rrecords[6]['name'] is None, (
                'checking name is None'
            )
            assert rrecords[6]['comment'] == '', (
                'check empty key comment'
            )
            assert rrecords[7]['name'] is None, (
                'checking name is None'
            )
            assert rrecords[7]['comment'] == ' --- testing comment ---', (
                "check empty key comment"
            )
            assert rrecords[8]['name'] is None, (
                'checking name is None'
            )
            assert rrecords[8]['comment'] == '--- testing comment ---', (
                "check empty key comment"
            )
            assert rrecords[9]['name'] == 'COMMENT', (
                'checking name is COMMENT'
            )
            assert rrecords[9]['comment'] == 'testing', (
                "check comment"
            )


def test_header_from_cards():
    """
    test generating a header from cards, writing it out and getting
    back what we put in
    """
    hdr_from_cards = FITSHDR([
        "IVAL    =                   35 / integer value                                  ",  # noqa
        "SHORTS  = 'hello world'                                                         ",  # noqa
        "UND     =                                                                       ",  # noqa
        "LONGS   = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiu&'",  # noqa
        "CONTINUE  'smod tempor incididunt ut labore et dolore magna aliqua'             ",  # noqa
        "DBL     =                 1.25                                                  ",  # noqa
    ])
    header = [
        {'name': 'ival', 'value': 35, 'comment': 'integer value'},
        {'name': 'shorts', 'value': 'hello world'},
        {'name': 'und', 'value': None},
        {'name': 'longs', 'value': lorem_ipsum},
        {'name': 'dbl', 'value': 1.25},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.zeros(10)
            fits.write_image(data, header=hdr_from_cards)

            rh = fits[0].read_header()
            compare_headerlist_header(header, rh)

        with FITS(fname) as fits:
            rh = fits[0].read_header()
            compare_headerlist_header(header, rh)
