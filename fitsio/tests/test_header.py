import os
import tempfile
import warnings
import numpy as np
from .makedata import make_data, lorem_ipsum
from .checks import check_header, compare_headerlist_header
from ..fitslib import FITS, read_header, write
from ..header import FITSHDR
from ..hdu.base import INVALID_HDR_CHARS


def test_free_form_string():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        with open(fname, 'w') as f:
            s = ("SIMPLE  =                    T / Standard FITS                                  " + # noqa
                 "BITPIX  =                   16 / number of bits per data pixel                  " + # noqa
                 "NAXIS   =                    0 / number of data axes                            " + # noqa
                 "EXTEND  =                    T / File contains extensions                       " + # noqa
                 "PHOTREF =   'previous MegaCam' / Source: cum.photcat                            " + # noqa
                 "EXTRA   =                    7 / need another line following PHOTREF            " + # noqa
                 "END                                                                             " # noqa
                 )
            f.write(s + ' ' * (2880-len(s)))
        hdr = read_header(fname)
        assert hdr['PHOTREF'] == 'previous MegaCam'


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
                # force hierarch + continue
                "long_keyword_name": lorem_ipsum,
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
            adata = make_data()
            fits.write_image(data, header=adata['keys'], extname='First')
            fits.write_image(data, header=adata['keys'], extname='second')

        cases = [
            ('First', True),
            ('FIRST', False),
            ('second', True),
            ('seConD', False),
        ]
        for ext, ci in cases:
            h = read_header(fname, ext=ext, case_sensitive=ci)
            compare_headerlist_header(adata['keys'], h)


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


def test_bad_header_write_raises():
    """
    Test that an invalid header raises.
    """

    for c in INVALID_HDR_CHARS:
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.fits')
            try:
                hdr = {'bla%sg' % c: 3}
                data = np.zeros(10)

                write(fname, data, header=hdr, clobber=True)
            except Exception as e:
                assert "header key 'BLA%sG' has" % c in str(e)


def test_header_template():
    """
    test adding bunch of cards from a split template
    """

    header_template = """SIMPLE  =                    T /
BITPIX  =                    8 / bits per data value
NAXIS   =                    0 / number of axes
EXTEND  =                    T / Extensions are permitted
ORIGIN  = 'LSST DM Header Service'/ FITS file originator

     ---- Date, night and basic image information ----
DATE    =                      / Creation Date and Time of File
DATE-OBS=                      / Date of the observation (image acquisition)
DATE-BEG=                      / Time at the start of integration
DATE-END=                      / end date of the observation
MJD     =                      / Modified Julian Date that the file was written
MJD-OBS =                      / Modified Julian Date of observation
MJD-BEG =                      / Modified Julian Date derived from DATE-BEG
MJD-END =                      / Modified Julian Date derived from DATE-END
OBSID   =                      / ImageName from Camera StartIntergration
GROUPID =                      / imageSequenceName from StartIntergration
OBSTYPE =                      / BIAS, DARK, FLAT, OBJECT
BUNIT   = 'adu     '           / Brightness units for pixel array

     ---- Telescope info, location, observer ----
TELESCOP= 'LSST AuxTelescope'  / Telescope name
INSTRUME= 'LATISS'             / Instrument used to obtain these data
OBSERVER= 'LSST'               / Observer name(s)
OBS-LONG=           -70.749417 / [deg] Observatory east longitude
OBS-LAT =           -30.244639 / [deg] Observatory latitude
OBS-ELEV=               2663.0 / [m] Observatory elevation
OBSGEO-X=           1818938.94 / [m] X-axis Geocentric coordinate
OBSGEO-Y=          -5208470.95 / [m] Y-axis Geocentric coordinate
OBSGEO-Z=          -3195172.08 / [m] Z-axis Geocentric coordinate

    ---- Pointing info, etc. ----

DECTEL  =                      / Telescope DEC of observation
ROTPATEL=                      / Telescope Rotation
ROTCOORD= 'sky'                / Telescope Rotation Coordinates
RA      =                      / RA of Target
DEC     =                      / DEC of Target
ROTPA   =                      / Rotation angle relative to the sky (deg)
HASTART =                      / [HH:MM:SS] Telescope hour angle at start
ELSTART =                      / [deg] Telescope zenith distance at start
AZSTART =                      / [deg] Telescope azimuth angle at start
AMSTART =                      / Airmass at start
HAEND   =                      / [HH:MM:SS] Telescope hour angle at end
ELEND   =                      / [deg] Telescope zenith distance at end
AZEND   =                      / [deg] Telescope azimuth angle at end
AMEND   =                      / Airmass at end

    ---- Image-identifying used to build OBS-ID ----
TELCODE = 'AT'                 / The code for the telecope
CONTRLLR=                      / The controller (e.g. O for OCS, C for CCS)
DAYOBS  =                      / The observation day as defined by image name
SEQNUM  =                      / The sequence number from the image name
GROUPID =                      /

    ---- Information from Camera
CCD_MANU= 'ITL'                / CCD Manufacturer
CCD_TYPE= '3800C'              / CCD Model Number
CCD_SERN= '20304'              / Manufacturers? CCD Serial Number
LSST_NUM= 'ITL-3800C-098'      / LSST Assigned CCD Number
SEQCKSUM=                      / Checksum of Sequencer
SEQNAME =                      / SequenceName from Camera StartIntergration
REBNAME =                      / Name of the REB
CONTNUM =                      / CCD Controller (WREB) Serial Number
IMAGETAG=                      / DAQ Image id
TEMP_SET=                      / Temperature set point (deg C)
CCDTEMP =                      / Measured temperature (deg C)

    ---- Geometry from Camera ----
DETSIZE =                      / Size of sensor
OVERH   =                      / Over-scan pixels
OVERV   =                      / Vert-overscan pix
PREH    =                      / Pre-scan pixels

    ---- Filter/grating information ----
FILTER  =                      / Name of the filter
FILTPOS =                      / Filter position
GRATING =                      / Name of the second disperser
GRATPOS =                      / disperser position
LINSPOS =                      / Linear Stage

    ---- Exposure-related information ----
EXPTIME =                      / Exposure time in seconds
SHUTTIME=                      / Shutter exposure time in seconds
DARKTIME=                      / Dark time in seconds

    ---- Header information ----
FILENAME=                      / Original file name
HEADVER =                      / Version of header

    ---- Checksums ----
CHECKSUM=                      / checksum for the current HDU
DATASUM =                      / checksum of the data records\n"""

    lines = header_template.splitlines()
    hdr = FITSHDR()
    for line in lines:
        hdr.add_record(line)


def test_corrupt_continue():
    """
    test with corrupt continue, just make sure it doesn't crash
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        with warnings.catch_warnings(record=True) as _:

            hdr_from_cards = FITSHDR([
                "IVAL    =                   35 / integer value                                  ",  # noqa
                "SHORTS  = 'hello world'                                                         ",  # noqa
                "CONTINUE= '        '           /   '&' / Current observing orogram              ",  # noqa
                "UND     =                                                                       ",  # noqa
                "DBL     =                 1.25                                                  ",  # noqa
            ])

            with FITS(fname, 'rw') as fits:
                fits.write(None, header=hdr_from_cards)

            read_header(fname)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        with warnings.catch_warnings(record=True) as _:

            hdr_from_cards = FITSHDR([
                "IVAL    =                   35 / integer value                                  ",  # noqa
                "SHORTS  = 'hello world'                                                         ",  # noqa
                "PROGRAM = 'Setting the Scale: Determining the Absolute Mass Normalization and &'",  # noqa
                "CONTINUE  'Scaling Relations for Clusters at z~0.1&'                            ",  # noqa
                "CONTINUE  '&' / Current observing orogram                                       ",  # noqa
                "UND     =                                                                       ",  # noqa
                "DBL     =                 1.25                                                  ",  # noqa
            ])

            with FITS(fname, 'rw') as fits:
                fits.write(None, header=hdr_from_cards)

            read_header(fname)


def record_exists(header_records, key, value):
    for rec in header_records:
        if rec['name'] == key and rec['value'] == value:
            return True

    return False


def test_read_comment_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            data = np.arange(100).reshape(10, 10)
            fits.create_image_hdu(data)
            hdu = fits[-1]
            hdu.write_comment('A COMMENT 1')
            hdu.write_comment('A COMMENT 2')
            hdu.write_history('SOME HISTORY 1')
            hdu.write_history('SOME HISTORY 2')
            fits.close()

        with FITS(fname, 'r') as fits:
            hdu = fits[-1]
            header = hdu.read_header()
            records = header.records()
            assert record_exists(records, 'COMMENT', 'A COMMENT 1')
            assert record_exists(records, 'COMMENT', 'A COMMENT 2')
            assert record_exists(records, 'HISTORY', 'SOME HISTORY 1')
            assert record_exists(records, 'HISTORY', 'SOME HISTORY 2')


def test_write_key_dict():
    """
    test that write_key works using a standard key dict
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        with FITS(fname, 'rw') as fits:

            im = np.zeros((10, 10), dtype='i2')
            fits.write(im)

            keydict = {
                'name': 'test',
                'value': 35,
                'comment': 'keydict test',
            }
            fits[-1].write_key(**keydict)

            h = fits[-1].read_header()

            assert h['test'] == keydict['value']
            assert h.get_comment('test') == keydict['comment']


if __name__ == '__main__':
    test_header_write_read()
