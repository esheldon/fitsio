"""
Tests for detecting tile-compressed tables, the ZTABLE convention.

cfitsio cannot read tile-compressed tables, so fitsio detects them and raises
an error with an explanation when the data are accessed.  The header and
metadata remain available.

The test files were written by rustfits, see
fitsio/test_images/make_test_compressed_table.py.  The compressed table has an
uncompressed twin with the same data.
"""

import os
import shutil
import subprocess
import tempfile
import importlib.resources
from contextlib import contextmanager

import numpy as np
import pytest

from .checks import compare_array, compare_rec
from .makedata import make_data
from ..fitslib import FITS, read, read_header
from ..hdu import CompressedTableHDU, TableHDU, ImageHDU

EXTNAME = 'ctable'
NROWS = 300
ZTILELEN = 100
COMPRESSION = {
    'u1scalar': 'RICE_1',
    'i2scalar': 'RICE_1',
    'u2scalar': 'GZIP_2',
    'i4scalar': 'RICE_1',
    'u4scalar': 'RICE_1',
    'i8scalar': 'GZIP_2',
    'f4scalar': 'GZIP_2',
    'f8scalar': 'GZIP_2',
    'Sscalar': 'GZIP_1',
    'bscalar': 'GZIP_1',
    'i4vec': 'GZIP_2',
    'f4arr': 'GZIP_1',
    'f4var': 'GZIP_2',
}


@contextmanager
def _test_files():
    """
    Get the paths to the compressed table file and its uncompressed twin.
    """
    ref = importlib.resources.files('fitsio') / 'test_images'
    fzref = ref / 'test_compressed_table.fits.fz'
    plainref = ref / 'test_compressed_table.fits'
    with importlib.resources.as_file(fzref) as fzname:
        with importlib.resources.as_file(plainref) as plainname:
            yield str(fzname), str(plainname)


def _have_program(name):
    return shutil.which(name) is not None


def test_compressed_table_hdu_types():
    """
    The compressed table is a CompressedTableHDU, and other HDUs are
    unaffected.
    """
    with _test_files() as (fzname, plainname):
        with FITS(fzname) as fits, FITS(plainname) as pfits:
            assert len(fits) == 4
            assert isinstance(fits[0], ImageHDU)
            assert isinstance(fits[1], TableHDU)
            assert not isinstance(fits[1], CompressedTableHDU)
            assert isinstance(fits[2], CompressedTableHDU)
            assert isinstance(fits[2], TableHDU)
            assert isinstance(fits[3], ImageHDU)

            assert fits[EXTNAME] is fits[2]
            assert fits[2].is_compressed()
            assert not fits[1].is_compressed()
            assert fits[1].get_compression() is None

            assert not pfits[2].is_compressed()
            assert not isinstance(pfits[2], CompressedTableHDU)

            # the HDUs around the compressed table read fine
            compare_array(pfits[0].read(), fits[0].read(), 'image before')
            compare_array(pfits[3].read(), fits[3].read(), 'image after')
            compare_rec(pfits[1].read(), fits[1].read(), 'table before')

            # printing works and explains the situation
            rep = repr(fits)
            assert 'BINARY_TBL' in rep
            rep = repr(fits[2])
            assert 'compressed table' in rep
            assert 'cfitsio cannot read' in rep
            assert 'funpack' in rep and 'rustfits' in rep
            assert 'RICE_1' in rep and 'GZIP_2' in rep and 'GZIP_1' in rep
            assert 'rows per tile: %d' % ZTILELEN in rep


def test_compressed_table_info():
    """
    The metadata is that of the uncompressed table, with the compression
    details added.
    """
    with _test_files() as (fzname, plainname):
        with FITS(fzname) as fits, FITS(plainname) as pfits:
            hdu = fits[EXTNAME]
            phdu = pfits[EXTNAME]

            assert hdu.get_nrows() == NROWS
            assert hdu.get_nrows() == phdu.get_nrows()
            assert hdu.get_colnames() == phdu.get_colnames()
            assert hdu.get_extname() == EXTNAME
            assert hdu.get_extnum() == 2
            assert hdu.get_filename() == fzname
            assert hdu.has_data()

            assert hdu.get_compression() == COMPRESSION
            info = hdu.get_info()
            assert info['is_compressed_table'] == 1
            assert info['ztilelen'] == ZTILELEN
            for c in info['colinfo']:
                assert c['zctype'] == COMPRESSION[c['name']]

            pinfo = phdu.get_info()
            assert pinfo['is_compressed_table'] == 0
            for c, pc in zip(info['colinfo'], pinfo['colinfo']):
                assert c['name'] == pc['name']
                assert c['zform'] == pc['tform']

            # the offsets are for the compressed HDU in the file
            offsets = hdu.get_offsets()
            assert offsets['header_start'] == fits[1].get_offsets()['data_end']
            assert offsets['data_start'] > offsets['header_start']
            assert offsets['data_end'] > offsets['data_start']
            assert fits[3].get_offsets()['header_start'] == offsets['data_end']


def test_compressed_table_read_not_supported():
    """
    Reading the data raises NotImplementedError with an explanation
    """
    rows = [5, 2, 0]
    columns = ['i4scalar', 'Sscalar']

    with _test_files() as (fzname, plainname):
        with FITS(fzname) as fits:
            hdu = fits[EXTNAME]

            calls = [
                lambda: hdu.read(),
                lambda: hdu.read(columns=columns, rows=rows),
                lambda: hdu.read_column('i4scalar'),
                lambda: hdu.read_columns(columns, rows=rows),
                lambda: hdu.read_rows(rows),
                lambda: hdu.read_slice(0, 10),
                lambda: hdu[:],
                lambda: hdu[rows],
                lambda: hdu['i4scalar'][:],
                lambda: hdu[columns][2:5],
                lambda: iter(hdu),
                lambda: hdu.where('i4scalar > 10'),
            ]
            for call in calls:
                with pytest.raises(NotImplementedError) as err:
                    call()
                mess = str(err.value)
                assert 'tile-compressed table' in mess
                assert 'funpack' in mess and 'rustfits' in mess

        # the convenience function goes through the same path
        with pytest.raises(NotImplementedError):
            read(fzname, ext=EXTNAME)
        with pytest.raises(NotImplementedError):
            read(fzname, ext=2, rows=rows, columns=columns)


def test_compressed_table_header():
    """
    The header can be read and holds the ZTABLE keywords
    """
    with _test_files() as (fzname, plainname):
        with FITS(fzname) as fits, FITS(plainname) as pfits:
            hdr = fits[EXTNAME].read_header()
            phdr = pfits[EXTNAME].read_header()

        assert hdr['ZTABLE'] is True
        assert hdr['ZNAXIS2'] == NROWS
        assert hdr['ZTILELEN'] == ZTILELEN
        assert hdr['ZPCOUNT'] == phdr['PCOUNT']
        assert hdr['ZNAXIS1'] == phdr['NAXIS1']
        assert hdr['TFIELDS'] == phdr['TFIELDS']
        assert hdr['MYKEY'] == phdr['MYKEY']
        assert hdr['EXTNAME'] == EXTNAME

        for i in range(1, 14):
            assert hdr['TTYPE%d' % i] == phdr['TTYPE%d' % i]
            assert hdr['ZFORM%d' % i] == phdr['TFORM%d' % i]
            assert hdr['ZCTYP%d' % i] == COMPRESSION[hdr['TTYPE%d' % i]]

        # the same through the convenience function
        mhdr = read_header(fzname, ext=EXTNAME)
        assert mhdr['ZTABLE'] is True
        assert mhdr['ZNAXIS2'] == NROWS

        # other extensions are unaffected
        mhdr = read_header(fzname, ext=1)
        assert 'ZTABLE' not in mhdr
        assert mhdr['NAXIS2'] == 3


def test_compressed_table_write_not_supported():
    """
    Writing data to a compressed table raises an error, in either mode
    """
    with _test_files() as (fzname, plainname):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.fits.fz')
            shutil.copy(fzname, fname)

            data = np.zeros(3, dtype=[('i4scalar', 'i4')])
            for mode in ['r', 'rw']:
                with FITS(fname, mode) as fits:
                    hdu = fits[EXTNAME]
                    calls = [
                        lambda: hdu.write(data),
                        lambda: hdu.write_column('i4scalar', data['i4scalar']),
                        lambda: hdu.write_var_column('f4var', [np.zeros(2)]),
                        lambda: hdu.append(data),
                        lambda: hdu.insert_column('newcol', data['i4scalar']),
                        lambda: hdu.delete_rows([0, 1]),
                        lambda: hdu.resize(10),
                    ]
                    for call in calls:
                        with pytest.raises(NotImplementedError):
                            call()

                    # the table is unchanged
                    assert hdu.get_nrows() == NROWS

            with FITS(fname) as fits:
                assert fits[EXTNAME].read_header()['ZNAXIS2'] == NROWS


def test_compressed_table_write_keys():
    """
    Header keywords can be written to a compressed table
    """
    with _test_files() as (fzname, plainname):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.fits.fz')
            shutil.copy(fzname, fname)

            with FITS(fname, 'rw') as fits:
                hdu = fits[EXTNAME]

                hdu.write_key('NEWKEY', 3.5, comment='a new keyword')
                hdu.write_comment('a comment')
                hdu.write_history('a history')

                hdr = hdu.read_header()
                assert hdr['NEWKEY'] == 3.5
                assert hdr['MYKEY'] == 'some value'
                assert hdr['ZTABLE'] is True

                hdu.write_checksum()
                hdu.verify_checksum()

            with FITS(fname) as fits:
                hdu = fits[EXTNAME]
                assert isinstance(hdu, CompressedTableHDU)
                hdr = hdu.read_header()
                assert hdr['NEWKEY'] == 3.5
                assert 'CHECKSUM' in hdr
                assert hdu.get_nrows() == NROWS
                hdu.verify_checksum()


@pytest.mark.skipif(not _have_program('fpack'), reason='fpack not found')
def test_compressed_table_fpack():
    """
    A table compressed by fpack, the reference compressor from cfitsio, is
    detected

    Complex columns are left out:  fpack compresses them with GZIP_2, which
    cfitsio cannot then uncompress, so funpack fails on them as well.
    """
    from numpy.lib import recfunctions as rfn

    adata = make_data()
    data = adata['data']
    names = [n for n in data.dtype.names if data[n].dtype.kind != 'c']
    data = rfn.repack_fields(data[names])

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')
        fzname = fname + '.fz'

        with FITS(fname, 'rw', clobber=True) as fits:
            fits.write_table(data, header=adata['keys'], extname='mytable')

        subprocess.run(
            ['fpack', '-table', fname], check=True, capture_output=True
        )
        assert os.path.exists(fzname)

        with FITS(fzname) as fits:
            hdu = fits['mytable']
            assert isinstance(hdu, CompressedTableHDU)
            assert hdu.is_compressed()
            assert hdu.get_nrows() == data.size
            assert hdu.get_colnames() == list(data.dtype.names)

            comp = hdu.get_compression()
            assert set(comp.keys()) == set(data.dtype.names)
            assert set(comp.values()) <= set(['RICE_1', 'GZIP_1', 'GZIP_2'])

            with pytest.raises(NotImplementedError):
                hdu.read()
            with pytest.raises(NotImplementedError):
                hdu.read_column(data.dtype.names[0])

            hdr = hdu.read_header()
            assert hdr['ZTABLE'] is True
            for key in adata['keys']:
                assert key in hdr
