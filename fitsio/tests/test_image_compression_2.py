import os
import tempfile
import numpy as np
import fitsio


def test_compression_case0():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write(img)
        fits.close()
        fits = fitsio.FITS(fn)
        assert len(fits) == 1


def test_compression_case1():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write(img, compress='RICE', tile_dims=(10, 5), qlevel=7.,
                   qmethod='SUBTRACTIVE_DITHER_2',
                   dither_seed=42)
        fits.close()
        fits = fitsio.FITS(fn)
        assert len(fits) == 2
        fits.close()
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [('ZTILE1', 5),
                         ('ZTILE2', 10),
                         ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_2'),
                         ('ZDITHER0', 42),
                         ('ZCMPTYPE', 'RICE_ONE'),]:
            assert hdr[key] == val


def test_compression_case2():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn + '[compress G]', 'rw', clobber=True)
        fits.write(img)
        fits.close()
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [('ZTILE1', 20),
                         ('ZTILE2', 1),
                         ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_1'),
                         ('ZCMPTYPE', 'GZIP_1'),]:
            assert hdr[key] == val


def test_compression_case3():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn + '[compress G 5 10; qz 8.0]',
                           'rw', clobber=True)
        fits.write(img, dither_seed=42)
        fits.close()
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [('ZTILE1', 5),
                         ('ZTILE2', 10),
                         ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_2'),
                         ('ZCMPTYPE', 'GZIP_1'),
                         ('ZDITHER0', 42)]:
            assert hdr[key] == val


def test_compression_case4():

    with tempfile.TemporaryDirectory() as tmpdir:
        fnpat = os.path.join(tmpdir, 'test-%i.fits')

        H, W = 200, 200
        bigimg = np.random.uniform(size=(H, W))
        results = []
        # None: don't even use compression at all
        # 0: lossless gzip
        for i, qlevel in enumerate([None, 0, 16, 4, 1]):
            fn = fnpat % i
            ql = qlevel
            kw = {}
            if ql is None:
                kw.update(compress=0)
                ql = 0
            print()
            print('Case 4, qlevel', qlevel)
            fits = fitsio.FITS(fn + '[compress G 100 100; qz %f]' % ql,
                               'rw', clobber=True)
            fits.write(bigimg, dither_seed=42, **kw)
            fits.close()
            filesize = os.stat(fn).st_size
            img2 = fitsio.read(fn)
            rms = np.sqrt(np.mean((img2 - bigimg)**2))
            results.append((qlevel, filesize, rms))
        for qlevel, filesize, rms in results:
            qs = '%4i' % qlevel if qlevel is not None else 'None'
            print('qlevel %s -> file size %7i, rms %f' % (qs, filesize, rms))
        # No compression
        q, sz, rms = results[0]
        assert sz == 2880 * (1 + int(np.ceil(H * W * 8 / 2880.)))
        assert rms == 0.0
        # GZIP lossless
        q, sz, rms = results[1]
        assert rms == 0.0
        # Decreasing file size
        for r1, r2 in zip(results, results[1:]):
            q1, sz1, rms1 = r1
            q2, sz2, rms2 = r2
            assert sz1 > sz2
            assert rms1 <= rms2


def test_compression_case5():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn + '[compress HS 10 10; s 2.0]', 'rw',
                           clobber=True)
        fits.write(img, dither_seed=42)
        fits.close()
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [('ZTILE1', 10),
                         ('ZTILE2', 10),
                         ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_1'),
                         ('ZCMPTYPE', 'HCOMPRESS_1'),
                         ('ZDITHER0', 42),
                         ('ZNAME1', 'SCALE'),
                         ('ZVAL1', 2.0),
                         ('ZNAME2', 'SMOOTH'),
                         ('ZVAL2', 1),
                         ]:
            assert hdr[key] == val


def test_compression_case6():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn + '[compress HS 10 10; s 2.0]', 'rw',
                           clobber=True)
        fits.write(img, dither_seed=42, hcomp_scale=1.0, hcomp_smooth=False)
        fits.close()
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [('ZTILE1', 10),
                         ('ZTILE2', 10),
                         ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_1'),
                         ('ZCMPTYPE', 'HCOMPRESS_1'),
                         ('ZDITHER0', 42),
                         ('ZNAME1', 'SCALE'),
                         ('ZVAL1', 1.),
                         ('ZNAME2', 'SMOOTH'),
                         ('ZVAL2', 0),
                         ]:
            assert hdr[key] == val


def test_compression_case7():
    # Check that if not specified, qlevel defaults to 4.
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        H, W = 200, 200
        bigimg = np.random.uniform(size=(H, W))
        # Default qlevel
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write(bigimg, compress='GZIP')
        fits.close()
        size_def = os.stat(fn).st_size
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [('ZQUANTIZ', 'SUBTRACTIVE_DITHER_1'),
                         ('ZCMPTYPE', 'GZIP_1'),
                         ]:
            assert hdr[key] == val
        # qlevel=0
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write(bigimg, compress='GZIP', qlevel=0)
        fits.close()
        size_0 = os.stat(fn).st_size
        # qlevel=4
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write(bigimg, compress='GZIP', qlevel=4)
        fits.close()
        size_4 = os.stat(fn).st_size
        # qlevel=16
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write(bigimg, compress='GZIP', qlevel=16)
        fits.close()
        size_16 = os.stat(fn).st_size
        # zero means NO COMPRESSION
        assert size_0 > size_4
        # heh, lower values mean MORE COMPRESSION
        assert size_4 < size_16
        assert size_def == size_4


def test_compression_case8():
    # Check multi-HDU case with a normal file
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        fits = fitsio.FITS(fn, 'rw', clobber=True)
        # A
        fits.write(img, extname='A')
        # B
        fits.write(img, extname='B', compress='GZIP')
        # C
        fits.write(img, extname='C', compress='GZIP',
                   qmethod='SUBTRACTIVE_DITHER_2')
        # D
        fits.write(img, extname='D')
        # E
        fits.write(img, extname='E', compress='GZIP')
        # F
        fits.write(img, extname='F', compress=None)
        fits.close()
        F = fitsio.FITS(fn)
        assert len(F) == 6
        hdrA = F['A'].read_header()
        hdrB = F['B'].read_header()
        hdrC = F['C'].read_header()
        hdrD = F['D'].read_header()
        hdrE = F['E'].read_header()
        hdrF = F['F'].read_header()
        # A is uncompressed
        assert 'ZCMPTYPE' not in hdrA
        # B is gzip
        assert hdrB['ZCMPTYPE'] == 'GZIP_1'
        assert hdrB['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_1'
        # C is gzip with SD2
        assert hdrC['ZCMPTYPE'] == 'GZIP_1'
        assert hdrC['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_2'
        # D is not compressed
        assert 'ZCMPTYPE' not in hdrD
        # E is GZIP again
        assert hdrE['ZCMPTYPE'] == 'GZIP_1'
        assert hdrE['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_1'
        # F is not compressed
        assert 'ZCMPTYPE' not in hdrF


def common_case9(filename, temp_filename, fitsclass, in_memory):
    img = np.ones((20, 20))
    fits = fitsclass(filename + '[compress G; qz 8]', 'rw', clobber=True)
    # A
    fits.write(img, extname='A')
    # B
    fits.write(img, extname='B', compress='RICE')
    # C
    fits.write(img, extname='C', compress='GZIP',
               qmethod='SUBTRACTIVE_DITHER_1')
    # D
    fits.write(img, extname='D')
    # E
    # FIXME -- we should test compress=None as well!
    # fits.write(img, compress=None, extname='E')
    fits.write(img, extname='E', compress=0)
    # F
    fits.write(img, extname='F')

    if in_memory:
        data = fits.read_raw()
        f = open(temp_filename, 'wb')
        f.write(data)
        f.close()
        filename = temp_filename
    fits.close()

    F = fitsio.FITS(filename, 'r')
    assert len(F) == 7
    prim = fitsio.read(filename, ext=0)
    assert prim is None
    hdrA = F['A'].read_header()
    hdrB = F['B'].read_header()
    hdrC = F['C'].read_header()
    hdrD = F['D'].read_header()
    hdrE = F['E'].read_header()
    hdrF = F['F'].read_header()
    # A is EFNS gzip
    assert hdrA['ZCMPTYPE'] == 'GZIP_1'
    assert hdrA['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_2'
    # B is rice
    assert hdrB['ZCMPTYPE'] == 'RICE_ONE'
    # B *also* has SD2! ... THIS is perhaps unexpected and not what we want!
    assert hdrB['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_2'
    # C is gzip with SD1
    assert hdrC['ZCMPTYPE'] == 'GZIP_1'
    assert hdrC['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_1'
    # D should default to GZIP, SD2?
    assert hdrD['ZCMPTYPE'] == 'GZIP_1'
    # E is uncompressed
    assert 'ZCMPTYPE' not in hdrE

    # THESE FAIL in case C
    assert hdrD['ZQUANTIZ'] == 'SUBTRACTIVE_DITHER_2'
    # F defaults back to GZIP
    assert hdrF['ZCMPTYPE'] == 'GZIP_1'


def test_compression_case9_A():
    # Check multi-HDU case with the Extended Filename Syntax
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')
        common_case9(fn, None, fitsio.FITS, False)


def test_compression_case9_B():
    # Check multi-HDU case with the Extended Filename Syntax
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')
        common_case9('mem://', fn, fitsio.FITS, True)


class FITS_mem(fitsio.FITS):
    def reopen(self):
        self.update_hdu_list()


def test_compression_case9_C():
    # Check multi-HDU case with the Extended Filename Syntax
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')
        common_case9('mem://', fn, FITS_mem, True)
