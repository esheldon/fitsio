import os
import tempfile
import numpy as np
import fitsio


def test_compression_nocompress():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            fits.write(img)
        with fitsio.FITS(fn) as fits:
            assert len(fits) == 1


def test_compression_diskfile_kwargs():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            fits.write(
                img,
                compress='RICE',
                tile_dims=(10, 5),
                qlevel=7.0,
                qmethod='SUBTRACTIVE_DITHER_2',
                dither_seed=42,
            )
        with fitsio.FITS(fn) as fits:
            assert len(fits) == 2
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [
            ('ZTILE1', 5),
            ('ZTILE2', 10),
            ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_2'),
            ('ZDITHER0', 42),
            ('ZCMPTYPE', 'RICE_ONE'),
        ]:
            assert hdr[key] == val


def test_compression_efns():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS(fn + '[compress G]', 'rw', clobber=True) as fits:
            fits.write(img)
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [
            ('ZTILE1', 20),
            ('ZTILE2', 1),
            ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_1'),
            ('ZCMPTYPE', 'GZIP_1'),
        ]:
            assert hdr[key] == val


def test_compression_efns_kwargs():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS(
            fn + '[compress G 5 10; qz 8.0]', 'rw', clobber=True
        ) as fits:
            fits.write(img, dither_seed=42)
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [
            ('ZTILE1', 5),
            ('ZTILE2', 10),
            ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_2'),
            ('ZCMPTYPE', 'GZIP_1'),
            ('ZDITHER0', 42),
        ]:
            assert hdr[key] == val


def test_compression_qlevels_none_zero():
    default_kws = {
        "compress": fitsio.GZIP_2,
        "tile_dims": np.array([100, 100]),
        "qmethod": fitsio.SUBTRACTIVE_DITHER_2,
    }
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
            kw.update(default_kws)
            if ql is None:
                kw.update(compress=0)
                ql = 0
            kw["qlevel"] = ql
            with fitsio.FITS(fn, 'rw', clobber=True) as fits:
                fits.write(bigimg, dither_seed=42, **kw)
            filesize = os.stat(fn).st_size
            img2 = fitsio.read(fn)
            rms = np.sqrt(np.mean((img2 - bigimg) ** 2))
            results.append((qlevel, filesize, rms))
        # No compression
        q, sz, rms = results[0]
        assert sz == 2880 * (1 + int(np.ceil(H * W * 8 / 2880.0)))
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


def test_compression_hcomp_args():
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS(
            fn + '[compress HS 10 10; s 2.0]', 'rw', clobber=True
        ) as fits:
            fits.write(img, dither_seed=42)
        hdr = fitsio.read_header(fn, ext=1)
        for key, val in [
            ('ZTILE1', 10),
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


def test_compression_qlevel_default():
    # Check that if not specified, qlevel defaults to 4.
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        H, W = 200, 200
        bigimg = np.random.uniform(size=(H, W))
        # Default qlevel
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            fits.write(bigimg, compress='GZIP')
        size_def = os.stat(fn).st_size
        hdr = fitsio.read_header(fn, ext=1)
        print(hdr)
        for key, val in [
            ('ZQUANTIZ', 'SUBTRACTIVE_DITHER_1'),
            ('ZCMPTYPE', 'GZIP_1'),
        ]:
            assert hdr[key] == val
        # qlevel=0
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            fits.write(bigimg, compress='GZIP', qlevel=0)
        size_0 = os.stat(fn).st_size
        # qlevel=4
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            fits.write(bigimg, compress='GZIP', qlevel=4)
        size_4 = os.stat(fn).st_size
        # qlevel=16
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            fits.write(bigimg, compress='GZIP', qlevel=16)
        size_16 = os.stat(fn).st_size
        # zero means NO COMPRESSION
        assert size_0 > size_4
        # heh, lower values mean MORE COMPRESSION
        assert size_4 < size_16
        assert size_def == size_4


def test_compression_multihdu_diskfile():
    # Check multi-HDU case with a normal file
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS(fn, 'rw', clobber=True) as fits:
            # A
            fits.write(img, extname='A')
            # B
            fits.write(img, extname='B', compress='GZIP')
            # C
            fits.write(
                img,
                extname='C',
                compress='GZIP',
                qmethod='SUBTRACTIVE_DITHER_2',
            )
            # D
            fits.write(img, extname='D')
            # E
            fits.write(img, extname='E', compress='GZIP')
            # F
            fits.write(img, extname='F', compress=None)
        with fitsio.FITS(fn) as fits:
            assert len(fits) == 6
            hdrA = fits['A'].read_header()
            hdrB = fits['B'].read_header()
            hdrC = fits['C'].read_header()
            hdrD = fits['D'].read_header()
            hdrE = fits['E'].read_header()
            hdrF = fits['F'].read_header()
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


def test_compression_multihdu_memfile():
    # Check multi-HDU case with a normal file
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, 'test.fits')

        img = np.ones((20, 20))
        with fitsio.FITS("mem://", 'rw', clobber=True) as fits:
            # A
            fits.write(img, extname='A')
            # B
            fits.write(img, extname='B', compress='GZIP')
            # C
            fits.write(
                img,
                extname='C',
                compress='GZIP',
                qmethod='SUBTRACTIVE_DITHER_2',
            )
            # D
            fits.write(img, extname='D')
            # E
            fits.write(img, extname='E', compress='GZIP')
            # F
            fits.write(img, extname='F', compress=None)

            data = fits.read_raw()
            with open(fn, 'wb') as f:
                f.write(data)

        with fitsio.FITS(fn) as fits:
            assert len(fits) == 6
            hdrA = fits['A'].read_header()
            hdrB = fits['B'].read_header()
            hdrC = fits['C'].read_header()
            hdrD = fits['D'].read_header()
            hdrE = fits['E'].read_header()
            hdrF = fits['F'].read_header()
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
