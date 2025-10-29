import pytest
import sys
import os
import tempfile
from .checks import (
    # check_header,
    compare_array,
)
import numpy as np
from ..fitslib import (
    FITS,
    read,
    write,
    RICE_1,
    SUBTRACTIVE_DITHER_1,
    GZIP_1,
    GZIP_2,
    PLIO_1,
    HCOMPRESS_1,
)
from ..util import cfitsio_is_bundled, cfitsio_version

CFITSIO_VERSION = cfitsio_version(asfloat=True)


@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize(
    'compress',
    [
        'rice',
        'hcompress',
        'plio',
        'gzip',
        'gzip_2',
        'gzip_lossless',
        'gzip_2_lossless',
    ],
)
@pytest.mark.parametrize(
    'dtype', ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'f4', 'f8']
)
def test_compressed_write_read(compress, dtype, with_nan):
    """
    Test writing and reading a rice compressed image
    """
    nrows = 5
    ncols = 20
    if compress in ['rice', 'hcompress'] or 'gzip' in compress:
        pass
    elif compress == 'plio':
        if dtype not in ['i1', 'i2', 'i4', 'f4', 'f8']:
            return
    else:
        raise ValueError('unexpected compress %s' % compress)

    if 'lossless' in compress:
        qlevel = None
    else:
        qlevel = 16

    seed = 1919
    rng = np.random.RandomState(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        if dtype[0] == 'f':
            data = rng.normal(size=(nrows, ncols))
            if compress == 'plio':
                data = data.clip(min=0)
            data = data.astype(dtype)
        else:
            data = np.arange(
                nrows * ncols,
                dtype=dtype,
            ).reshape(nrows, ncols)

        if "f" in dtype and with_nan and compress != "plio":
            data[3, 11] = np.nan

        csend = compress.replace('_lossless', '')
        write(fname, data, compress=csend, qlevel=qlevel)
        rdata = read(fname, ext=1)

        if 'lossless' in compress or dtype[0] in ['i', 'u']:
            np.testing.assert_array_equal(
                data,
                rdata,
                err_msg="%s compressed images ('%s')" % (compress, dtype),
            )
        else:
            # lossy floating point
            np.testing.assert_allclose(
                data,
                rdata,
                rtol=0,
                atol=0.2,
                err_msg="%s compressed images ('%s')" % (compress, dtype),
            )

        with FITS(fname) as fits:
            assert fits[1].is_compressed(), "is compressed"


@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize(
    'compress',
    [
        'rice',
        'hcompress',
        'plio',
        'gzip',
        'gzip_2',
        'gzip_lossless',
        'gzip_2_lossless',
    ],
)
@pytest.mark.parametrize(
    'dtype', ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'f4', 'f8']
)
def test_compressed_write_read_fitsobj(compress, dtype, with_nan):
    """
    Test writing and reading a rice compressed image

    In this version, keep the fits object open
    """

    if (
        "gzip" in compress
        and dtype in ["u2", "i2", "u4", "i4"]
        and not cfitsio_is_bundled()
    ):
        pytest.xfail(
            reason=(
                "Non-bundled cfitsio libraries have a bug. "
                "See https://github.com/HEASARC/cfitsio/pull/97."
            )
        )

    nrows = 5
    ncols = 20
    if compress in ['rice', 'hcompress'] or 'gzip' in compress:
        pass
    elif compress == 'plio':
        if dtype not in ['i1', 'i2', 'i4', 'f4', 'f8']:
            return
    else:
        raise ValueError('unexpected compress %s' % compress)

    if 'lossless' in compress:
        qlevel = None
        # qlevel = 9999
    else:
        qlevel = 16

    seed = 1919
    rng = np.random.RandomState(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        with FITS(fname, 'rw') as fits:
            # note i8 not supported for compressed!

            if dtype[0] == 'f':
                data = rng.normal(size=(nrows, ncols))
                if compress == 'plio':
                    data = data.clip(min=0)
                data = data.astype(dtype)
            else:
                data = np.arange(
                    nrows * ncols,
                    dtype=dtype,
                ).reshape(nrows, ncols)

            if "f" in dtype and with_nan and compress != "plio":
                data[3, 11] = np.nan

            csend = compress.replace('_lossless', '')
            fits.write_image(data, compress=csend, qlevel=qlevel)
            rdata = fits[-1].read()

            if 'lossless' in compress or dtype[0] in ['i', 'u']:
                # for integers we have chosen a wide range of values, so
                # there will be no quantization and we expect no
                # information loss
                np.testing.assert_array_equal(
                    data,
                    rdata,
                    "%s compressed images ('%s')" % (compress, dtype),
                )
            else:
                # lossy floating point
                np.testing.assert_allclose(
                    data,
                    rdata,
                    rtol=0,
                    atol=0.2,
                    err_msg="%s compressed images ('%s')" % (compress, dtype),
                )

        with FITS(fname) as fits:
            assert fits[1].is_compressed(), "is compressed"


@pytest.mark.skipif(sys.version_info < (3, 9), reason='importlib bug in 3.8')
@pytest.mark.skipif(CFITSIO_VERSION < 3.49, reason='bug in cfitsio < 3.49')
def test_gzip_tile_compressed_read_lossless_astropy():
    """
    Test reading an image gzip compressed by astropy (fixed by cfitsio 3.49)
    """
    import importlib.resources

    ref = (
        importlib.resources.files("fitsio")
        / 'test_images'
        / 'test_gzip_compressed_image.fits.fz'
    )  # noqa
    with importlib.resources.as_file(ref) as gzip_file:
        data = read(gzip_file)

    compare_array(data, data * 0.0, "astropy lossless compressed image")


@pytest.mark.parametrize("with_nan", [False, True])
def test_compress_preserve_zeros(with_nan):
    """
    Test writing and reading gzip compressed image
    """

    zinds = [
        (1, 3),
        (2, 9),
    ]

    dtypes = ['f4', 'f8']

    seed = 2020
    rng = np.random.RandomState(seed)

    # Do not test hcompress as it doesn't support SUBTRACTIVE_DITHER_2
    for compress in ['gzip', 'gzip_2', 'rice']:
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.fits')

            with FITS(fname, 'rw') as fits:
                for dtype in dtypes:
                    data = rng.normal(size=5 * 20).reshape(5, 20).astype(dtype)
                    for zind in zinds:
                        data[zind[0], zind[1]] = 0.0
                    if with_nan:
                        data[3, 15] = np.nan

                    fits.write_image(
                        data,
                        compress=compress,
                        qlevel=16,
                        qmethod='SUBTRACTIVE_DITHER_2',
                    )
                    rdata = fits[-1].read()

                    for zind in zinds:
                        assert rdata[zind[0], zind[1]] == 0.0
                    if with_nan:
                        assert np.isnan(rdata[3, 15])


@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize(
    'compress',
    [
        'rice',
        'hcompress',
        'plio',
    ],
)
@pytest.mark.parametrize(
    'seed_type',
    ['matched', 'unmatched', 'checksum', 'checksum_int'],
)
@pytest.mark.parametrize(
    'use_fits_object',
    [False, True],
)
@pytest.mark.parametrize(
    'dtype',
    ['f4', 'f8'],
)
def test_compressed_seed(
    compress, seed_type, use_fits_object, dtype, with_nan
):
    """
    Test writing and reading a rice compressed image
    """
    nrows = 5
    ncols = 20

    qlevel = 16

    seed = 1919
    rng = np.random.RandomState(seed)

    if seed_type == 'matched':
        # dither_seed = 9881
        dither_seed1 = 9881
        dither_seed2 = 9881
    elif seed_type == 'unmatched':
        # dither_seed = None
        dither_seed1 = 3
        dither_seed2 = 4
    elif seed_type == 'checksum':
        dither_seed1 = 'checksum'
        dither_seed2 = b'checksum'
    elif seed_type == 'checksum_int':
        dither_seed1 = -1
        # any negative means use checksum
        dither_seed2 = -3

    with tempfile.TemporaryDirectory() as tmpdir:
        fname1 = os.path.join(tmpdir, 'test1.fits')
        fname2 = os.path.join(tmpdir, 'test2.fits')

        data = rng.normal(size=(nrows, ncols))
        if compress == 'plio':
            data = data.clip(min=0)
        data = data.astype(dtype)

        if "f" in dtype and with_nan and compress != "plio":
            data[3, 11] = np.nan

        if use_fits_object:
            with FITS(fname1, 'rw') as fits1:
                fits1.write(
                    data,
                    compress=compress,
                    qlevel=qlevel,
                    # dither_seed=dither_seed,
                    dither_seed=dither_seed1,
                )
                rdata1 = fits1[-1].read()

            with FITS(fname2, 'rw') as fits2:
                fits2.write(
                    data,
                    compress=compress,
                    qlevel=qlevel,
                    # dither_seed=dither_seed,
                    dither_seed=dither_seed2,
                )
                rdata2 = fits2[-1].read()
        else:
            write(
                fname1,
                data,
                compress=compress,
                qlevel=qlevel,
                # dither_seed=dither_seed,
                dither_seed=dither_seed1,
            )
            rdata1 = read(fname1)

            write(
                fname2,
                data,
                compress=compress,
                qlevel=qlevel,
                # dither_seed=dither_seed,
                dither_seed=dither_seed2,
            )
            rdata2 = read(fname2)

        mess = "%s compressed images ('%s')" % (compress, dtype)

        if seed_type in ['checksum', 'checksum_int', 'matched']:
            np.testing.assert_array_equal(rdata1, rdata2, mess)
        else:
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(rdata1, rdata2, mess)

        if "f" in dtype and with_nan and compress != "plio":
            assert np.isnan(rdata1[3, 11])
            assert np.isnan(rdata2[3, 11])
        else:
            assert np.all(np.isfinite(rdata1))
            assert np.all(np.isfinite(rdata2))


@pytest.mark.parametrize(
    'dither_seed',
    ['blah', 10_001],
)
def test_compressed_seed_bad(dither_seed):
    """
    Test writing and reading a rice compressed image
    """
    compress = 'rice'
    dtype = 'f4'
    nrows = 5
    ncols = 20

    qlevel = 16

    seed = 1919
    rng = np.random.RandomState(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.fits')

        data = rng.normal(size=(nrows, ncols))
        data = data.astype(dtype)

        with pytest.raises(ValueError):
            write(
                fname,
                data,
                compress=compress,
                qlevel=qlevel,
                dither_seed=dither_seed,
            )


def test_memory_compressed_seed():
    import fitsio

    dtype = 'f4'
    nrows = 300
    ncols = 500

    seed = 1919
    rng = np.random.RandomState(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname1 = os.path.join(tmpdir, 'test1.fits')
        fname2 = os.path.join(tmpdir, 'test2.fits')

        data = rng.normal(size=(nrows, ncols))
        data = data.astype(dtype)

        fitsio.write(
            fname1,
            data.copy(),
            dither_seed='checksum',
            compress='RICE',
            qlevel=1e-4,
            tile_dims=(100, 100),
            clobber=True,
        )
        hdr = fitsio.read_header(fname1, ext=1)
        dither1 = hdr['ZDITHER0']
        assert dither1 == 8269

        fits = fitsio.FITS('mem://[compress R 100,100; qz -1e-4]', 'rw')
        fits.write(data.copy(), dither_seed='checksum')
        data = fits.read_raw()
        fits.close()
        f = open(fname2, 'wb')
        f.write(data)
        f.close()
        hdr = fitsio.read_header(fname2, ext=1)
        dither2 = hdr['ZDITHER0']
        assert dither1 == dither2


def test_image_compression_inmem_subdither2():
    H, W = 100, 100
    rng = np.random.RandomState(seed=10)
    img = rng.normal(size=(H, W))
    img[40:50, :] = 0.0
    with FITS('mem://[compress G 100,100; qz 0]', 'rw') as F:
        F.write(img)
        rawdata = F.read_raw()

    with tempfile.TemporaryDirectory() as tmpdir:
        pth = os.path.join(tmpdir, 'out.fits')
        with open(pth, 'wb') as f:
            f.write(rawdata)
        im2 = read(pth)
        z = im2[40:50, :]

    minval = z.min()
    assert minval == 0


@pytest.mark.parametrize(
    "kw,val",
    [
        ("compress", RICE_1),
        ("tile_dims", (10, 2)),
        ("tile_dims", np.array([10, 2])),
        ("tile_dims", [10, 2]),
        ("qlevel", 10.0),
        ("qmethod", SUBTRACTIVE_DITHER_1),
        ("hcomp_scale", 10.0),
        ("hcomp_smooth", True),
    ],
)
@pytest.mark.parametrize("set_val_to_none", [False, True])
def test_image_compression_raises_on_python_set(kw, val, set_val_to_none):
    H, W = 100, 100
    rng = np.random.RandomState(seed=10)
    img = rng.normal(size=(H, W))
    if set_val_to_none:
        kws = {kw: None}
    else:
        kws = {kw: val}

    with FITS('mem://[compress G 100,100; qz 0]', 'rw') as F:
        with pytest.raises(ValueError):
            F.write(img, **kws)

    with FITS('mem://[compress G 100,100; qz 4.0]', 'rw') as F:
        F.write(img, dither_seed=10)


@pytest.mark.parametrize(
    "compress",
    [
        RICE_1,
        GZIP_1,
        GZIP_2,
        PLIO_1,
        HCOMPRESS_1,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
    ],
)
@pytest.mark.parametrize("fname", ["mem://", "test.fits"])
def test_image_compression_inmem_lossess_int(compress, dtype, fname):
    if not cfitsio_is_bundled():
        pytest.xfail(
            reason=(
                "Non-bundled cfitsio libraries have a bug. "
                "See https://github.com/HEASARC/cfitsio/pull/97 "
                "and https://github.com/HEASARC/cfitsio/pull/99."
            ),
        )
    if compress == PLIO_1 and dtype in [np.int16, np.uint32, np.int32]:
        pytest.skip(
            reason="PLIO lossless compression of int16, uint32, and "
            "int32 types is not supported by cfitsio",
        )
    rng = np.random.RandomState(seed=10)
    img = rng.normal(size=(300, 300))
    if dtype in [
        np.uint8,
        np.uint16,
        np.uint32,
    ]:
        img = np.abs(img)
    img = img.astype(dtype)
    with tempfile.TemporaryDirectory() as tmpdir:
        if "mem://" not in fname:
            fpth = os.path.join(tmpdir, fname)
        else:
            fpth = fname

        with FITS(fpth, 'rw') as F:
            F.write(img, compress=compress, qlevel=0)
            rimg = F[-1].read()
            assert rimg is not None
            assert np.array_equal(rimg, img)


def test_image_compression_inmem_lossessgzip_int_zeros():
    img = np.zeros((300, 300)).astype(np.int32)
    with FITS('mem://', 'rw') as F:
        F.write(img, compress='GZIP', qlevel=0)
        rimg = F[-1].read()
        assert rimg is not None
        assert np.array_equal(rimg, img)


def test_image_compression_inmem_lossessgzip_float():
    rng = np.random.RandomState(seed=10)
    img = rng.normal(size=(300, 300))
    with FITS('mem://', 'rw') as F:
        F.write(img, compress='GZIP', qlevel=0)
        rimg = F[-1].read()
        assert rimg is not None
        assert np.array_equal(rimg, img)


def test_image_mem_reopen_noop():
    rng = np.random.RandomState(seed=10)
    img = rng.normal(size=(300, 300))
    with FITS('mem://', 'rw') as F:
        F.write(img)
        rimg = F[-1].read()
        assert rimg is not None
        assert np.array_equal(rimg, img)
        F.reopen()
        rimg = F[-1].read()
        assert rimg is not None
        assert np.array_equal(rimg, img)

    with FITS('mem://', 'rw') as F:
        F.write(img)
        F.reopen()
        rimg = F[-1].read()
        assert rimg is not None
        assert np.array_equal(rimg, img)


@pytest.mark.parametrize("nan_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "fname",
    [
        "test.fits",
        "mem://",
    ],
)
def test_image_compression_nulls(fname, dtype, nan_value):
    data = np.arange(36).reshape((6, 6)).astype(dtype)
    data[1, 1] = nan_value

    # everything comes back as nan
    if nan_value is not np.nan:
        msk = ~np.isfinite(data)
        cdata = data.copy()
        cdata[msk] = np.nan
    else:
        cdata = data

    with tempfile.TemporaryDirectory() as tmpdir:
        if "mem://" not in fname:
            fpth = os.path.join(tmpdir, fname)
        else:
            fpth = fname

        with FITS(fpth, "rw") as fits:
            fits.write(
                data,
                compress='RICE_1',
                tile_dims=(3, 3),
                dither_seed=10,
                qlevel=2,
            )
            read_data = fits[1].read()

            np.testing.assert_allclose(
                read_data,
                cdata,
            )

        if "mem://" not in fpth:
            with FITS(fpth, "r") as fits:
                read_data = fits[1].read()
                np.testing.assert_allclose(
                    read_data,
                    cdata,
                )


if __name__ == '__main__':
    test_compressed_seed(
        compress='rice',
        match_seed=False,
        use_fits_object=True,
        dtype='f4',
    )
