import pytest
import sys
import os
import tempfile
from .checks import (
    # check_header,
    compare_array,
    compare_array_abstol,
)
import numpy as np
from ..fitslib import (
    FITS,
    read,
    write,
)


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
    ]
)
def test_compressed_write_read(compress):
    """
    Test writing and reading a rice compressed image
    """
    nrows = 5
    ncols = 20
    if compress in ['rice', 'hcompress'] or 'gzip' in compress:
        dtypes = ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'f4', 'f8']
    elif compress == 'plio':
        dtypes = ['i1', 'i2', 'i4', 'f4', 'f8']
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

        for ext, dtype in enumerate(dtypes):
            if dtype[0] == 'f':
                data = rng.normal(size=(nrows, ncols))
                if compress == 'plio':
                    data = data.clip(min=0)
                data = data.astype(dtype)
            else:
                data = np.arange(
                    nrows * ncols, dtype=dtype,
                ).reshape(nrows, ncols)

            csend = compress.replace('_lossless', '')
            write(fname, data, compress=csend, qlevel=qlevel)
            rdata = read(fname, ext=ext+1)

            if 'lossless' in compress or dtype[0] in ['i', 'u']:
                compare_array(
                    data, rdata,
                    "%s compressed images ('%s')" % (compress, dtype)
                )
            else:
                # lossy floating point
                compare_array_abstol(
                    data,
                    rdata,
                    0.2,
                    "%s compressed images ('%s')" % (compress, dtype),
                )

        with FITS(fname) as fits:
            for ii in range(len(dtypes)):
                i = ii + 1
                assert fits[i].is_compressed(), "is compressed"


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
    ]
)
def test_compressed_write_read_fitsobj(compress):
    """
    Test writing and reading a rice compressed image

    In this version, keep the fits object open
    """
    nrows = 5
    ncols = 20
    if compress in ['rice', 'hcompress'] or 'gzip' in compress:
        dtypes = ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'f4', 'f8']
        # dtypes = ['u2']
    elif compress == 'plio':
        dtypes = ['i1', 'i2', 'i4', 'f4', 'f8']
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

            for dtype in dtypes:
                if dtype[0] == 'f':
                    data = rng.normal(size=(nrows, ncols))
                    if compress == 'plio':
                        data = data.clip(min=0)
                    data = data.astype(dtype)
                else:
                    data = np.arange(
                        nrows * ncols, dtype=dtype,
                    ).reshape(nrows, ncols)

                csend = compress.replace('_lossless', '')
                fits.write_image(data, compress=csend, qlevel=qlevel)
                rdata = fits[-1].read()

                if 'lossless' in compress or dtype[0] in ['i', 'u']:
                    # for integers we have chosen a wide range of values, so
                    # there will be no quantization and we expect no
                    # information loss
                    compare_array(
                        data, rdata,
                        "%s compressed images ('%s')" % (compress, dtype)
                    )
                else:
                    # lossy floating point
                    compare_array_abstol(
                        data,
                        rdata,
                        0.2,
                        "%s compressed images ('%s')" % (compress, dtype),
                    )

        with FITS(fname) as fits:
            for ii in range(len(dtypes)):
                i = ii + 1
                assert fits[i].is_compressed(), "is compressed"


@pytest.mark.skipif(sys.version_info < (3, 9),
                    reason='importlib bug in 3.8')
def test_gzip_tile_compressed_read_lossless_astropy():
    """
    Test reading an image gzip compressed by astropy (fixed by cfitsio 3.49)
    """
    import importlib.resources
    ref = importlib.resources.files("fitsio") / 'test_images' / 'test_gzip_compressed_image.fits.fz'  # noqa
    with importlib.resources.as_file(ref) as gzip_file:
        data = read(gzip_file)

    compare_array(data, data*0.0, "astropy lossless compressed image")


def test_compress_preserve_zeros():
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

                    data = rng.normal(size=5*20).reshape(5, 20).astype(dtype)
                    for zind in zinds:
                        data[zind[0], zind[1]] = 0.0

                    fits.write_image(
                        data,
                        compress=compress,
                        qlevel=16,
                        qmethod='SUBTRACTIVE_DITHER_2',
                    )
                    rdata = fits[-1].read()

                    for zind in zinds:
                        assert rdata[zind[0], zind[1]] == 0.0


@pytest.mark.parametrize(
    'compress',
    [
        'rice',
        'hcompress',
        'plio',
    ]
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
def test_compressed_seed(compress, seed_type, use_fits_object, dtype):
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

        if use_fits_object:
            with FITS(fname1, 'rw') as fits1:
                fits1.write(
                    data, compress=compress, qlevel=qlevel,
                    # dither_seed=dither_seed,
                    dither_seed=dither_seed1,
                )
                rdata1 = fits1[-1].read()

            with FITS(fname2, 'rw') as fits2:
                fits2.write(
                    data, compress=compress, qlevel=qlevel,
                    # dither_seed=dither_seed,
                    dither_seed=dither_seed2,
                )
                rdata2 = fits2[-1].read()
        else:
            write(
                fname1, data, compress=compress, qlevel=qlevel,
                # dither_seed=dither_seed,
                dither_seed=dither_seed1,
            )
            rdata1 = read(fname1)

            write(
                fname2, data, compress=compress, qlevel=qlevel,
                # dither_seed=dither_seed,
                dither_seed=dither_seed2,
            )
            rdata2 = read(fname2)

        mess = "%s compressed images ('%s')" % (compress, dtype)

        if seed_type in ['checksum', 'checksum_int', 'matched']:
            assert np.all(rdata1 == rdata2), mess
        else:
            assert np.all(rdata1 != rdata2), mess


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
                fname, data, compress=compress, qlevel=qlevel,
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

        fitsio.write(fname1, data.copy(), dither_seed='checksum',
                     compress='RICE', qlevel=1e-4, tile_dims=(100, 100),
                     clobber=True)
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


if __name__ == '__main__':
    test_compressed_seed(
        compress='rice',
        match_seed=False,
        use_fits_object=True,
        dtype='f4',
    )
