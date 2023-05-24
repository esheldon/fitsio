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
        # dtypes = ['u2']
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

                if dtype[0] == 'f':
                    compare_array_abstol(
                        data,
                        rdata,
                        0.2,
                        "%s compressed images ('%s')" % (compress, dtype),
                    )
                else:
                    # for integers we have chosen a wide range of values, so
                    # there will be no quantization and we expect no
                    # information loss
                    compare_array(
                        data, rdata,
                        "%s compressed images ('%s')" % (compress, dtype)
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
