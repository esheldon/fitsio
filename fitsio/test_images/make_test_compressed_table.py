"""
Make the test files for reading tile-compressed tables

    test_compressed_table.fits.fz  holds a tile-compressed table
    test_compressed_table.fits     holds the same table, uncompressed

The files are written with rustfits, which can write tile-compressed tables.
The table is the third HDU, following an image and an uncompressed table, and
is followed by another image.  The tile length is smaller than the row count,
so that there are several tiles with a partial last tile.
"""

import os
import numpy as np
import rustfits

seed = 8675309
nrows = 300
ztilelen = 100

outdir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(outdir, 'test_compressed_table.fits')
fzname = fname + '.fz'

rng = np.random.RandomState(seed)

dtype = [
    ('u1scalar', 'u1'),
    ('i2scalar', 'i2'),
    ('u2scalar', 'u2'),
    ('i4scalar', 'i4'),
    ('u4scalar', 'u4'),
    ('i8scalar', 'i8'),
    ('f4scalar', 'f4'),
    ('f8scalar', 'f8'),
    ('Sscalar', 'S8'),
    ('bscalar', '?'),
    ('i4vec', 'i4', 3),
    ('f4arr', 'f4', (2, 3)),
    ('f4var', 'O'),
]

# mostly small ranges, so that the data compress; the extremes of the
# integer ranges are included to check the sign and unsigned handling
data = np.zeros(nrows, dtype=dtype)
data['u1scalar'] = rng.randint(0, 20, size=nrows)
data['i2scalar'] = rng.randint(-20, 20, size=nrows)
data['u2scalar'] = rng.randint(0, 20, size=nrows)
data['i4scalar'] = rng.randint(-20, 20, size=nrows)
data['u4scalar'] = rng.randint(0, 20, size=nrows)
data['i8scalar'] = rng.randint(-20, 20, size=nrows)
data['f4scalar'] = rng.randint(-20, 20, size=nrows) / 4.0
data['f8scalar'] = rng.randint(-20, 20, size=nrows) / 4.0
data['Sscalar'] = ['row%05d' % i for i in range(nrows)]
data['bscalar'] = rng.uniform(size=nrows) > 0.5
data['i4vec'] = rng.randint(-20, 20, size=(nrows, 3))
data['f4arr'] = rng.randint(-20, 20, size=(nrows, 2, 3)) / 4.0
for i in range(nrows):
    data['f4var'][i] = (rng.randint(-20, 20, size=i % 5) / 4.0).astype('f4')

data['u1scalar'][:2] = [0, 255]
data['i2scalar'][:2] = [-32768, 32767]
data['u2scalar'][:2] = [0, 65535]
data['i4scalar'][:2] = [-(2**31), 2**31 - 1]
data['u4scalar'][:2] = [0, 2**32 - 1]
data['i8scalar'][:2] = [-(2**63), 2**63 - 1]

# exercise all the algorithms; unspecified columns get the defaults
compress = {
    'u1scalar': 'RICE_1',
    'i2scalar': 'RICE_1',
    'i4vec': 'GZIP_2',
    'f4arr': 'GZIP_1',
}

image1 = np.arange(20, dtype='f4').reshape(4, 5)
image2 = np.arange(12, dtype='i4').reshape(3, 4)
plain = np.zeros(3, dtype=[('x', 'f8'), ('y', 'i4')])
plain['x'] = [1.5, 2.5, 3.5]
plain['y'] = [1, 2, 3]

for name, compressed in [(fzname, True), (fname, False)]:
    if os.path.exists(name):
        os.remove(name)

    with rustfits.FITS(name, 'w+') as fits:
        fits.write(image1, extname='image1')
        fits.write(plain, extname='plain')

        kw = {}
        if compressed:
            kw['compress'] = compress
            kw['ztilelen'] = ztilelen

        fits.create_table_hdu(
            np.dtype(dtype),
            nrows=nrows,
            var_dtypes={'f4var': 'f4'},
            extname='ctable',
            units={'f8scalar': 'deg'},
            **kw,
        )
        fits[2].write(data)
        fits[2].header['MYKEY'] = ('some value', 'a keyword to read back')

        fits.write(image2, extname='image2')

    print('wrote', name)
