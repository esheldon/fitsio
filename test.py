import fitsio
from astropy.table import Table
import numpy as np

print('Reading X column test')
filename = 'test_x.fits'
data, header = fitsio.read(filename, header=True)
print(data.dtype)

print('Writing test')
filename = 'test_write.fits'
t = Table([[1], [2], [3]], dtype=[np.bool, np.uint8, np.int8]).as_array()  # 'b1', 'u1', 'i1'
fitsio.write(filename, t)

data, header = fitsio.read(filename, header=True)
print(header)
print(data.dtype)
