import sys
import numpy as np
from functools import lru_cache

from .._fitsio_wrap import cfitsio_use_standard_strings

lorem_ipsum = (
    'Lorem ipsum dolor sit amet, consectetur adipiscing '
    'elit, sed do eiusmod tempor incididunt ut labore '
    'et dolore magna aliqua'
)


@lru_cache(maxsize=1)
def make_data():

    nvec = 2
    ashape = (21, 21)
    Sdtype = 'S6'
    Udtype = 'U6'

    # all currently available types, scalar, 1-d and 2-d array columns
    dtype = [
        ('u1scalar', 'u1'),
        ('i1scalar', 'i1'),
        ('b1scalar', '?'),
        ('u2scalar', 'u2'),
        ('i2scalar', 'i2'),
        ('u4scalar', 'u4'),
        ('i4scalar', '<i4'),  # mix the byte orders a bit, test swapping
        ('i8scalar', 'i8'),
        ('f4scalar', 'f4'),
        ('f8scalar', '>f8'),
        ('c8scalar', 'c8'),  # complex, two 32-bit
        ('c16scalar', 'c16'),  # complex, two 32-bit

        ('u1vec', 'u1', nvec),
        ('i1vec', 'i1', nvec),
        ('b1vec', '?', nvec),
        ('u2vec', 'u2', nvec),
        ('i2vec', 'i2', nvec),
        ('u4vec', 'u4', nvec),
        ('i4vec', 'i4', nvec),
        ('i8vec', 'i8', nvec),
        ('f4vec', 'f4', nvec),
        ('f8vec', 'f8', nvec),
        ('c8vec', 'c8', nvec),
        ('c16vec', 'c16', nvec),

        ('u1arr', 'u1', ashape),
        ('i1arr', 'i1', ashape),
        ('b1arr', '?', ashape),
        ('u2arr', 'u2', ashape),
        ('i2arr', 'i2', ashape),
        ('u4arr', 'u4', ashape),
        ('i4arr', 'i4', ashape),
        ('i8arr', 'i8', ashape),
        ('f4arr', 'f4', ashape),
        ('f8arr', 'f8', ashape),
        ('c8arr', 'c8', ashape),
        ('c16arr', 'c16', ashape),

        # special case of (1,)
        ('f8arr_dim1', 'f8', (1, )),


        ('Sscalar', Sdtype),
        ('Svec',   Sdtype, nvec),
        ('Sarr',   Sdtype, ashape),
    ]

    if cfitsio_use_standard_strings():
        dtype += [
            ('Sscalar_nopad', Sdtype),
            ('Svec_nopad', Sdtype, nvec),
            ('Sarr_nopad', Sdtype, ashape),
        ]

    if sys.version_info > (3, 0, 0):
        dtype += [
           ('Uscalar', Udtype),
           ('Uvec', Udtype, nvec),
           ('Uarr', Udtype, ashape),
        ]

        if cfitsio_use_standard_strings():
            dtype += [
               ('Uscalar_nopad', Udtype),
               ('Uvec_nopad', Udtype, nvec),
               ('Uarr_nopad', Udtype, ashape),
            ]

    dtype2 = [
        ('index', 'i4'),
        ('x', 'f8'),
        ('y', 'f8'),
    ]

    nrows = 4
    data = np.zeros(nrows, dtype=dtype)

    dtypes = [
        'u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'i8', 'f4', 'f8', 'c8', 'c16',
    ]
    for t in dtypes:
        if t in ['c8', 'c16']:
            data[t+'scalar'] = [complex(i+1, (i+1)*2) for i in range(nrows)]
            vname = t + 'vec'
            for row in range(nrows):
                for i in range(nvec):
                    index = (row + 1) * (i + 1)
                    data[vname][row, i] = complex(index, index*2)
            aname = t+'arr'
            for row in range(nrows):
                for i in range(ashape[0]):
                    for j in range(ashape[1]):
                        index = (row + 1) * (i + 1) * (j + 1)
                        data[aname][row, i, j] = complex(index, index*2)

        else:
            data[t+'scalar'] = 1 + np.arange(nrows, dtype=t)
            data[t+'vec'] = 1 + np.arange(
                nrows*nvec, dtype=t,
            ).reshape(nrows, nvec)
            arr = 1 + np.arange(nrows*ashape[0]*ashape[1], dtype=t)
            data[t+'arr'] = arr.reshape(nrows, ashape[0], ashape[1])

    for t in ['b1']:
        data[t+'scalar'] = (np.arange(nrows) % 2 == 0).astype('?')
        data[t+'vec'] = (
            np.arange(nrows*nvec) % 2 == 0
        ).astype('?').reshape(nrows, nvec)

        arr = (np.arange(nrows*ashape[0]*ashape[1]) % 2 == 0).astype('?')
        data[t+'arr'] = arr.reshape(nrows, ashape[0], ashape[1])

    # strings get padded when written to the fits file.  And the way I do
    # the read, I read all bytes (ala mrdfits) so the spaces are preserved.
    #
    # so we need to pad out the strings with blanks so we can compare

    data['Sscalar'] = ['%-6s' % s for s in ['hello', 'world', 'good', 'bye']]
    data['Svec'][:, 0] = '%-6s' % 'hello'
    data['Svec'][:, 1] = '%-6s' % 'world'

    s = 1 + np.arange(nrows*ashape[0]*ashape[1])
    s = ['%-6s' % el for el in s]
    data['Sarr'] = np.array(s).reshape(nrows, ashape[0], ashape[1])

    if cfitsio_use_standard_strings():
        data['Sscalar_nopad'] = ['hello', 'world', 'good', 'bye']
        data['Svec_nopad'][:, 0] = 'hello'
        data['Svec_nopad'][:, 1] = 'world'

        s = 1 + np.arange(nrows*ashape[0]*ashape[1])
        s = ['%s' % el for el in s]
        data['Sarr_nopad'] = np.array(s).reshape(nrows, ashape[0], ashape[1])

    if sys.version_info >= (3, 0, 0):
        data['Uscalar'] = [
            '%-6s' % s for s in ['hello', 'world', 'good', 'bye']
        ]
        data['Uvec'][:, 0] = '%-6s' % 'hello'
        data['Uvec'][:, 1] = '%-6s' % 'world'

        s = 1 + np.arange(nrows*ashape[0]*ashape[1])
        s = ['%-6s' % el for el in s]
        data['Uarr'] = np.array(s).reshape(nrows, ashape[0], ashape[1])

        if cfitsio_use_standard_strings():
            data['Uscalar_nopad'] = ['hello', 'world', 'good', 'bye']
            data['Uvec_nopad'][:, 0] = 'hello'
            data['Uvec_nopad'][:, 1] = 'world'

            s = 1 + np.arange(nrows*ashape[0]*ashape[1])
            s = ['%s' % el for el in s]
            data['Uarr_nopad'] = np.array(s).reshape(
                nrows, ashape[0], ashape[1],
            )

    # use a dict list so we can have comments
    # for long key we used the largest possible

    keys = [
        {'name': 'test1', 'value': 35},
        {'name': 'empty', 'value': ''},
        {'name': 'long_keyword_name', 'value': 'stuff'},
        {'name': 'test2', 'value': 'stuff',
         'comment': 'this is a string keyword'},
        {'name': 'dbl', 'value': 23.299843,
         'comment': "this is a double keyword"},
        {'name': 'edbl', 'value': 1.384123233e+43,
         'comment': "double keyword with exponent"},
        {'name': 'lng', 'value': 2**63-1, 'comment': 'this is a long keyword'},
        {'name': 'lngstr', 'value': lorem_ipsum, 'comment': 'long string'}
    ]

    # a second extension using the convenience function
    nrows2 = 10
    data2 = np.zeros(nrows2, dtype=dtype2)
    data2['index'] = np.arange(nrows2, dtype='i4')
    data2['x'] = np.arange(nrows2, dtype='f8')
    data2['y'] = np.arange(nrows2, dtype='f8')

    #
    # ascii table
    #

    nvec = 2
    ashape = (2, 3)
    Sdtype = 'S6'
    Udtype = 'U6'

    # we support writing i2, i4, i8, f4 f8, but when reading cfitsio always
    # reports their types as i4 and f8, so can't really use i8 and we are
    # forced to read all floats as f8 precision

    adtype = [
        ('i2scalar', 'i2'),
        ('i4scalar', 'i4'),
        #  ('i8scalar', 'i8'),
        ('f4scalar', 'f4'),
        ('f8scalar', 'f8'),
        ('Sscalar', Sdtype),
    ]
    if sys.version_info >= (3, 0, 0):
        adtype += [('Uscalar', Udtype)]

    nrows = 4
    try:
        tdt = np.dtype(adtype, align=True)
    except TypeError:  # older numpy may not understand `align` argument
        tdt = np.dtype(adtype)
    adata = np.zeros(nrows, dtype=tdt)

    adata['i2scalar'][:] = -32222 + np.arange(nrows, dtype='i2')
    adata['i4scalar'][:] = -1353423423 + np.arange(nrows, dtype='i4')
    adata['f4scalar'][:] = (
        -2.55555555555555555555555e35 + np.arange(nrows, dtype='f4')*1.e35
    )
    adata['f8scalar'][:] = (
        -2.55555555555555555555555e110 + np.arange(nrows, dtype='f8')*1.e110
    )
    adata['Sscalar'] = ['hello', 'world', 'good', 'bye']

    if sys.version_info >= (3, 0, 0):
        adata['Uscalar'] = ['hello', 'world', 'good', 'bye']

    ascii_data = adata

    #
    # for variable length columns
    #

    # all currently available types, scalar, 1-d and 2-d array columns
    dtype = [
        ('u1scalar', 'u1'),
        ('u1obj', 'O'),
        ('i1scalar', 'i1'),
        ('i1obj', 'O'),
        ('u2scalar', 'u2'),
        ('u2obj', 'O'),
        ('i2scalar', 'i2'),
        ('i2obj', 'O'),
        ('u4scalar', 'u4'),
        ('u4obj', 'O'),
        ('i4scalar', '<i4'),  # mix the byte orders a bit, test swapping
        ('i4obj', 'O'),
        ('i8scalar', 'i8'),
        ('i8obj', 'O'),
        ('f4scalar', 'f4'),
        ('f4obj', 'O'),
        ('f8scalar', '>f8'),
        ('f8obj', 'O'),

        ('u1vec', 'u1', nvec),
        ('i1vec', 'i1', nvec),
        ('u2vec', 'u2', nvec),
        ('i2vec', 'i2', nvec),
        ('u4vec', 'u4', nvec),
        ('i4vec', 'i4', nvec),
        ('i8vec', 'i8', nvec),
        ('f4vec', 'f4', nvec),
        ('f8vec', 'f8', nvec),

        ('u1arr', 'u1', ashape),
        ('i1arr', 'i1', ashape),
        ('u2arr', 'u2', ashape),
        ('i2arr', 'i2', ashape),
        ('u4arr', 'u4', ashape),
        ('i4arr', 'i4', ashape),
        ('i8arr', 'i8', ashape),
        ('f4arr', 'f4', ashape),
        ('f8arr', 'f8', ashape),

        # special case of (1,)
        ('f8arr_dim1', 'f8', (1, )),

        ('Sscalar', Sdtype),
        ('Sobj', 'O'),
        ('Svec', Sdtype, nvec),
        ('Sarr', Sdtype, ashape),
    ]

    if sys.version_info > (3, 0, 0):
        dtype += [
           ('Uscalar', Udtype),
           ('Uvec', Udtype, nvec),
           ('Uarr', Udtype, ashape)]

    nrows = 4
    vardata = np.zeros(nrows, dtype=dtype)

    for t in ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'i8', 'f4', 'f8']:
        vardata[t+'scalar'] = 1 + np.arange(nrows, dtype=t)
        vardata[t+'vec'] = 1 + np.arange(nrows*nvec, dtype=t).reshape(
            nrows, nvec,
        )
        arr = 1 + np.arange(nrows*ashape[0]*ashape[1], dtype=t)
        vardata[t+'arr'] = arr.reshape(nrows, ashape[0], ashape[1])

        for i in range(nrows):
            vardata[t+'obj'][i] = vardata[t+'vec'][i]

    # strings get padded when written to the fits file.  And the way I do
    # the read, I real all bytes (ala mrdfits) so the spaces are preserved.
    #
    # so for comparisons, we need to pad out the strings with blanks so we
    # can compare

    vardata['Sscalar'] = [
        '%-6s' % s for s in ['hello', 'world', 'good', 'bye']
    ]
    vardata['Svec'][:, 0] = '%-6s' % 'hello'
    vardata['Svec'][:, 1] = '%-6s' % 'world'

    s = 1 + np.arange(nrows * ashape[0] * ashape[1])
    s = ['%-6s' % el for el in s]
    vardata['Sarr'] = np.array(s).reshape(nrows, ashape[0], ashape[1])

    if sys.version_info >= (3, 0, 0):
        vardata['Uscalar'] = [
            '%-6s' % s for s in ['hello', 'world', 'good', 'bye']
        ]
        vardata['Uvec'][:, 0] = '%-6s' % 'hello'
        vardata['Uvec'][:, 1] = '%-6s' % 'world'

        s = 1 + np.arange(nrows*ashape[0]*ashape[1])
        s = ['%-6s' % el for el in s]
        vardata['Uarr'] = np.array(s).reshape(nrows, ashape[0], ashape[1])

    for i in range(nrows):
        vardata['Sobj'][i] = vardata['Sscalar'][i].rstrip()

    #
    # for bitcol columns
    #
    nvec = 2
    ashape = (21, 21)

    dtype = [
        ('b1vec', '?', nvec),
        ('b1arr', '?', ashape)
    ]

    nrows = 4
    bdata = np.zeros(nrows, dtype=dtype)

    for t in ['b1']:
        bdata[t+'vec'] = (np.arange(nrows*nvec) % 2 == 0).astype('?').reshape(
            nrows, nvec,
        )
        arr = (np.arange(nrows*ashape[0]*ashape[1]) % 2 == 0).astype('?')
        bdata[t+'arr'] = arr.reshape(nrows, ashape[0], ashape[1])

    return {
        'data': data,
        'keys': keys,
        'data2': data2,
        'ascii_data': ascii_data,
        'vardata': vardata,
        'bdata': bdata,
    }
