"""
image HDU classes for fitslib, part of the fitsio package.

See the main docs at https://github.com/esheldon/fitsio

  Copyright (C) 2011  Erin Sheldon, BNL.  erin dot sheldon at gmail dot com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
from __future__ import with_statement, print_function
from functools import reduce

import numpy

from math import floor
from .base import HDUBase, IMAGE_HDU
from ..util import IS_PY3, array_to_native

# for python3 compat
if IS_PY3:
    xrange = range


class ImageHDU(HDUBase):
    def _update_info(self):
        """
        Call parent method and make sure this is in fact a
        image HDU.  Set dims in C order
        """
        super(ImageHDU, self)._update_info()

        if self._info['hdutype'] != IMAGE_HDU:
            mess = "Extension %s is not a Image HDU" % self.ext
            raise ValueError(mess)

        # convert to c order
        if 'dims' in self._info:
            self._info['dims'] = list(reversed(self._info['dims']))

    def has_data(self):
        """
        Determine if this HDU has any data

        For images, check that the dimensions are not zero.

        For tables, check that the row count is not zero
        """
        ndims = self._info.get('ndims', 0)
        if ndims == 0:
            return False
        else:
            return True

    def is_compressed(self):
        """
        returns true of this extension is compressed
        """
        return self._info['is_compressed_image'] == 1

    def get_comptype(self):
        """
        Get the compression type.

        None if the image is not compressed.
        """
        return self._info['comptype']

    def get_dims(self):
        """
        get the shape of the image.  Returns () for empty
        """
        if self._info['ndims'] != 0:
            dims = self._info['dims']
        else:
            dims = ()

        return dims

    def reshape(self, dims):
        """
        reshape an existing image to the requested dimensions

        parameters
        ----------
        dims: sequence
            Any sequence convertible to i8
        """

        adims = numpy.array(dims, ndmin=1, dtype='i8')
        self._FITS.reshape_image(self._ext+1, adims)

    def write(self, img, start=0, **keys):
        """
        Write the image into this HDU

        If data already exist in this HDU, they will be overwritten.  If the
        image to write is larger than the image on disk, or if the start
        position is such that the write would extend beyond the existing
        dimensions, the on-disk image is expanded.

        parameters
        ----------
        img: ndarray
            A simple numpy ndarray
        start: integer or sequence
            Where to start writing data.  Can be an integer offset
            into the entire array, or a sequence determining where
            in N-dimensional space to start.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        dims = self.get_dims()

        if img.dtype.fields is not None:
            raise ValueError("got recarray, expected regular ndarray")
        if img.size == 0:
            raise ValueError("data must have at least 1 row")

        # data must be c-contiguous and native byte order
        if not img.flags['C_CONTIGUOUS']:
            # this always makes a copy
            img_send = numpy.ascontiguousarray(img)
            array_to_native(img_send, inplace=True)
        else:
            img_send = array_to_native(img, inplace=False)

        if IS_PY3 and img_send.dtype.char == 'U':
            # for python3, we convert unicode to ascii
            # this will error if the character is not in ascii
            img_send = img_send.astype('S', copy=False)

        if not numpy.isscalar(start):
            # convert to scalar offset
            # note we use the on-disk data type to get itemsize

            offset = _convert_full_start_to_offset(dims, start)
        else:
            offset = start

        # see if we need to resize the image
        if self.has_data():
            self._expand_if_needed(dims, img.shape, start, offset)

        self._FITS.write_image(self._ext+1, img_send, offset+1)
        self._update_info()

    def read(self, **keys):
        """
        Read the image.

        If the HDU is an IMAGE_HDU, read the corresponding image.  Compression
        and scaling are dealt with properly.
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if not self.has_data():
            return None

        dtype, shape = self._get_dtype_and_shape()
        array = numpy.zeros(shape, dtype=dtype)
        self._FITS.read_image(self._ext+1, array)
        return array

    def _get_dtype_and_shape(self):
        """
        Get the numpy dtype and shape for image
        """
        npy_dtype = self._get_image_numpy_dtype()

        if self._info['ndims'] != 0:
            shape = self._info['dims']
        else:
            raise IOError("no image present in HDU")

        return npy_dtype, shape

    def _get_image_numpy_dtype(self):
        """
        Get the numpy dtype for the image
        """
        try:
            ftype = self._info['img_equiv_type']
            npy_type = _image_bitpix2npy[ftype]
        except KeyError:
            raise KeyError("unsupported fits data type: %d" % ftype)

        return npy_type

    def __getitem__(self, arg):
        """
        Get data from an image using python [] slice notation.

        e.g., [2:25, 4:45].
        """
        return self._read_image_slice(arg)

    def _read_image_slice(self, arg):
        """
        workhorse to read a slice
        """
        if 'ndims' not in self._info:
            raise ValueError("Attempt to slice empty extension")

        if isinstance(arg, slice):
            # one-dimensional, e.g. 2:20
            return self._read_image_slice((arg,))

        if not isinstance(arg, tuple):
            raise ValueError("arguments must be slices, one for each "
                             "dimension, e.g. [2:5] or [2:5,8:25] etc.")

        # should be a tuple of slices, one for each dimension
        # e.g. [2:3, 8:100]
        nd = len(arg)
        if nd != self._info['ndims']:
            raise ValueError("Got slice dimensions %d, "
                             "expected %d" % (nd, self._info['ndims']))

        targ = arg
        arg = []
        for a in targ:
            if isinstance(a, slice):
                arg.append(a)
            elif isinstance(a, int):
                arg.append(slice(a, a+1, 1))
            else:
                raise ValueError("arguments must be slices, e.g. 2:12")

        dims = self._info['dims']
        arrdims = []
        first = []
        last = []
        steps = []
        npy_dtype = self._get_image_numpy_dtype()

        # check the args and reverse dimensions since
        # fits is backwards from numpy
        dim = 0
        for slc in arg:
            start = slc.start
            stop = slc.stop
            step = slc.step

            if start is None:
                start = 0
            if stop is None:
                stop = dims[dim]
            if step is None:
                # Ensure sane defaults.
                if start <= stop:
                    step = 1
                else:
                    step = -1

            # Sanity checks for proper syntax.
            if (step > 0 and stop < start) or (step < 0 and start < stop):
                return numpy.empty(0, dtype=npy_dtype)
            if start < 0:
                start = dims[dim] + start
                if start < 0:
                    raise IndexError("Index out of bounds")

            if stop < 0:
                stop = dims[dim] + start + 1

            # move to 1-offset
            start = start + 1

            if stop > dims[dim]:
                stop = dims[dim]
            if stop < start:
                # A little black magic here.  The stop is offset by 2 to
                # accommodate the 1-offset of CFITSIO, and to move past the end
                # pixel to get the complete set after it is flipped along the
                # axis.  Maybe there is a clearer way to accomplish what this
                # offset is glossing over.
                # @at88mph 2019.10.10
                stop = stop + 2

            first.append(start)
            last.append(stop)

            # Negative step values are not used in CFITSIO as the dimension is
            # already properly calcualted.
            # @at88mph 2019.10.21
            steps.append(abs(step))
            arrdims.append(int(floor((stop - start) / step)) + 1)

            dim += 1

        first.reverse()
        last.reverse()
        steps.reverse()
        first = numpy.array(first, dtype='i8')
        last = numpy.array(last, dtype='i8')
        steps = numpy.array(steps, dtype='i8')

        array = numpy.zeros(arrdims, dtype=npy_dtype)
        self._FITS.read_image_slice(self._ext+1, first, last, steps,
                                    self._ignore_scaling, array)
        return array

    def _expand_if_needed(self, dims, write_dims, start, offset):
        """
        expand the on-disk image if the indended write will extend
        beyond the existing dimensions
        """
        from operator import mul

        if numpy.isscalar(start):
            start_is_scalar = True
        else:
            start_is_scalar = False

        existing_size = reduce(mul, dims, 1)
        required_size = offset + reduce(mul, write_dims, 1)

        if required_size > existing_size:
            # we need to expand the image
            ndim = len(dims)
            idim = len(write_dims)

            if start_is_scalar:
                if start == 0:
                    start = [0]*ndim
                else:
                    raise ValueError(
                        "When expanding "
                        "an existing image while writing, the start keyword "
                        "must have the same number of dimensions "
                        "as the image or be exactly 0, got %s " % start)

            if idim != ndim:
                raise ValueError(
                    "When expanding "
                    "an existing image while writing, the input image "
                    "must have the same number of dimensions "
                    "as the original.  "
                    "Got %d instead of %d" % (idim, ndim))
            new_dims = []
            for i in xrange(ndim):
                required_dim = start[i] + write_dims[i]

                if required_dim < dims[i]:
                    # careful not to shrink the image!
                    dimsize = dims[i]
                else:
                    dimsize = required_dim

                new_dims.append(dimsize)

            self.reshape(new_dims)

    def __repr__(self):
        """
        Representation for ImageHDU
        """
        text, spacing = self._get_repr_list()
        text.append("%simage info:" % spacing)
        cspacing = ' '*4

        # need this check for when we haven't written data yet
        if 'ndims' in self._info:
            if self._info['comptype'] is not None:
                text.append(
                    "%scompression: %s" % (cspacing, self._info['comptype']))

            if self._info['ndims'] != 0:
                dimstr = [str(d) for d in self._info['dims']]
                dimstr = ",".join(dimstr)
            else:
                dimstr = ''

            dt = _image_bitpix2npy[self._info['img_equiv_type']]
            text.append("%sdata type: %s" % (cspacing, dt))
            text.append("%sdims: [%s]" % (cspacing, dimstr))

        text = '\n'.join(text)
        return text


def _convert_full_start_to_offset(dims, start):
    # convert to scalar offset
    # note we use the on-disk data type to get itemsize
    ndim = len(dims)

    # convert sequence to pixel start
    if len(start) != ndim:
        m = "start has len %d, which does not match requested dims %d"
        raise ValueError(m % (len(start), ndim))

    # this is really strides / itemsize
    strides = [1]
    for i in xrange(1, ndim):
        strides.append(strides[i-1] * dims[ndim-i])

    strides.reverse()
    s = start
    start_index = sum([s[i]*strides[i] for i in xrange(ndim)])

    return start_index


# remember, you should be using the equivalent image type for this
_image_bitpix2npy = {
    8: 'u1',
    10: 'i1',
    16: 'i2',
    20: 'u2',
    32: 'i4',
    40: 'u4',
    64: 'i8',
    -32: 'f4',
    -64: 'f8'}
