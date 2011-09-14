import os
import tempfile
import numpy
import fitsio

import unittest

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadWrite)
    unittest.TextTestRunner(verbosity=2).run(suite)

class TestReadWrite(unittest.TestCase):
    def setUp(self):
        
        dtype=[('i1scalar','i1'),
               ('f','f4'),
               ('fvec','f4',2),
               ('darr','f8',(2,3)),#] 
               ('s','S5'),
               ('svec','S6',3),
               ('sarr','S2',(3,4))]
        dtype2=[('index','i4'),
                ('x','f8'),
                ('y','f8')]

        nrows=4
        data=numpy.zeros(4, dtype=dtype)

        data['i1scalar'] = 1 + numpy.arange(nrows, dtype='i1')
        data['f'] = 1 + numpy.arange(nrows, dtype='f4')
        data['fvec'] = 1 + numpy.arange(nrows*2,dtype='f4').reshape(nrows,2)
        data['darr'] = 1 + numpy.arange(nrows*2*3,dtype='f8').reshape(nrows,2,3)

        # strings get padded when written to the fits file.  And the way I do
        # the read, I real all bytes (ala mrdfits) so the spaces are preserved.
        # 
        # so for comparisons, we need to pad out the strings with blanks so we
        # can compare

        data['s'] = ['%-5s' % s for s in ['hello','world','and','bye']]
        data['svec'][:,0] = '%-6s' % 'hello'
        data['svec'][:,1] = '%-6s' % 'there'
        data['svec'][:,2] = '%-6s' % 'world'

        s = 1 + numpy.arange(nrows*3*4)
        s = ['%-2s' % el for el in s]
        data['sarr'] = numpy.array(s).reshape(nrows,3,4)

        self.data = data

        # use a dict list so we can have comments
        self.keys = [{'name':'test1','value':35},
                     {'name':'test2','value':'stuff','comment':'this is a string column'},
                     {'name':'dbl', 'value':23.299843,'comment':"this is a double columm"},
                     {'name':'lng','value':3423432,'comment':'this is a long column'}]

        # a second extension using the convenience function
        nrows2=10
        data2 = numpy.zeros(nrows2, dtype=dtype2)
        data2['index'] = numpy.arange(nrows2,dtype='i4')
        data2['x'] = numpy.arange(nrows2,dtype='f8')
        data2['y'] = numpy.arange(nrows2,dtype='f8')
        self.data2 = data2

    def testOpen(self):
        """
        See if we can open a file READWRITE, with a clobber of any existing
        file
        """
        fname=tempfile.mktemp(prefix='fitsio-FileOpen-',suffix='.fits')
        try:
            try:
                with fitsio.FITS(fname,'rw',clobber=True) as fits:
                    pass
                success=True
            except:
                success=False

            self.assertTrue(success)
        finally:
            if os.path.exists(fname):
                os.remove(fname)
 
    def testImageWriteRead(self):
        """
        Test a basic image write, data and a header, then reading back in to
        check the values
        """

        fname=tempfile.mktemp(prefix='fitsio-ImageWrite-',suffix='.fits')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:
                for dtype in ['u1','i1','u2','i2','u4','i4','i8','f4','f8']:
                    data = numpy.arange(5*20,dtype=dtype).reshape(5,20)
                    header={'DTYPE':dtype,'NBYTES':data.dtype.itemsize}
                    fits.write_image(data, header=header)
                    rdata = fits[-1].read()
                    res=numpy.where(rdata != data)
                    for w in res:
                        self.assertEqual(w.size,0,"testing read/write image '%s'" % dtype)

                    rh = fits[-1].read_header()
                    for k,v in header.iteritems():
                        rv = rh[k]
                        if isinstance(rv,str):
                            v = v.strip()
                            rv = rv.strip()
                        self.assertEqual(v,rv,"testing equal key '%s'" % k)
        finally:
            if os.path.exists(fname):
                os.remove(fname)
 

    def testTableWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values
        """

        fname=tempfile.mktemp(prefix='fitsio-TableWrite-',suffix='.fits')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                try:
                    fits.write_table(self.data, header=self.keys, extname='mytable')
                    write_success=True
                except:
                    write_success=False

                self.assertTrue(write_success,"testing write does not raise an error")
                if not write_success:
                    skipTest("cannot test result if write failed")

                d = fits[1].read()
                nbad = compare_tables(self.data, d)
                for f in self.data.dtype.names:
                    res=numpy.where(self.data[f] != d[f])
                    for w in res:
                        self.assertEqual(w.size,0,"testing column %s" % f)

                h = fits[1].read_header()
                for entry in self.keys:
                    name=entry['name'].upper()
                    value=entry['value']
                    hvalue = h[name]
                    if isinstance(hvalue,str):
                        hvalue = hvalue.strip()
                    self.assertEqual(value,hvalue,"testing header key '%s'" % name)

                    if 'comment' in entry:
                        self.assertEqual(entry['comment'].strip(),
                                         h.get_comment(name).strip(),
                                         "testing comment for header key '%s'" % name)

            # see if our convenience functions are working
            fitsio.write(fname, self.data2, 
                         extname="newext", 
                         header={'ra':335.2,'dec':-25.2})
            d = fitsio.read(fname, ext='newext')
            for f in self.data2.dtype.names:
                res=numpy.where(d[f] != self.data2[f])
                for w in res:
                    self.assertEqual(w.size,0,"test convenience reading back all")

            # now test read_column
            with fitsio.FITS(fname) as fits:

                for f in self.data2.dtype.names:
                    d = fits['newext'].read_column(f)
                    res=numpy.where(d != self.data2['index'])
                    for w in res:
                        self.assertEqual(w.size,0,"test reading back read_column('%s')" % f)
        finally:
            if os.path.exists(fname):
                os.remove(fname)



