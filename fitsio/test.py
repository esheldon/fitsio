import os
import tempfile
import numpy
import fitsio

import unittest

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadWrite)
    unittest.TextTestRunner(verbosity=2).run(suite)

def testsimple():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSimple)
    unittest.TextTestRunner(verbosity=2).run(suite)

def testgz():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadWriteGZOnly)
    unittest.TextTestRunner(verbosity=2).run(suite)
def testbuff():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBufferProblem)
    unittest.TextTestRunner(verbosity=2).run(suite)



class TestReadWrite(unittest.TestCase):
    def setUp(self):
        
        nvec = 2
        ashape = (2,3)
        Sdtype = 'S6'
        # all currently available types, scalar, 1-d and 2-d array columns
        dtype=[('u1scalar','u1'),
               ('i1scalar','i1'),
               ('u2scalar','u2'),
               ('i2scalar','i2'),
               ('u4scalar','u4'),
               ('i4scalar','i4'),
               ('i8scalar','i8'),
               ('f4scalar','f4'),
               ('f8scalar','f8'),

               ('u1vec','u1',nvec),
               ('i1vec','i1',nvec),
               ('u2vec','u2',nvec),
               ('i2vec','i2',nvec),
               ('u4vec','u4',nvec),
               ('i4vec','i4',nvec),
               ('i8vec','i8',nvec),
               ('f4vec','f4',nvec),
               ('f8vec','f8',nvec),
 
               ('u1arr','u1',ashape),
               ('i1arr','i1',ashape),
               ('u2arr','u2',ashape),
               ('i2arr','i2',ashape),
               ('u4arr','u4',ashape),
               ('i4arr','i4',ashape),
               ('i8arr','i8',ashape),
               ('f4arr','f4',ashape),
               ('f8arr','f8',ashape),

               ('Sscalar',Sdtype),
               ('Svec',   Sdtype, nvec),
               ('Sarr',   Sdtype, ashape)]

        dtype2=[('index','i4'),
                ('x','f8'),
                ('y','f8')]

        nrows=4
        data=numpy.zeros(nrows, dtype=dtype)

        for t in ['u1','i1','u2','i2','u4','i4','i8','f4','f8']:
            data[t+'scalar'] = 1 + numpy.arange(nrows, dtype=t)
            data[t+'vec'] = 1 + numpy.arange(nrows*nvec,dtype=t).reshape(nrows,nvec)
            arr = 1 + numpy.arange(nrows*ashape[0]*ashape[1],dtype=t)
            data[t+'arr'] = arr.reshape(nrows,ashape[0],ashape[1])



        # strings get padded when written to the fits file.  And the way I do
        # the read, I real all bytes (ala mrdfits) so the spaces are preserved.
        # 
        # so for comparisons, we need to pad out the strings with blanks so we
        # can compare

        data['Sscalar'] = ['%-6s' % s for s in ['hello','world','good','bye']]
        data['Svec'][:,0] = '%-6s' % 'hello'
        data['Svec'][:,1] = '%-6s' % 'world'

        s = 1 + numpy.arange(nrows*ashape[0]*ashape[1])
        s = ['%-6s' % el for el in s]
        data['Sarr'] = numpy.array(s).reshape(nrows,ashape[0],ashape[1])

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

                    self.compare_array(data, rdata, "images")

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
 
    def testTileCompressedWriteRead(self):
        """
        Test a basic image write, data and a header, then reading back in to
        check the values
        """

        fname=tempfile.mktemp(prefix='fitsio-ImageWrite-',suffix='.fits.fz')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:
                # note i8 not supported for compressed!
                for compress in ['rice','gzip','hcompress','plio']:
                    if compress=='plio':
                        dtypes = ['i1','i2','i4','f4','f8']
                    else:
                        dtypes = ['u1','i1','u2','i2','u4','i4','f4','f8']

                    for dtype in dtypes:
                        data = numpy.arange(5*20,dtype=dtype).reshape(5,20)
                        header={'DTYPE':dtype,'NBYTES':data.dtype.itemsize}
                        fits.write_image(data, header=header, compress=compress)
                        rdata = fits[-1].read()

                        self.compare_array(data, rdata, "%s compressed images" % compress)


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
 

    def testMoveByName(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values
        """

        fname=tempfile.mktemp(prefix='fitsio-MoveByName-',suffix='.fits')
        nrows=3
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                data1=numpy.zeros(nrows,dtype=[('ra','f8'),('dec','f8')])
                data1['ra'] = numpy.random.random(nrows)
                data1['dec'] = numpy.random.random(nrows)
                fits.write_table(data1, extname='mytable')

                fits[-1].write_key("EXTVER", 1)

                data2=numpy.zeros(nrows,dtype=[('ra','f8'),('dec','f8')])
                data2['ra'] = numpy.random.random(nrows)
                data2['dec'] = numpy.random.random(nrows)

                fits.write_table(data2, extname='mytable')
                fits[-1].write_key("EXTVER", 2)

                hdunum1=fits.movnam_hdu('mytable',extver=1)
                self.assertEqual(hdunum1,2)
                hdunum2=fits.movnam_hdu('mytable',extver=2)
                self.assertEqual(hdunum2,3)

        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def testExtVer(self):
        """
        Test using extname and extver, all combinations I can think of
        """

        fname=tempfile.mktemp(prefix='fitsio-ExtVer-',suffix='.fits')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                img1=numpy.arange(2*3,dtype='i4').reshape(2,3) + 5
                img2=numpy.arange(2*3,dtype='i4').reshape(2,3) + 6
                img3=numpy.arange(2*3,dtype='i4').reshape(2,3) + 7

                nrows=3
                data1=numpy.zeros(nrows,dtype=[('num','i4'),('ra','f8'),('dec','f8')])
                data1['num'] = 1
                data1['ra'] = numpy.random.random(nrows)
                data1['dec'] = numpy.random.random(nrows)

                data2=numpy.zeros(nrows,dtype=[('num','i4'),('ra','f8'),('dec','f8')])
                data2['num'] = 2
                data2['ra'] = numpy.random.random(nrows)
                data2['dec'] = numpy.random.random(nrows)

                data3=numpy.zeros(nrows,dtype=[('num','i4'),('ra','f8'),('dec','f8')])
                data3['num'] = 3
                data3['ra'] = numpy.random.random(nrows)
                data3['dec'] = numpy.random.random(nrows)


                fits.write_image(img1, extname='myimage', extver=1)
                fits.write_table(data1)
                fits.write_table(data2,extname='mytable', extver=1)
                fits.write_image(img2, extname='myimage', extver=2)
                fits.write_table(data3, extname='mytable',extver=2)
                fits.write_image(img3)

                d1  = fits[1].read()
                d2  = fits['mytable'].read()
                d2b = fits['mytable',1].read()
                d3  = fits['mytable',2].read()


                for f in data1.dtype.names:
                    self.compare_rec(data1, d1, "data1")
                    self.compare_rec(data2, d2, "data2")
                    self.compare_rec(data2, d2b, "data2b")
                    self.compare_rec(data3, d3, "data3")

                dimg1  = fits[0].read()
                dimg1b = fits['myimage',1].read()
                dimg2  = fits['myimage',2].read()
                dimg3  = fits[5].read()

                self.compare_array(img1, dimg1,"img1")
                self.compare_array(img1, dimg1b,"img1b")
                self.compare_array(img2, dimg2,"img2")
                self.compare_array(img3, dimg3,"img3")


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
                self.compare_rec(self.data, d, "table read/write")

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
            self.compare_rec(self.data2, d, "table data2")
            # now test read_column
            with fitsio.FITS(fname) as fits:

                for f in self.data2.dtype.names:
                    d = fits['newext'].read_column(f)
                    self.compare_array(self.data2[f], d, "table field read '%s'" % f)
        finally:
            if os.path.exists(fname):
                os.remove(fname)


    def testTableSubsets(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values
        """

        fname=tempfile.mktemp(prefix='fitsio-TableWrite-',suffix='.fits')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                fits.write_table(self.data, header=self.keys, extname='mytable')


                rows = [1,3]
                d = fits[1].read(rows=rows)
                self.compare_rec_subrows(self.data, d, rows, "table subset")
                columns = ['i1scalar','f4arr']
                d = fits[1].read(columns=columns, rows=rows)

                for f in columns:
                    d = fits[1].read_column(f,rows=rows)
                    self.compare_array(self.data[f][rows], d, "row subset, multi-column '%s'" % f)
                for f in self.data.dtype.names:
                    d = fits[1].read_column(f,rows=rows)
                    self.compare_array(self.data[f][rows], d, "row subset, column '%s'" % f)

        finally:
            if os.path.exists(fname):
                os.remove(fname)



    def testGZWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values

        this code all works, but the file is zere size when done!
        """

        fname=tempfile.mktemp(prefix='fitsio-GZTableWrite-',suffix='.fits.gz')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                fits.write_table(self.data, header=self.keys, extname='mytable')

                d = fits[1].read()
                self.compare_rec(self.data, d, "gzip write/read")

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
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def compare_headerlist_header(self, header_list, header):
        pass
    def compare_array(self, arr1, arr2, name):
        self.assertEqual(arr1.shape, arr2.shape,
                         "testing arrays '%s' shapes are equal: "
                         "input %s, read: %s" % (name, arr1.shape, arr2.shape))

        res=numpy.where(arr1 != arr2)
        for i,w in enumerate(res):
            self.assertEqual(w.size,0,"testing array '%s' dim %d are equal" % (name,i))

    def compare_rec(self, rec1, rec2, name):
        for f in rec1.dtype.names:
            self.assertEqual(rec1[f].shape, rec2[f].shape,
                             "testing '%s' field '%s' shapes are equal: "
                             "input %s, read: %s" % (name, f,rec1[f].shape, rec2[f].shape))

            res=numpy.where(rec1[f] != rec2[f])
            for w in res:
                self.assertEqual(w.size,0,"testing column %s" % f)
    def compare_rec_subrows(self, rec1, rec2, rows, name):
        for f in rec1.dtype.names:
            self.assertEqual(rec1[f][rows].shape, rec2[f].shape,
                             "testing '%s' field '%s' shapes are equal: "
                             "input %s, read: %s" % (name, f,rec1[f].shape, rec2[f].shape))

            res=numpy.where(rec1[f][rows] != rec2[f])
            for w in res:
                self.assertEqual(w.size,0,"testing column %s" % f)

            #self.assertEqual(2,3,"on purpose error")


class TestBufferProblem(unittest.TestCase):
    def setUp(self):
        
        nvec = 2
        ashape = (2,3)
        Sdtype = 'S6'
        # all currently available types, scalar, 1-d and 2-d array columns
        dtype=[('u1scalar','u1'),
               ('i1scalar','i1'),
               ('u2scalar','u2'),
               ('i2scalar','i2'),
               ('u4scalar','u4'),
               ('i4scalar','i4'),
               ('i8scalar','i8'),
               ('f4scalar','f4'),
               ('f8scalar','f8'),

               ('u1vec','u1',nvec),
               ('i1vec','i1',nvec),
               ('u2vec','u2',nvec),
               ('i2vec','i2',nvec),
               ('u4vec','u4',nvec),
               ('i4vec','i4',nvec),
               ('i8vec','i8',nvec),
               ('f4vec','f4',nvec),
               ('f8vec','f8',nvec),
 
               ('u1arr','u1',ashape),
               ('i1arr','i1',ashape),
               ('u2arr','u2',ashape),
               ('i2arr','i2',ashape),
               ('u4arr','u4',ashape),
               ('i4arr','i4',ashape),
               ('i8arr','i8',ashape),
               ('f4arr','f4',ashape),
               ('f8arr','f8',ashape),

               ('Sscalar',Sdtype),
               ('Svec',   Sdtype, nvec),
               ('Sarr',   Sdtype, ashape)]

        dtype2=[('index','i4'),
                ('x','f8'),
                ('y','f8')]

        nrows=4
        data=numpy.zeros(4, dtype=dtype)

        for t in ['u1','i1','u2','i2','u4','i4','i8','f4','f8']:
            data[t+'scalar'] = 1 + numpy.arange(nrows, dtype=t)
            data[t+'vec'] = 1 + numpy.arange(nrows*nvec,dtype=t).reshape(nrows,nvec)
            arr = 1 + numpy.arange(nrows*ashape[0]*ashape[1],dtype=t)
            data[t+'arr'] = arr.reshape(nrows,ashape[0],ashape[1])



        # strings get padded when written to the fits file.  And the way I do
        # the read, I real all bytes (ala mrdfits) so the spaces are preserved.
        # 
        # so for comparisons, we need to pad out the strings with blanks so we
        # can compare

        data['Sscalar'] = ['%-6s' % s for s in ['hello','world','good','bye']]
        data['Svec'][:,0] = '%-6s' % 'hello'
        data['Svec'][:,1] = '%-6s' % 'world'

        s = 1 + numpy.arange(nrows*ashape[0]*ashape[1])
        s = ['%-6s' % el for el in s]
        data['Sarr'] = numpy.array(s).reshape(nrows,ashape[0],ashape[1])

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

    def testBuffer(self):
        """
        """

        fname=tempfile.mktemp(prefix='fitsio-GZTableWrite-',suffix='.fits')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:


                fits.write_table(self.data, header=self.keys, extname='mytable')
                rd = fits[-1].read()

                img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                fits.write_image(img)

                timg = fits[-1].read()

                res=numpy.where(timg != img)
                for w in res:
                    self.assertEqual(w.size,0,"testing image write/read consistent")

                d = fits[1].read()
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
        finally:
            if os.path.exists(fname):
                os.remove(fname)



class TestReadWriteGZOnly(unittest.TestCase):
    def setUp(self):
        
        nvec = 2
        ashape = (2,3)
        Sdtype = 'S6'
        # all currently available types, scalar, 1-d and 2-d array columns
        dtype=[('u1scalar','u1'),
               ('i1scalar','i1'),
               ('u2scalar','u2'),
               ('i2scalar','i2'),
               ('u4scalar','u4'),
               ('i4scalar','i4'),
               ('i8scalar','i8'),
               ('f4scalar','f4'),
               ('f8scalar','f8'),

               ('u1vec','u1',nvec),
               ('i1vec','i1',nvec),
               ('u2vec','u2',nvec),
               ('i2vec','i2',nvec),
               ('u4vec','u4',nvec),
               ('i4vec','i4',nvec),
               ('i8vec','i8',nvec),
               ('f4vec','f4',nvec),
               ('f8vec','f8',nvec),
 
               ('u1arr','u1',ashape),
               ('i1arr','i1',ashape),
               ('u2arr','u2',ashape),
               ('i2arr','i2',ashape),
               ('u4arr','u4',ashape),
               ('i4arr','i4',ashape),
               ('i8arr','i8',ashape),
               ('f4arr','f4',ashape),
               ('f8arr','f8',ashape),

               ('Sscalar',Sdtype),
               ('Svec',   Sdtype, nvec),
               ('Sarr',   Sdtype, ashape)]

        dtype2=[('index','i4'),
                ('x','f8'),
                ('y','f8')]

        nrows=4
        data=numpy.zeros(4, dtype=dtype)

        for t in ['u1','i1','u2','i2','u4','i4','i8','f4','f8']:
            data[t+'scalar'] = 1 + numpy.arange(nrows, dtype=t)
            data[t+'vec'] = 1 + numpy.arange(nrows*nvec,dtype=t).reshape(nrows,nvec)
            arr = 1 + numpy.arange(nrows*ashape[0]*ashape[1],dtype=t)
            data[t+'arr'] = arr.reshape(nrows,ashape[0],ashape[1])



        # strings get padded when written to the fits file.  And the way I do
        # the read, I real all bytes (ala mrdfits) so the spaces are preserved.
        # 
        # so for comparisons, we need to pad out the strings with blanks so we
        # can compare

        data['Sscalar'] = ['%-6s' % s for s in ['hello','world','good','bye']]
        data['Svec'][:,0] = '%-6s' % 'hello'
        data['Svec'][:,1] = '%-6s' % 'world'

        s = 1 + numpy.arange(nrows*ashape[0]*ashape[1])
        s = ['%-6s' % el for el in s]
        data['Sarr'] = numpy.array(s).reshape(nrows,ashape[0],ashape[1])

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

    def testGZWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values

        this code all works, but the file is zere size when done!
        """

        fname=tempfile.mktemp(prefix='fitsio-TableWrite-',suffix='.fits.gz')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                #img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                #img=numpy.arange(5*5,dtype='f4').reshape(5,5)
                #fits.write_image(img)

                fits.write_table(self.data, header=self.keys, extname='mytable')
                rd = fits[-1].read()
            #with fitsio.FITS(fname,'rw',clobber=True) as fits:
                #fits.reopen()
                #return
                #c = fits[-1].read_column('i4scalar')

                img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                fits.write_image(img)
                #img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                #fits.write_image(img)

                timg = fits[-1].read()

                """
                d = fits[1].read()
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
                """
        finally:
            if os.path.exists(fname):
                os.remove(fname)




class TestReadWriteGZOnly(unittest.TestCase):
    def setUp(self):
        
        nvec = 2
        ashape = (2,3)
        Sdtype = 'S6'
        # all currently available types, scalar, 1-d and 2-d array columns
        dtype=[('u1scalar','u1'),
               ('i1scalar','i1'),
               ('u2scalar','u2'),
               ('i2scalar','i2'),
               ('u4scalar','u4'),
               ('i4scalar','i4'),
               ('i8scalar','i8'),
               ('f4scalar','f4'),
               ('f8scalar','f8'),

               ('u1vec','u1',nvec),
               ('i1vec','i1',nvec),
               ('u2vec','u2',nvec),
               ('i2vec','i2',nvec),
               ('u4vec','u4',nvec),
               ('i4vec','i4',nvec),
               ('i8vec','i8',nvec),
               ('f4vec','f4',nvec),
               ('f8vec','f8',nvec),
 
               ('u1arr','u1',ashape),
               ('i1arr','i1',ashape),
               ('u2arr','u2',ashape),
               ('i2arr','i2',ashape),
               ('u4arr','u4',ashape),
               ('i4arr','i4',ashape),
               ('i8arr','i8',ashape),
               ('f4arr','f4',ashape),
               ('f8arr','f8',ashape),

               ('Sscalar',Sdtype),
               ('Svec',   Sdtype, nvec),
               ('Sarr',   Sdtype, ashape)]

        dtype2=[('index','i4'),
                ('x','f8'),
                ('y','f8')]

        nrows=4
        data=numpy.zeros(4, dtype=dtype)

        for t in ['u1','i1','u2','i2','u4','i4','i8','f4','f8']:
            data[t+'scalar'] = 1 + numpy.arange(nrows, dtype=t)
            data[t+'vec'] = 1 + numpy.arange(nrows*nvec,dtype=t).reshape(nrows,nvec)
            arr = 1 + numpy.arange(nrows*ashape[0]*ashape[1],dtype=t)
            data[t+'arr'] = arr.reshape(nrows,ashape[0],ashape[1])



        # strings get padded when written to the fits file.  And the way I do
        # the read, I real all bytes (ala mrdfits) so the spaces are preserved.
        # 
        # so for comparisons, we need to pad out the strings with blanks so we
        # can compare

        data['Sscalar'] = ['%-6s' % s for s in ['hello','world','good','bye']]
        data['Svec'][:,0] = '%-6s' % 'hello'
        data['Svec'][:,1] = '%-6s' % 'world'

        s = 1 + numpy.arange(nrows*ashape[0]*ashape[1])
        s = ['%-6s' % el for el in s]
        data['Sarr'] = numpy.array(s).reshape(nrows,ashape[0],ashape[1])

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

    def testGZWriteRead(self):
        """
        Test a basic table write, data and a header, then reading back in to
        check the values

        this code all works, but the file is zere size when done!
        """

        fname=tempfile.mktemp(prefix='fitsio-TableWrite-',suffix='.fits.gz')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                #img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                #img=numpy.arange(5*5,dtype='f4').reshape(5,5)
                #fits.write_image(img)

                fits.write_table(self.data, header=self.keys, extname='mytable')
                rd = fits[-1].read()
            #with fitsio.FITS(fname,'rw',clobber=True) as fits:
                #fits.reopen()
                #return
                #c = fits[-1].read_column('i4scalar')

                img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                fits.write_image(img)
                #img=numpy.arange(2048*2048,dtype='f4').reshape(2048,2048)
                #fits.write_image(img)

                timg = fits[-1].read()

                """
                d = fits[1].read()
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
                """
        finally:
            if os.path.exists(fname):
                os.remove(fname)


class TestSimple(unittest.TestCase):
    def testWrite(self):
        """
        """

        fname=tempfile.mktemp(prefix='fitsio-ReadWrite-',suffix='.fits')
        try:
            with fitsio.FITS(fname,'rw',clobber=True) as fits:

                """
                d=numpy.arange(2*3,dtype='i4').reshape(2,3)
                fits.write_image(d)
                fits.write_image(d,extname="test")
                fits.write_image(d,extname="test",extver=2)
                """
                d=numpy.zeros(3,dtype=[('index','i4'),('ra','f8')])
                d['index'] = 1,2,3
                d['ra'] = 234., 44.2, 88.9
                fits.write_table(d)

        finally:
            if os.path.exists(fname):
                os.remove(fname)


