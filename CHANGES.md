version 0.9.2
--------------------------

New Features
    - Much faster writing to tables when there are many columns.
    - Header object now has a setitem feature
        h['item'] = value
    - Header stores values now instead of the string rep
    - You can force names of fields read from tables to upper
      or lower case, either during construction of the FITS object
      using or at read time using the lower= and upper= keywords.

bug fixes
    - more sensible data structure for header keywords.  Now works in all known
      cases when reading and rewriting string fields.

version 0.9.1
-------------------------

New features

    - Added reading of image slices, e.g. f[ext][2:25, 10:100]
    - Added insert_column(name, data, colnum=) method for HDUs., 2011-11-14 ESS
    - Added a verify_checksum() method for HDU objects. 2011-10-24, ESS
    - Headers are cleaned of required keyword before writing.  E.g. if you have
      with fitsio.FITS(file,'rw') as fits:
        fits.write(data, header=h)
      Keywords like NAXIS, TTYPE* etc are removed.  This allows you to read
      a header from a fits file and write it to another without clobbering
      the required keywords.

    - when accessing a column subset object, more metadata are shown
        f[ext][name]
    - can write None as an image for extension 0, as supported by
      the spirit standard.  Similarly reading gives None in that case.
    - the setup.py is now set up for registering versions to pypi.

bug fixes

    - Fixed bug that occured sometimes when reading individual columns where a
      few bytes were not read.  Now using the internal cfitsio buffers more
      carefully.

    - Using fits_read_tblbytes when reading full rows fixes a bug that showed
      up in a particular file.

    - required header keywords are stripped from input header objects before
      writing.

version 0.9.0 (2011-10-21)
-------------------------

This is the first "official" release. A patched version of cfitsio 3.28 is now
bundled.  This will make it easier for folks to install, and provide a
consistent code base with which to develop.  Thanks to Eli Rykoff for
suggesting a bundle.  Thanks to Eli and Martin White for helping extensively
with testing.

On OS X, we now link properly with universal binaries on intel. Thanks to Eli
Rykoff for help with OS X testing and bug fixes.

New features


    - Write and read variable length columns.  When writing a table, any fields
      declared "object" ("O" type char) in the input array will be written to a
      variable length column.  For numbers, this means vectors of varying
      length.  For strings, it means varying length strings.

      When reading, there are two options.  1) By default the data are read
      into fixed length fields with padding to the maximum size in the table
      column.  This is a "least surprise" approach, since fancy indexing and
      other array ops will work as expectd.  2) To save memory, construct the
      FITS object with vstorage='object' to store the data as objects.  This
      storage can also be written back out to a new FITS file with variable
      length columns. You can also over-ride the default vstorage when calling
      read functions.
      
    - Write and read ascii tables.  cfitsio supports writing scalar 2- and
      4-byte integers, floats and doubles. But for reading only 4-byte integers
      and doubles are supported, presumably because of the ambiguity in the
      tform fields.  Scalar strings are fully supported in both reading and
      writing.  No array fields are supported for ascii.

    - Append rows to an existing table using the append method.
            >>> fits.write_table(data1)
            >>> fits[-1].append(data2)

    - Using the new "where" method, you can select rows in a table where an
      input expression evaluates to true.  The table is scanned row by row
      without a large read.  This is surprisingly fast, and useful for figuring
      out what sections of a large file you want to extract. only requires
      enough memory to hold the row indices.

            >>> w=fits[ext].where('x > 3 && y < 25')
            >>> data=fits[ext].read(rows=w)
            >>> data=fits[ext][w]

    - You can now read rows and columns from a table HDU using slice notation. e.g.
      to read row subsets from extension 1
            fits=fitsio.FITS(filename)
            data=fits[1][:]
            data=fits[1][10:30]
            data=fits[1][10:30:2]
      You can also specify a list of rows
            rows=[3,8,25]
            data=fits[1][rows]
      this is equivalent to
            data=fits[1].read(rows=rows)
      To get columns subsets, the notation is similar.  The data are read
      when the rows are specified.  If a sequence of columns is entered, 
      a recarray is returned, otherwise a simple array.
            data=fits[1]['x'][:]
            data=fits[1]['x','y'][3:20]
            data=fits[1][column_list][row_list]


    - Added support for EXTVER header keywords.  When choosing an HDU by name,
      this allows one to select among HDUs that have the same name. Thanks to
      Eli Rykoff for suggesting this feature and helping with testing.

    - Name matching for table columns and extension names is not
      case-insensitive by default.  You can turn on case sensitivity by
      constructing the FITS object with case_sensitive=True, or sending
      that keyword to the convenience functions read and read_header.

    - Added write_checksum method to the FITSHDU class, which computes the
      checksum for the HDU, both the data portion alone (DATASUM keyword)
      and the checksum complement for the entire HDU (CHECKSUM).

    - Added an extensive test suite.  Use this to run the tests
        fitsio.test.test()

    - Added fitsio.cfitsio_version() function, returns the cfitsio
      version as a string.

    - added read_slice method, which is used to implement the slice
      notation introduced above.

significant code changes

    - Now using fits_read_tblbytes when reading all rows and columns. This
      is just as fast but does not bypass, and thus confuse, the read buffers.
    - Removed many direct uses of the internal cfitsio struct objects,
      preferring to use provided access functions.  This allowed compilation
      on older cfitsio that had different struct representations.

bug fixes

    - too many to list in this early release.
