-----------------------

New features

    - You can now read rows from a table HDU using slice notation. e.g.
      to read from extension 1
            fits=fitsio.FITS(filename)
            data=fits[1][:]
            data=fits[1][10:30]
            data=fits[1][10:30:2]
      You can also specify a list of rows
            rows=[3,8,25]
            data=fits[1][rows]
      this is equivalent to
            data=fits[1].read(rows=rows)

    - added read_slice method, which is used to implement the slice
      notation introduced above.

    - Using the new "where" method, you can get the set of rows in
      a table where an input expression evaluates to true.   
        >>> w=fits[ext].where('x > 3 && y < 25')
        >>> data=fits[ext].read(rows=w)
        >>> data=fits[ext][w]

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

significant code changes

    - Now using fits_read_tblbytes when reading all rows and columns. This
      is just as fast but does not bypass, and thus confuse, the read buffers.
    - Removed many direct uses of the internal cfitsio struct objects,
      preferring to use provided access functions.  This allowed compilation
      on older cfitsio that had different struct representations.

bug fixes

    - too many to list in this early release.
