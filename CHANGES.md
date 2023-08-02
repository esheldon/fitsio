version 1.2.0
--------------

Changes

    - move to cfitsio 4

Bug Fixes

    - Fixed issue where the patched C fitsio headers were not
      first in the include path.
    - Added extra header guards and headers to help with compilation issues.
    - Fixed builds of the bundled cfitsio code to hide symbols and directly
      include the *.o files when linking.

version 1.1.10
--------------

Changes

Bug Fixes

    - Fix errors on 32 bit builds where default numpy integer
      types were 32 rather than 64 bit assumed by the C code.
    - Fix checks for rows being sent to C codes

version 1.1.9
-------------

Changes

    - Row subsets of table data are returned in the order sent by the user
      rather than sorted and unique.  E.g. rows = [2, 2, 1] will return
      data corresponding to that order of rows, including duplicates.
    - Removed deprecated `pkg_resources` from tests (M. Becker).
    - converted tests to use pytest
    - Improved doc strings (N. Tessore)

Bug Fixes
    - Bug updating header string cards was adding new key rather than updating
    - Bug gzip compression when not using quantization

version 1.1.8
-------------

Bug Fixes
    - Bug in repr of FITS where it only worked the first time it was generated

version 1.1.7
-------------

Bug Fixes

    - Bug converting extension names, should just ensure ascii rather than
      subset of ascii
    - Bugs in the repr of headers where history keywords had extra spaces.
      Also a bug with not upper casing things like 1.0E20 which is the
      fits standard

version 1.1.6
-------------

Bug Fixes

    - fixed bug where Updating an existing record in an image header raises an
      exception (user ussegliog)
    - fix bug append not forwarding arguments to write (Nicolas Tessore)

version 1.1.5
---------------------------------

Bug Fixes

    - Deal with case that a header keyword has non-ascii
      characters and the value is numerical. In this case
      we cannot convert the value because fits_read_key
      can segfault in some scenarios.  We instead return
      a string.
    - Fixed residual segfaults for header cards with non-ascii
      characters.
    - HIERARCH keywords are now properly parsed.
    - Header keywords with `*`, `#` or `?` in them now properly raise an error
      before writing since the generated FITS files cannot be read.

Changes

    - Non-allowed characters in header keywords are now converted to `_` instead
      of `JUNK___...`.

version 1.1.4
---------------------------------

New Features

    - Moved most testing to GitHub actions (linux, osx).
    - Added testing on ppc64le w/ TravisCI (thanks @asellappen)

Bug Fixes

    - Don't remove BLANK keywords in header clean
    - Preserve order of comments in header

Compatibility changes

    - moved to sing `bool` rather than `np.bool` to be compatible
      with numpy 1.2

version 1.1.3
---------------------------------

This release moves to cfitsio 3.49, which has bug fixes and now properly
supports reading certain classes of lossless compressed files

New Features

    - Added keywords to control compression
        - qlevel control the quantization level
        - qmethod set the quantization method
        - hcomp_scale, hcomp_smooth HCOMPRESS specific settings

        A nice result of this is that one can do lossless gzip compression
        (setting qlevel=0) and
    - Work around some types of garbage characters that might appear
      in headers

BACKWARDS INCOMPATIBLE CHANGES

    - non-ascii junk in headers is replaced by ascii characters to
      avoid segmentation faults in the python standard library
      when non-unicode characters are detected.  This will cause
      codes that check for consistency between copied headers
      to fail, since the header data is modified.

Bug Fixes

    - Write integer keywords using the long long support rather than long
    - Fix bug where a new file is started and the user can access a
      fictional HDU, causing book keeping problems
    - Return zero length result when requested rows have
      zero length (rainwoodman)

version 1.1.2
---------------------------------

Bug Fixes

    - Fixed deprecation warnings for extra keyword arguments.
    - Fixed SyntaxWarning: "is" with a literal (Michka Popoff)

version 1.1.1
---------------------------------

Bug Fixes

    - Fix bug in drvrnet.c in printf statement, causing compile
      issues on some systems.

version 1.1.0
---------------------------------

Bumping the minor version due to the update of the cfitsio version

This reverts to the behavior that compression settings are set as a toggle,
which is the cfitsio convention.  The user needs to turn compression on and off
selectively.  The alternative behavior, introduced in 1.0.1, broke the mode
where compression is set in the filename, as well as breaking with convention.

New Features

    - Updated to cfitsio version 3.470 (#261)
    - Add ability to stride (step value) when slicing (Dustin Jenkins)
    - Add feature to flip along axis when slicing (Dustin Jenkins)
    - Feature to ignore image scaling (Dustin Jenkins)

Bug Fixes

    - Fix error reading with an empty rows argument (rainwoodman)
    - Fix bug when reading slice with step, but no start/stop (Mike Jarvis)
    - Fix bug with clobber when compression is sent in filename


Deprecations

    - Removed the use of `**kwargs` in various read/write routines. This
      pattern was causing silent bugs. All functions now use explicit
      keyword arguments. A warning will be raised in any keyword arguments
      are passed. In version `1.2`, this warning will become an error.

version 1.0.5
---------------------------------

Bug Fixes

    - fixed bug getting `None` keywords
    - fixed bug writing 64 bit images (#256, #257)
    - fixed HISTORY card value not being read

version 1.0.4
---------------------------------

New Features

    - support for empty keywords in header, which are supported
      by the standard and are used for cosmetic comments

Bug Fixes

    - Fix for inserting bit columns and appending data with bitcols
    - deal with non-standard header values such as NAN
      [by returning as strings
    - fixed many bugs reading headers; these were a casualty of
      the header reading optimizations put in for  1.0.1

version 1.0.3
---------------------------------

This is a bug fix release

Bug Fixes

    - The new header reading code did not deal properly with some
      HIERARCH non-standard header key values.

version 1.0.2
---------------------------------

This is a bug fix release

Bug Fixes

    - the read_header function was not treating the case_sensitive
      keyword properly (Stephen Bailey)

version 1.0.1
---------------------------------

Backwards Incompatible Changes

    - Support for python 3 strings.
    - Support for proper string null termination.  This means you can read back exactly
      what you wrote.  However this departs from previous fitsio which used
      the non-standard cfitsio convention of padding strings with spaces.
    - Scalar indexing of FITS objects now returns a scalar, consistent
      with numpy indexing rules (rainwoodman)

New Features

    - Installation moved to setuptools from distutils.
    - Bundling of cfitsio now done with patches against the upstream
      version instead of direct edits to the upstream code.
    - Speed improvements for the read_header conveniance function, and
      reading of headers in general.

Bug Fixes

    - CONTINUE in headers are now properly read.  Note there is a corner
      case that is mis-handled by the underlying cfitsio library.  A bug
      report has been sent.  (thanks for help with Alex Drlica-Wagner
      identifying and testing this issue)
    - Fixed bug where some long strings were not properly written to headers
    - Fixed bug where compression settings for an open FITS object was inherited
      from the previous HDU by a new HDU
    - Fixed bug where comment strings were lost when setting the value in
      a FITSHDR entry
    - Fixed bug where get_comment was raising ValueError rather than KeyError
    - For py3 need to ensure by hand that strings sizes are greater than 0

Deprecations

    - removed `convert` keyword in `FITSRecord` and `FITSHDR` classes.

version 0.9.12
---------------------------------

New Features

    - Deal properly with undefined value header entries
    - can delete rows from a table
    - can insert rows with resize()
    - can create empty HDU extension for extensions beyond 0 (Felipe Menanteau)
    - sanitize string input for py3
    - GZIP_2 compression support (Felipe Menanteau)
    - Improvements to python packaging for easier installation.
    - Using cfitsio 3.430 now with patches for known bugs
    - Now support reading and writing bit columns (Eli Rykoff)
    - Can now read CONTINUE keywords in headers.  It is currently
      treated as a comment; full implementation to come. (Alex Drlica-Wagner)
    - Can now use a standard key dict when writing a header key using
      the write_key method via `**`, e.g. `write_key(**key_dict)`
      (Alex Drlica-Wagner)
    - Delete row sets and row ranges using the delete_rows() method
      for tables
    - Resize tables, adding or removing rows, using the resize() method for
      tables
    - make write_key usable with standard dictionary using the `**keydict`
      style
    - allow writing empty HDUs after the first one using
        ignore_empty=True to the FITS constructor or
        the write convenience function (Felipe Menanteau)
        We might make this the default in the future if
        it is found to be benign

Bug Fixes

    - Only raise exception when PLIO u4/u8 is selected now that u1/u2 is supported
      in cfitsio (Eli Rykoff)
    - link curl library if cfitsio linked to it
    - don't require numpy to run setup (Simon Conseil)
    - strings with understores in headers, such as `1_000_000` are now not converted to numbers in py3
    - check that the input fields names for tables are unique after converting
      to upper case
    - link against libm explicitly for compatibility on some systems


version 0.9.11
---------------------------------

New Features

    - Added trim_strings option to constructor and as keyword for read methods.
      If trim_strings=True is set, white space is trimmed from the end
      of all string columns upon reading.  This was introduced because
      cfitsio internally pads strings out with spaces to the full column
      width when writing, against the FITS standard.

    - Added read_raw() method to the FITS class, to read the raw underlying data
      from the file (Dustin Lang)

Bug Fixes

    - Fix bug reading hierarch keywords. recent changes to keyword parsing had
      broken reading of hierarch keywords
    - Fix for strings that look like expressions, e.g. '3-4' which were
      being evaluated rather than returned as strings.
    - Fix bug for missing key in FITSHDR object using the hdr[key]
      notation.  Also raise KeyError rather than ValueError

version 0.9.10
---------------

Bug Fixes

    - Fix variable length string column copying in python 3
    - Fix bug checking for max size in a variable length table column.
    - Raise an exception when writing to a table with data
      that has shape ()
    - exit test suite with non-zero exit code if a test fails

Continuous integration

    - the travis ci now runs unit tests, ignoring those that may fail
      when certain libraries/headers are not installed on the users system (for
      now this is only bzip2 support)
    - only particular pairs of python version/numpy version are tested

python3 compatibility

    - the compatibility is now built into the code rather than
      using 2to3 to modify code at install time.

Workarounds

    - It turns out that when python, numpy etc. are compiled with gcc 4*
      and fitsio is compiled with gcc 5* there is a problem, in some cases,
      reading from an array with not aligned memory.  This has to do with using
      the -O3 optimization flag when compiling cfitsio.  For replacing -O3 with
      -O2 fixes the issue.  This was an issue on linux in both anaconda python2
      and python3.


version 0.9.9.1
----------------------------------

New tag so that pypi will accept the updated version

version 0.9.9
----------------------------------

New Features

    - header_start, data_start, data_end now available in the
      info dictionary, as well as the new get_offsets() method
      to access these in a new dict.
      (thanks Dimitri Muna for the initial version of this)

Bug Fixes

    - Fix bug when writing new COMMENT fields (thanks Alex Drlica-Wagner for
      initial fix)
    - deal correctly with aligned data in some scenarios
      (thanks Ole Streicher)
    - use correct data type long for tile_dims_fits in
      the set_compression C code.  This avoids a crash
      on 32 but systems. (thanks Ole Streicher)
    - use correct data type npy_int64 for pointer in
      get_long_slices (this function is not not correctly
      named).  Avoids crash on some 32 bit systems.
      (thanks Ole Streicher)
    - use correct data type npy_int64 for pointer in
      PyFITSObject_create_image_hdu, rather than npy_intp.
      (thanks Ole Streicher)

version 0.9.8
----------------------------------

New Features

    - added read_scamp_head function to read the .head files output
        by SCAMP and return a FITSHDR object
    - reserved header space when creating image and table extensions
        and a header is being written.  This can improve performance
        substantially, especially on distributed file systems.
    - When possible write image data at HDU creation.  This can
        be a big performance improvement, especially on distributed file
        systems.
    - Support for reading bzipped FITS files.  (Dustin Lang)

    - Added option to use the system CFITSIO instead of the bundled one,
        by sending --use-system-fitsio. Strongly recommend only use cfitsio
        that are as new as the bundled one.  Also note the bundled cfitsio
        sometimes contains patches that are not yet upstream in an
        official cfitsio release
    - proper support for reading unsigned images compressed with PLIO.
        This is a patch directly on the cfitsio code base.  The same
        code is in the upstream, but not yet released.
    - New method reshape(dims) for images
    - When writing into an existing image HDU, and larger dimensions
        are required, the image is automatically expanded.

Bug Fixes

    - Fixed broken boolean fields in new versions of numpy (rainwoodman) Fixed
    - bug when image was None (for creating empty first HDU) removed -iarch in
    - setup.py for mac OS X.  This should
        work for versions Mavericks and Snow Leapard (Christopher Bonnett)
    - Reading a single string column was failing in some cases, this
        has been fixed
    - When creating a TableColumnSubset using [cols], the existence
        of the columns is checked immediately, rather than waiting for the
        check in the read()
    - make sure to convert correct endianness when writing during image HDU
        creation
    - Corrected the repr for single column subsets
    - only clean bzero,bscale,bunit from headers for TableHDU

Dev features

    - added travis ci

version 0.9.7
----------------------------------

New Features

    - python 3 compatibility
    - Adding a new HDU is now near constant time
    - Can now create an empty image extension using create_image_hdu
        and sending the dims= and dtype= keywords
    - Can now write into a sub-section of an existing image using the
        start= keyword.
    - Can now use a scalar slice for reading images, e.g.
        hdu[row1:row2, col]
      although this still currently retains the extra dimension
    - Use warnings instead of printing to stdout
    - IOError is now used to indicate a number of errors that
        were previously ValueError


version 0.9.6
--------------

New Features

    - use cfitsio 3370 to support new tile compression features
    - FITSRecord class to encapsulate all the ways one can represent header
      records.  This is now used internally in the FITSHDR class instead of raw
      dicts, but as FITSRecord inherits from dict this should be transparent.
    - FITSCard class; inherits from FITSRecord and is a special case for header
      card strings
    - One can directly add a fits header card string to the FITSHDR object
      using add_record

Bug Fixes

    - use literal_eval instead of eval for evaluating header values (D. Lang)
    - If input to write_keys is a FITSHDR, just use it instead of creating a
      new FITSHDR object. (D. Lang)
    - update existing keys when adding records to FITSHDR, except for
      comment and history fields.
    - fixed bug with empty string in header card
    - deal with cfitsio treating first 4 comments specially

version 0.9.5
--------------------------------

Note the version 0.9.4 was skipped because some people had been using the
master branch in production, which had version 0.9.4 set.  This will allow
automatic version detection to work.  In the future master will not have
the next version set until release.

New Features

    - Re-factored code to use sub-classes for each HDU type.  These are called
      ImageHDU, TableHDU, and AsciiTableHDU.
    - Write and read 32-bit and 64-bit complex table columns
    - Write and read boolean table columns (contributed by Dustin Lang)
    - Specify tile dimensions for compressed images.
    - write_comment and write_history methods added.
    - is_compressed() for image HDUs, True if tile compressed.
    - added `**keys` to the image hdu reading routines to provide a more uniform
      interface for all hdu types

Bug Fixes

    - Correct appending to COMMENT and HISTORY fields when writing a full
      header object.
    - Correct conversion of boolean keywords, writing and reading.
    - Strip out compression related reserved keywords when writing a
      user-provided header.
    - Simplified reading string columns in ascii tables so that certain
      incorrectly formatted tables from  CASUTools are now read accurately.
      The change was minimal and did not affect reading well formatted tables,
      so seemed worth it.
    - Support non-standard TSHORT and TFLOAT columns in ascii tables as
      generated by CASUTools.  They are non-standard but supporting them
      does not seem to break anything (pulled from Simon Walker).

All changes E. Sheldon except where noted.

version 0.9.3
--------------------------
New Features

    - Can write lists of arrays and dictionaries of arrays
      to fits tables.
    - Added iteration over HDUs in FITS class
    - Added iteration to the FITSHDU object
    - Added iteration to the FITSHDR header object
    - added checking that a hdu exists in the file, either
        by extension number or name, using the "in" syntax.  e.g.
            fits=fitsio.FITS(filename)
            if 'name' in fits:
                data=fits['name'].read()
    - added `**keys` to the read_header function
    - added get_exttype() to the FITSHDU class
        'BINARY_TBL' 'ASCII_TBL' 'IMAGE_HDU'
    - added get_nrows() for binary tables
    - added get_colnames()
    - added get_filename()
    - added get_info()
    - added get_nrows()
    - added get_vstorage()
    - added is_compressed()
    - added get_ext()

minor changes

    - raise error on malformed TDIM

Backwards incompatible changes

    - renamed some attributes; use the getters instead
        - `colnames` -> `_colnames`
        - `info` -> `_info`
        - `filename` -> `_filename`
        - `ext` -> `_ext`
        - `vstorage` -> `_vstorage`
        - `is_comparessed` -> `_is_compressed`
            ( use the getter )

Bug Fixes

    - newer numpys (1.6.2) were barfing adding a python float to u4 arrays.
    - Give a more clear error message for malformed TDIM header keywords
    - fixed bug displaying column info for string array columns in tables
    - got cfitsio patch to deal with very large compressed images, which were
      not read properly.  This is now in the latest cfitsio.
    - implemented workaround for bug where numpy declareds 'i8' arrays as type
      npy_longlong, which is not correct.
    - fixed bug in order of iteration of HDUs

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

    - Added reading of image slices, e.g. `f[ext][2:25, 10:100]`
    - Added insert_column(name, data, colnum=) method for HDUs., 2011-11-14 ESS
    - Added a verify_checksum() method for HDU objects. 2011-10-24, ESS
    - Headers are cleaned of required keyword before writing.  E.g. if you have
      with fitsio.FITS(file,'rw') as fits:
        fits.write(data, header=h)
      Keywords like NAXIS, TTYPE* etc are removed.  This allows you to read
      a header from a fits file and write it to another without clobbering
      the required keywords.

    - when accessing a column subset object, more metadata are shown
        `f[ext][name]`
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
            >>> fits=fitsio.FITS(filename)
            >>> data=fits[1][:]
            >>> data=fits[1][10:30]
            >>> data=fits[1][10:30:2]

      You can also specify a list of rows
            >>> rows=[3,8,25]
            >>> data=fits[1][rows]

      This is equivalent to
            >>> data=fits[1].read(rows=rows)

      To get columns subsets, the notation is similar.  The data are read
      when the rows are specified.  If a sequence of columns is entered,
      a recarray is returned, otherwise a simple array.
            >>> data=fits[1]['x'][:]
            >>> data=fits[1]['x','y'][3:20]
            >>> data=fits[1][column_list][row_list]


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
