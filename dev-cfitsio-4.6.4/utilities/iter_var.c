#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

/*
  This program illustrates how to use the CFITSIO iterator function.
  It reads the input 'vari.fits' file, moves to the binary
  table in the "COMPRESSED_IMAGE" extension, and prints the 
  float-values in the column COMPRESSED_DATA.
*/
int main(int argc, char *argv[])
{
    /* external work function is passed to the iterator */
    extern int flux_rate(long totalrows, long offset, long firstrow,
	   long nrows, int ncols, iteratorCol *cols, void *user_strct);
    fitsfile *fptr;
    iteratorCol cols[3];  /* structure used by the iterator function */
    int n_cols=1; /* number of columns */
    long rows_per_loop, offset;

    int status, nkeys, keypos, hdutype, ii, jj;
    char filename[]  = "vari.fits";     /* name of rate FITS file */

    status = 0; 

    fits_open_file(&fptr, filename, READWRITE, &status); /* open file */

    /* move to the desired binary table extension */
    if (fits_movnam_hdu(fptr, BINARY_TBL, "COMPRESSED_IMAGE", 0, &status) )
        fits_report_error(stderr, status);    /* print out error messages */

    /* define input column structure members for the iterator function */
    fits_iter_set_by_name(&cols[0], fptr, "COMPRESSED_DATA", 0,  InputCol);

    rows_per_loop = 0;  /* use default optimum number of rows */
    offset = 0;         /* process all the rows */

    /* apply the rate function to each row of the table */
    printf("Calling iterator function...%d\n", status);

    fits_iterate_data(n_cols, cols, offset, rows_per_loop,
                      flux_rate, 0L, &status);

    fits_close_file(fptr, &status);      /* all done */

    if (status)
        fits_report_error(stderr, status);  /* print out error messages */

    return(status);
}
/*--------------------------------------------------------------------------*/
int flux_rate(long totalrows, long offset, long firstrow, long nrows,
             int ncols, iteratorCol *cols, void *user_strct ) 

/*
   Sample iterator function that prints the values (assumed to be
   of type float).
*/
{
    /* declare variables static to preserve their values between calls */
    static float *counts;

    /*--------------------------------------------------------*/
    /*  Initialization procedures: execute on the first call  */
    /*--------------------------------------------------------*/
    if (firstrow == 1)
    {

printf("Datatype of column = %d\n",fits_iter_get_datatype(&cols[0]));

       /* assign the input pointers to the appropriate arrays and null ptrs*/
       counts       = (float *)  fits_iter_get_array(&cols[0]);

    }

    /*--------------------------------------------*/
    /*  Main loop: process all the rows of data */
    /*--------------------------------------------*/

    /*  NOTE: 1st element of array is the null pixel value!  */
    /*  Loop from 1 to nrows, not 0 to nrows - 1.  */


    for (int ii = 1; ii <= nrows; ii++)
    {
       long repeat = fits_iter_get_repeat(&cols[0]);
       printf("repeat = %ld, %f\n", repeat, counts[ii]);
    }


    return(0);  /* return successful status */
}
