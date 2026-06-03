#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

/*
  This program illustrates how to use the CFITSIO iterator function.
  It reads and modifies the input 'iter_image.fit' image file by dividing
  all the pixel values by a factor of 10 (DESTROYING THE ORIGINAL IMAGE!!!)
*/
int main(int argc, char *argv[])
{
    /* external work function is passed to the iterator: */
    extern int div_image(long totalrows, long offset, long firstrow,
	   long nrows, int ncols, iteratorCol *cols, void *user_strct);
    fitsfile *fptr;
    iteratorCol cols[3];  /* structure used by the iterator function */
    int n_cols =1;
    long rows_per_loop =0 ,  /* use default optimum number of rows */
         offset =0;  /* process all the rows */

    int status=0, nkeys, keypos, hdutype, ii, jj;
    char filename[]  = "iter_image.fit";     /* name of image FITS file */

    fits_open_file(&fptr, filename, READWRITE, &status); /* open file */

    /* define input column structure members for the iterator function */
    fits_iter_set_file(&cols[0], fptr);
    fits_iter_set_iotype(&cols[0], InputOutputCol);
    fits_iter_set_datatype(&cols[0], 0);

    /* apply the div_image function to each row of the image */
    printf("Calling iterator function...%d\n", status);

    fits_iterate_data(n_cols, cols, offset, rows_per_loop,
                      div_image, 0L, &status);

    fits_close_file(fptr, &status);      /* all done */

    if (status)
        fits_report_error(stderr, status);  /* print out error messages */

    return status;
}
/*--------------------------------------------------------------------------*/
int div_image(long totalrows, long offset, long firstrow, long nrows,
             int ncols, iteratorCol *cols, void *user_strct ) 

/*
   Sample iterator function that takes all pixel values of the image
   and divides each individually by 100.
*/
{
    int status = 0;

    /* declare variables static to preserve their values between calls */
    static int *counts;

    /*--------------------------------------------------------*/
    /*  Initialization procedures: execute on the first call  */
    /*--------------------------------------------------------*/
    if (firstrow == 1)
    {
       if (ncols != 1)
           return(-1);  /* number of columns incorrect */

       /* assign the input pointers to the appropriate arrays and null ptrs*/
       counts       = (int *)  fits_iter_get_array(&cols[0]);
    }

    /*--------------------------------------------*/
    /*  Main loop: process all the rows of data */
    /*--------------------------------------------*/

    /*  NOTE: 1st element of array is the null pixel value!  */
    /*  Loop from 1 to nrows, not 0 to nrows - 1.  */

    for (int ii = 1; ii <= nrows; ii++)
    {
       counts[ii] /= 10 ;
    }
    printf("firstrows, nrows = %ld %ld\n", firstrow, nrows);

    return(0);  /* return successful status */
}
