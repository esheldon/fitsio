/*
 * fitsio_pywrap.c
 *
 * This is a CPython wrapper for the cfitsio library.

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
*/

#include <string.h>
#include <Python.h>
#include "fitsio.h"
#include "fitsio2.h"
#include "fitsio_pywrap_lists.h"
#include <numpy/arrayobject.h> 

// this is not defined anywhere in cfitsio except in
// the fits file structure
#define CFITSIO_MAX_ARRAY_DIMS 99

struct PyFITSObject {
    PyObject_HEAD
    fitsfile* fits;
};

static void 
set_ioerr_string_from_status(int status) {
    char status_str[FLEN_STATUS], errmsg[FLEN_ERRMSG];
    char message[1024];

    int nleft=1024;

    if (status) {
        fits_get_errstatus(status, status_str);  /* get the error description */

        sprintf(message, "FITSIO status = %d: %s\n", status, status_str);

        nleft -= strlen(status_str)+1;

        while ( nleft > 0 && fits_read_errmsg(errmsg) )  { /* get error stack messages */
            strncat(message, errmsg, nleft-1);
            nleft -= strlen(errmsg)+1;
            if (nleft >= 2) {
                strncat(message, "\n", nleft-1);
            }
            nleft-=2;
        }
        PyErr_SetString(PyExc_IOError, message);
    }
    return;
}



static int
PyFITSObject_init(struct PyFITSObject* self, PyObject *args, PyObject *kwds)
{
    char* filename;
    int mode;
    int status=0;
    int create=0;

    if (!PyArg_ParseTuple(args, (char*)"sii", &filename, &mode, &create)) {
        return -1;
    }

    if (create) {
        // create and open
        if (fits_create_file(&self->fits, filename, &status)) {
            set_ioerr_string_from_status(status);
            return -1;
        }
    } else {
        if (fits_open_file(&self->fits, filename, mode, &status)) {
            set_ioerr_string_from_status(status);
            return -1;
        }
    }

    return 0;
}

static PyObject *
PyFITSObject_repr(struct PyFITSObject* self) {

    if (self->fits != NULL) {
        int status=0;
        char filename[FLEN_FILENAME];
        char repr[2056];

        if (fits_file_name(self->fits, filename, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }

        sprintf(repr, "fits file: %s", filename);
        return PyString_FromString(repr);
    }  else {
        return PyString_FromString("");
    }
}

static PyObject *
PyFITSObject_filename(struct PyFITSObject* self) {

    if (self->fits != NULL) {
        int status=0;
        char filename[FLEN_FILENAME];
        PyObject* fnameObj=NULL;
        if (fits_file_name(self->fits, filename, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }

        fnameObj = PyString_FromString(filename);
        return fnameObj;
    }  else {
        PyErr_SetString(PyExc_ValueError, "file is not open, cannot determine name");
        return NULL;
    }
}



static PyObject *
PyFITSObject_close(struct PyFITSObject* self)
{
    int status=0;
    if (fits_close_file(self->fits, &status)) {
        self->fits=NULL;
        /*
        set_ioerr_string_from_status(status);
        return NULL;
        */
    }
    self->fits=NULL;
    Py_RETURN_NONE;
}



static void
PyFITSObject_dealloc(struct PyFITSObject* self)
{
    int status=0;
    fits_close_file(self->fits, &status);
#if PY_MAJOR_VERSION >= 3
    // introduced in python 2.6
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif
}



// if input is NULL or None, return NULL
// if maxlen is 0, the full string is copied, else
// it is maxlen+1 for the following null
/*
static char* copy_py_string(PyObject* obj, int maxlen, int* status) {
    char* buffer=NULL;
    int len=0;
    if (obj != NULL && obj != Py_None) {
        char* tmp;
        if (!PyString_Check(obj)) {
            PyErr_SetString(PyExc_ValueError, "Expected a string for extension name");
            *status=99;
            return NULL;
        }
        tmp = PyString_AsString(obj);
        if (maxlen > 0) {
            len = maxlen;
        } else {
            len = strlen(tmp);
        }
        buffer = malloc(sizeof(char)*(len+1));
        strncpy(buffer, tmp, len);
    }

    return buffer;
}
*/

// this will need to be updated for array string columns.
// I'm using a tcolumn* here, could cause problems
static long get_groupsize(tcolumn* colptr) {
    long gsize=0;
    if (colptr->tdatatype == TSTRING) {
        //gsize = colptr->twidth;
        gsize = colptr->trepeat;
    } else {
        gsize = colptr->twidth*colptr->trepeat;
    }
    return gsize;
}
static npy_int64* get_int64_from_array(PyObject* arr, npy_intp* ncols) {

    npy_int64* colnums;
    int npy_type=0;

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "int64 array must be an array.");
        return NULL;
    }

    npy_type = PyArray_TYPE(arr);
	if (npy_type != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "array must be an int64 array.");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(arr)) {
        PyErr_SetString(PyExc_TypeError, "int64 array must be a contiguous.");
        return NULL;
    }

    colnums = PyArray_DATA(arr);
    *ncols = PyArray_SIZE(arr);

    return colnums;
}

// move hdu by name and possibly version, return the hdu number
static PyObject *
PyFITSObject_movnam_hdu(struct PyFITSObject* self, PyObject* args) {
    int   status=0;
    int   hdutype=ANY_HDU; // means we don't care if its image or table
    char* extname=NULL;
    int   extver=0;        // zero means it is ignored
    int   hdunum=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"isi", &hdutype, &extname, &extver)) {
        return NULL;
    }

    if (fits_movnam_hdu(self->fits, hdutype, extname,  extver, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    
    fits_get_hdu_num(self->fits, &hdunum);
    return PyLong_FromLong((long)hdunum);
}



static PyObject *
PyFITSObject_movabs_hdu(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0, hdutype=0;
    int status=0;
    PyObject* hdutypeObj=NULL;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"i", &hdunum)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    hdutypeObj = PyLong_FromLong((long)hdutype);
    return hdutypeObj;
}

// get info for the specified HDU
static PyObject *
PyFITSObject_get_hdu_info(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0, hdutype=0, ext=0;
    int status=0, tstatus=0;
    PyObject* dict=NULL;

    char extname[FLEN_VALUE];
    char hduname[FLEN_VALUE];
    int extver=0, hduver=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"i", &hdunum)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }





    dict = PyDict_New();
    ext=hdunum-1;
    PyDict_SetItemString(dict, "hdunum", PyLong_FromLong((long)hdunum));
    PyDict_SetItemString(dict, "extnum", PyLong_FromLong((long)ext));
    PyDict_SetItemString(dict, "hdutype", PyLong_FromLong((long)hdutype));


    tstatus=0;
    if (fits_read_key(self->fits, TSTRING, "EXTNAME", extname, NULL, &tstatus)==0) {
        PyDict_SetItemString(dict, "extname", PyString_FromString(extname));
    } else {
        PyDict_SetItemString(dict, "extname", PyString_FromString(""));
    }
    tstatus=0;
    if (fits_read_key(self->fits, TSTRING, "HDUNAME", hduname, NULL, &tstatus)==0) {
        PyDict_SetItemString(dict, "hduname", PyString_FromString(hduname));
    } else {
        PyDict_SetItemString(dict, "hduname", PyString_FromString(""));
    }
    tstatus=0;
    if (fits_read_key(self->fits, TINT, "EXTVER", &extver, NULL, &tstatus)==0) {
        PyDict_SetItemString(dict, "extver", PyLong_FromLong((long)extver));
    } else {
        PyDict_SetItemString(dict, "extver", PyLong_FromLong((long)0));
    }
    tstatus=0;
    if (fits_read_key(self->fits, TINT, "HDUVER", &hduver, NULL, &tstatus)==0) {
        PyDict_SetItemString(dict, "hduver", PyLong_FromLong((long)hduver));
    } else {
        PyDict_SetItemString(dict, "hduver", PyLong_FromLong((long)0));
    }


    int ndims=0;
    int maxdim=CFITSIO_MAX_ARRAY_DIMS;
    LONGLONG dims[CFITSIO_MAX_ARRAY_DIMS];
    if (hdutype == IMAGE_HDU) {
        // move this into it's own func
        int tstatus=0;
        int bitpix=0;
        int bitpix_equiv=0;
        char comptype[20];
        PyObject* dimsObj=PyList_New(0);
        int i=0;

        //if (fits_read_imghdrll(self->fits, maxdim, simple_p, &bitpix, &ndims,
        //                       dims, pcount_p, gcount_p, extend_p, &status)) {
        if (fits_get_img_paramll(self->fits, maxdim, &bitpix, &ndims, dims, &tstatus)) {
            PyDict_SetItemString(dict, "error", PyString_FromString("could not determine image parameters"));
        } else {
            PyDict_SetItemString(dict, "ndims", PyLong_FromLong((long)ndims));
            PyDict_SetItemString(dict, "img_type", PyLong_FromLong((long)bitpix));

            fits_get_img_equivtype(self->fits, &bitpix_equiv, &status);
            PyDict_SetItemString(dict, "img_equiv_type", PyLong_FromLong((long)bitpix_equiv));

            tstatus=0;
            if (fits_read_key(self->fits, TSTRING, "ZCMPTYPE", comptype, NULL, &tstatus)==0) {
                PyDict_SetItemString(dict, "comptype", PyString_FromString(comptype));
            } else {
                Py_INCREF(Py_None);
                PyDict_SetItemString(dict, "comptype", Py_None);
            }

            for (i=0; i<ndims; i++) {
                PY_LONG_LONG d=dims[i];
                PyList_Append(dimsObj, PyLong_FromLongLong(d));
            }
            PyDict_SetItemString(dict, "dims", dimsObj);

        }

    } else if (hdutype == BINARY_TBL) {
        int tstatus=0;
        LONGLONG nrows=0;
        int ncols=0;
        PyObject* colinfo = PyList_New(0);
        int i=0,j=0;

        fits_get_num_rowsll(self->fits, &nrows, &tstatus);
        fits_get_num_cols(self->fits, &ncols, &tstatus);
        PyDict_SetItemString(dict, "nrows", PyLong_FromLongLong( (long long)nrows ));
        PyDict_SetItemString(dict, "ncols", PyLong_FromLong( (long)ncols));

        {
            tcolumn* col=NULL;
            struct stringlist* names=NULL;
            struct stringlist* tforms=NULL;
            names=stringlist_new();
            tforms=stringlist_new();

            for (i=0; i<ncols; i++) {
                stringlist_push_size(names, 70);
                stringlist_push_size(tforms, 70);
            }
            // just get the names: no other way to do it!
            fits_read_btblhdrll(self->fits, ncols, NULL, NULL, names->data, tforms->data, 
                                NULL, NULL, NULL, &tstatus);

            for (i=0; i<ncols; i++) {
                PyObject* d = PyDict_New();
                int type=0;
                LONGLONG repeat=0;
                LONGLONG width=0;

                PyDict_SetItemString(d, "name", PyString_FromString(names->data[i]));
                PyDict_SetItemString(d, "tform", PyString_FromString(tforms->data[i]));

                fits_get_coltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                PyDict_SetItemString(d, "type", PyLong_FromLong( (long)type));
                PyDict_SetItemString(d, "repeat", PyLong_FromLongLong( (long long)repeat));
                PyDict_SetItemString(d, "width", PyLong_FromLongLong( (long long)width));

                fits_get_eqcoltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                PyDict_SetItemString(d, "eqtype", PyLong_FromLong( (long)type));
                /*
                PyDict_SetItemString(d, "eqrepeat", PyLong_FromLongLong( (long long)repeat));
                PyDict_SetItemString(d, "eqwidth", PyLong_FromLongLong( (long long)width));
                */

                tstatus=0;
                if (fits_read_tdimll(self->fits, i+1, maxdim, &ndims, dims, &tstatus)) {
                    Py_INCREF(Py_None);
                    PyDict_SetItemString(d, "tdim", Py_None);
                } else {
                    PyObject* dimsObj=PyList_New(0);
                    for (j=0; j<ndims; j++) {
                        PY_LONG_LONG d=dims[j];
                        PyList_Append(dimsObj, PyLong_FromLongLong(d));
                    }

                    PyDict_SetItemString(d, "tdim", dimsObj);
                }

                // using the struct, could cause problems
                // actually, we can use ffgcprll to get this info, but will
                // be redundant with some others above
                col = &self->fits->Fptr->tableptr[i];
                PyDict_SetItemString(d, "tscale", PyFloat_FromDouble(col->tscale));
                PyDict_SetItemString(d, "tzero", PyFloat_FromDouble(col->tzero));

                PyList_Append(colinfo, d);
            }
            names=stringlist_delete(names);
            tforms=stringlist_delete(tforms);

            PyDict_SetItemString(dict, "colinfo", colinfo);
        }
    } else {
        int tstatus=0;
        LONGLONG nrows=0;
        int ncols=0;
        PyObject* colinfo = PyList_New(0);
        int i=0,j=0;

        fits_get_num_rowsll(self->fits, &nrows, &tstatus);
        fits_get_num_cols(self->fits, &ncols, &tstatus);
        PyDict_SetItemString(dict, "nrows", PyLong_FromLongLong( (long long)nrows ));
        PyDict_SetItemString(dict, "ncols", PyLong_FromLong( (long)ncols));

        {
            tcolumn* col=NULL;
            struct stringlist* names=NULL;
            struct stringlist* tforms=NULL;
            names=stringlist_new();
            tforms=stringlist_new();

            for (i=0; i<ncols; i++) {
                stringlist_push_size(names, 70);
                stringlist_push_size(tforms, 70);
            }
            // just get the names: no other way to do it!

            //                                        rowlen nrows
            fits_read_atblhdrll(self->fits, ncols, NULL, NULL,
            //          tfields             tbcol                units
                        NULL,   names->data, NULL, tforms->data, NULL,
            //          extname
                        NULL, &tstatus);



            for (i=0; i<ncols; i++) {
                PyObject* d = PyDict_New();
                int type=0;
                LONGLONG repeat=0;
                LONGLONG width=0;

                PyDict_SetItemString(d, "name", PyString_FromString(names->data[i]));
                PyDict_SetItemString(d, "tform", PyString_FromString(tforms->data[i]));

                fits_get_coltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                PyDict_SetItemString(d, "type", PyLong_FromLong( (long)type));
                PyDict_SetItemString(d, "repeat", PyLong_FromLongLong( (long long)repeat));
                PyDict_SetItemString(d, "width", PyLong_FromLongLong( (long long)width));

                fits_get_eqcoltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                PyDict_SetItemString(d, "eqtype", PyLong_FromLong( (long)type));
                /*
                PyDict_SetItemString(d, "eqrepeat", PyLong_FromLongLong( (long long)repeat));
                PyDict_SetItemString(d, "eqwidth", PyLong_FromLongLong( (long long)width));
                */

                tstatus=0;
                if (fits_read_tdimll(self->fits, i+1, maxdim, &ndims, dims, &tstatus)) {
                    Py_INCREF(Py_None);
                    PyDict_SetItemString(d, "tdim", Py_None);
                } else {
                    PyObject* dimsObj=PyList_New(0);
                    for (j=0; j<ndims; j++) {
                        PY_LONG_LONG d=dims[j];
                        PyList_Append(dimsObj, PyLong_FromLongLong(d));
                    }

                    PyDict_SetItemString(d, "tdim", dimsObj);
                }

                // using the struct, could cause problems
                // actually, we can use ffgcprll to get this info, but will
                // be redundant with some others above
                col = &self->fits->Fptr->tableptr[i];
                PyDict_SetItemString(d, "tscale", PyFloat_FromDouble(col->tscale));
                PyDict_SetItemString(d, "tzero", PyFloat_FromDouble(col->tzero));

                PyList_Append(colinfo, d);
            }
            names=stringlist_delete(names);
            tforms=stringlist_delete(tforms);

            PyDict_SetItemString(dict, "colinfo", colinfo);
        }

    }
    return dict;
}


// this is the parameter that goes in the type for fits_write_col
static int 
npy_to_fits_table_type(int npy_dtype) {

    char mess[255];
    switch (npy_dtype) {
        case NPY_UINT8:
            return TBYTE;
        case NPY_INT8:
            return TSBYTE;
        case NPY_UINT16:
            return TUSHORT;
        case NPY_INT16:
            return TSHORT;
        case NPY_UINT32:
            if (sizeof(unsigned int) == sizeof(npy_uint32)) {
                return TUINT;
            } else if (sizeof(unsigned long) == sizeof(npy_uint32)) {
                return TULONG;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine 4 byte unsigned integer type");
                return -9999;
            }
        case NPY_INT32:
            if (sizeof(int) == sizeof(npy_int32)) {
                return TINT;
            } else if (sizeof(long) == sizeof(npy_int32)) {
                return TLONG;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine 4 byte integer type");
                return -9999;
            }

        case NPY_INT64:
            if (sizeof(long long) == sizeof(npy_int64)) {
                return TLONGLONG;
            } else if (sizeof(long) == sizeof(npy_int64)) {
                return TLONG;
            } else if (sizeof(int) == sizeof(npy_int64)) {
                return TINT;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine 8 byte integer type");
                return -9999;
            }


        case NPY_FLOAT32:
            return TFLOAT;
        case NPY_FLOAT64:
            return TDOUBLE;

        case NPY_STRING:
            return TSTRING;

        case NPY_UINT64:
            PyErr_SetString(PyExc_TypeError, "Unsigned 8 byte integer images are not supported by the FITS standard");
            return -9999;

        default:
            sprintf(mess,"Unsupported numpy table datatype %d", npy_dtype);
            PyErr_SetString(PyExc_TypeError, mess);
            return -9999;
    }

    return 0;
}



static int 
npy_to_fits_image_types(int npy_dtype, int *fits_img_type, int *fits_datatype) {

    char mess[255];
    switch (npy_dtype) {
        case NPY_UINT8:
            *fits_img_type = BYTE_IMG;
            *fits_datatype = TBYTE;
            break;
        case NPY_INT8:
            *fits_img_type = SBYTE_IMG;
            *fits_datatype = TSBYTE;
            break;
        case NPY_UINT16:
            *fits_img_type = USHORT_IMG;
            *fits_datatype = TUSHORT;
            break;
        case NPY_INT16:
            *fits_img_type = SHORT_IMG;
            *fits_datatype = TSHORT;
            break;

        case NPY_UINT32:
            //*fits_img_type = ULONG_IMG;
            if (sizeof(unsigned short) == sizeof(npy_uint32)) {
                *fits_img_type = USHORT_IMG;
                *fits_datatype = TUSHORT;
            } else if (sizeof(unsigned int) == sizeof(npy_uint32)) {
                // there is no UINT_IMG, so use ULONG_IMG
                *fits_img_type = ULONG_IMG;
                *fits_datatype = TUINT;
            } else if (sizeof(unsigned long) == sizeof(npy_uint32)) {
                *fits_img_type = ULONG_IMG;
                *fits_datatype = TULONG;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine 4 byte unsigned integer type");
                *fits_datatype = -9999;
                return 1;
            }
            break;

        case NPY_INT32:
            //*fits_img_type = LONG_IMG;
            if (sizeof(unsigned short) == sizeof(npy_uint32)) {
                *fits_img_type = SHORT_IMG;
                *fits_datatype = TINT;
            } else if (sizeof(int) == sizeof(npy_int32)) {
                // there is no UINT_IMG, so use ULONG_IMG
                *fits_img_type = LONG_IMG;
                *fits_datatype = TINT;
            } else if (sizeof(long) == sizeof(npy_int32)) {
                *fits_img_type = LONG_IMG;
                *fits_datatype = TLONG;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine 4 byte integer type");
                *fits_datatype = -9999;
                return 1;
            }
            break;

        case NPY_INT64:
            //*fits_img_type = LONGLONG_IMG;
            if (sizeof(int) == sizeof(npy_int64)) {
                // there is no UINT_IMG, so use ULONG_IMG
                *fits_img_type = LONG_IMG;
                *fits_datatype = TINT;
            } else if (sizeof(long) == sizeof(npy_int64)) {
                *fits_img_type = LONG_IMG;
                *fits_datatype = TLONG;
            } else if (sizeof(long long) == sizeof(npy_int64)) {
                *fits_img_type = LONGLONG_IMG;
                *fits_datatype = TLONGLONG;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine 8 byte integer type");
                *fits_datatype = -9999;
                return 1;
            }
            break;


        case NPY_FLOAT32:
            *fits_img_type = FLOAT_IMG;
            *fits_datatype = TFLOAT;
            break;
        case NPY_FLOAT64:
            *fits_img_type = DOUBLE_IMG;
            *fits_datatype = TDOUBLE;
            break;

        case NPY_UINT64:
            PyErr_SetString(PyExc_TypeError, "Unsigned 8 byte integer images are not supported by the FITS standard");
            *fits_datatype = -9999;
            return 1;
            break;

        default:
            sprintf(mess,"Unsupported numpy image datatype %d", npy_dtype);
            PyErr_SetString(PyExc_TypeError, mess);
            *fits_datatype = -9999;
            return 1;
            break;
    }

    return 0;
}


/* 
 * this is really only for reading variable length columns since we should be
 * able to just read the bytes for normal columns
 */
static int fits_to_npy_table_type(int fits_dtype, int* isvariable) {

    if (fits_dtype < 0) {
        *isvariable=1;
    } else {
        *isvariable=0;
    }

    switch (abs(fits_dtype)) {
        case TBIT:
            return NPY_INT8;
        case TLOGICAL: // literal T or F stored as char
            return NPY_INT8;
        case TBYTE:
            return NPY_UINT8;
        case TSBYTE:
            return NPY_INT8;

        case TUSHORT:
            if (sizeof(unsigned short) == sizeof(npy_uint16)) {
                return NPY_UINT16;
            } else if (sizeof(unsigned short) == sizeof(npy_uint8)) {
                return NPY_UINT8;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TUSHORT");
                return -9999;
            }
        case TSHORT:
            if (sizeof(short) == sizeof(npy_int16)) {
                return NPY_INT16;
            } else if (sizeof(short) == sizeof(npy_int8)) {
                return NPY_INT8;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TSHORT");
                return -9999;
            }

        case TUINT:
            if (sizeof(unsigned int) == sizeof(npy_uint32)) {
                return NPY_UINT32;
            } else if (sizeof(unsigned int) == sizeof(npy_uint64)) {
                return NPY_UINT64;
            } else if (sizeof(unsigned int) == sizeof(npy_uint16)) {
                return NPY_UINT16;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TUINT");
                return -9999;
            }
        case TINT:
            if (sizeof(int) == sizeof(npy_int32)) {
                return NPY_INT32;
            } else if (sizeof(int) == sizeof(npy_int64)) {
                return NPY_INT64;
            } else if (sizeof(int) == sizeof(npy_int16)) {
                return NPY_INT16;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TINT");
                return -9999;
            }

        case TULONG:
            if (sizeof(unsigned long) == sizeof(npy_uint32)) {
                return NPY_UINT32;
            } else if (sizeof(unsigned long) == sizeof(npy_uint64)) {
                return NPY_UINT64;
            } else if (sizeof(unsigned long) == sizeof(npy_uint16)) {
                return NPY_UINT16;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TULONG");
                return -9999;
            }
        case TLONG:
            if (sizeof(unsigned long) == sizeof(npy_int32)) {
                return NPY_INT32;
            } else if (sizeof(unsigned long) == sizeof(npy_int64)) {
                return NPY_INT64;
            } else if (sizeof(long) == sizeof(npy_int16)) {
                return NPY_INT16;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TLONG");
                return -9999;
            }


        case TLONGLONG:
            if (sizeof(LONGLONG) == sizeof(npy_int64)) {
                return NPY_INT64;
            } else if (sizeof(LONGLONG) == sizeof(npy_int32)) {
                return NPY_INT32;
            } else if (sizeof(LONGLONG) == sizeof(npy_int16)) {
                return NPY_INT16;
            } else {
                PyErr_SetString(PyExc_TypeError, "could not determine numpy type for fits TLONGLONG");
                return -9999;
            }



        case TFLOAT:
            return NPY_FLOAT32;
        case TDOUBLE:
            return NPY_FLOAT64;

        case TSTRING:
            return NPY_STRING;

        default:
            PyErr_Format(PyExc_TypeError,"Unsupported FITS table datatype %d", fits_dtype); 
            return -9999;
    }

    return 0;
}





static int pyarray_get_ndim(PyObject* obj) {
    PyArrayObject* arr;
    arr = (PyArrayObject*) obj;
    return arr->nd;
}

// It is useful to create the extension first so we can write keywords
// into the header before adding data.  This avoid possibly moving the data
// if the header grows too large.
static PyObject *
PyFITSObject_create_image_hdu(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    // allow 10 dimensions
    int ndims=0;
    long dims[] = {0,0,0,0,0,0,0,0,0,0};
    int image_datatype=0; // fits type for image, AKA bitpix
    int datatype=0; // type for the data we entered
    //int comptype=NOCOMPRESS;
    int comptype=0; // same as NOCOMPRESS in newer cfitsio

    PyObject* array;
    int npy_dtype=0;
    int i=0;
    int status=0;

    char* extname=NULL;
    int extver=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    static char *kwlist[] = {"array","comptype", "extname", "extver", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|isi", kwlist,
                          &array, &comptype, &extname, &extver)) {
        goto create_image_hdu_cleanup;
    }


    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError, "input must be an array.");
        goto create_image_hdu_cleanup;
    }

    npy_dtype = PyArray_TYPE(array);
    if (npy_to_fits_image_types(npy_dtype, &image_datatype, &datatype)) {
        goto create_image_hdu_cleanup;
    }

    // order must be reversed for FITS
    ndims = pyarray_get_ndim(array);
    for (i=0; i<ndims; i++) {
        dims[ndims-i-1] = PyArray_DIM(array, i);
    }

    // can be NOCOMPRESS (0)
    if (fits_set_compression_type(self->fits, comptype, &status)) {
        set_ioerr_string_from_status(status);
        goto create_image_hdu_cleanup;
    }

    if (fits_create_img(self->fits, image_datatype, ndims, dims, &status)) {
        set_ioerr_string_from_status(status);
        goto create_image_hdu_cleanup;
    }

    if (extname != NULL) {
        if (strlen(extname) > 0) {

            // comments are NULL
            if (ffukys(self->fits, "EXTNAME", extname, NULL, &status)) {
                set_ioerr_string_from_status(status);
                goto create_image_hdu_cleanup;
            }
            if (extver > 0) {
                if (ffukyj(self->fits, "EXTVER", (LONGLONG) extver, NULL, &status)) {
                    set_ioerr_string_from_status(status);
                    goto create_image_hdu_cleanup;
                }
            }
        }
    }
    // this does a full close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        goto create_image_hdu_cleanup;
    }


create_image_hdu_cleanup:

    if (status != 0) {
        return NULL;
    }

    Py_RETURN_NONE;
}



// write the image to an existing HDU created using create_image_hdu
// dims are not checked
static PyObject *
PyFITSObject_write_image(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;
    LONGLONG nelements=1;
    LONGLONG firstpixel=1;
    int image_datatype=0; // fits type for image, AKA bitpix
    int datatype=0; // type for the data we entered

    PyObject* array;
    void* data=NULL;
    int npy_dtype=0;
    int status=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"iO", &hdunum, &array)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
 
    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError, "input must be an array.");
        return NULL;
    }

    npy_dtype = PyArray_TYPE(array);
    if (npy_to_fits_image_types(npy_dtype, &image_datatype, &datatype)) {
        return NULL;
    }


    data = PyArray_DATA(array);
    nelements = PyArray_SIZE(array);
    if (fits_write_img(self->fits, datatype, firstpixel, nelements, data, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this is a full file close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    Py_RETURN_NONE;
}


/*
 * Write tdims from the list.  The list must be the expected length.
 * Entries must be strings or None; if None the tdim is not written.
 *
 * The keys are written as TDIM{colnum}
 */
static int 
add_tdims_from_listobj(fitsfile* fits, PyObject* tdimObj, int ncols) {
    int status=0;
    size_t size=0, i=0;
    char keyname[20];
    int colnum=0;
    PyObject* tmp=NULL;
    char* tdim=NULL;

    if (tdimObj == NULL || tdimObj == Py_None) {
        // it is ok for it to be empty
        return 0;
    }

    if (!PyList_Check(tdimObj)) {
        PyErr_SetString(PyExc_ValueError, "Expected a list for tdims");
        return 1;
    }

    size = PyList_Size(tdimObj);
    if (size != ncols) {
        PyErr_Format(PyExc_ValueError, "Expected %d elements in tdims list, got %ld", ncols, size);
        return 1;
    }

    for (i=0; i<ncols; i++) {
        colnum=i+1;
        tmp = PyList_GetItem(tdimObj, i);
        if (tmp != Py_None) {
            if (!PyString_Check(tmp)) {
                PyErr_SetString(PyExc_ValueError, "Expected only strings or None for tdim");
                return 1;
            }

            tdim = PyString_AsString(tmp);
            sprintf(keyname, "TDIM%d", colnum);
            if (fits_write_key(fits, TSTRING, keyname, tdim, NULL, &status)) {
                set_ioerr_string_from_status(status);
                return 1;
            }
        }
    }


    return 0;
}


// create a new table structure.  No physical rows are added yet.
static PyObject *
PyFITSObject_create_table_hdu(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int table_type=0;
    int nfields=0;
    LONGLONG nrows=0; // start empty

    static char *kwlist[] = {"table_type","ttyp","tform","tunit", "tdim", "extname", "extver", NULL};
    // these are all strings
    PyObject* ttypObj=NULL;
    PyObject* tformObj=NULL;
    PyObject* tunitObj=NULL;    // optional
    PyObject* tdimObj=NULL;     // optional

    // these must be freed
    struct stringlist* ttyp=stringlist_new();
    struct stringlist* tform=stringlist_new();
    struct stringlist* tunit=stringlist_new();
    //struct stringlist* tdim=stringlist_new();
    char* extname=NULL;
    char* extname_use=NULL;
    int extver=0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iOO|OOsi", kwlist,
                          &table_type, &ttypObj, &tformObj, &tunitObj, &tdimObj, &extname, &extver)) {
        return NULL;
    }

    if (stringlist_addfrom_listobj(ttyp, ttypObj, "names")) {
        status=99;
        goto create_table_cleanup;
    }

    if (stringlist_addfrom_listobj(tform, tformObj, "formats")) {
        status=99;
        goto create_table_cleanup;
    }

    if (tunitObj != NULL && tunitObj != Py_None) {
        if (stringlist_addfrom_listobj(tunit, tunitObj,"units")) {
            status=99;
            goto create_table_cleanup;
        }
    }

    if (extname != NULL) {
        if (strlen(extname) > 0) {
            extname_use = extname;
        }
    }
    nfields = ttyp->size;
    if ( fits_create_tbl(self->fits, table_type, nrows, nfields, 
                         ttyp->data, tform->data, tunit->data, extname_use, &status) ) {
        set_ioerr_string_from_status(status);
        goto create_table_cleanup;
    }

    if (add_tdims_from_listobj(self->fits, tdimObj, nfields)) {
        status=99;
        goto create_table_cleanup;
    }

    if (extname_use != NULL) {
        if (extver > 0) {

            if (ffukyj(self->fits, "EXTVER", (LONGLONG) extver, NULL, &status)) {
                set_ioerr_string_from_status(status);
                goto create_table_cleanup;
            }
        }
    }

    // this does a full close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        goto create_table_cleanup;
    }

create_table_cleanup:
    ttyp = stringlist_delete(ttyp);
    tform = stringlist_delete(tform);
    tunit = stringlist_delete(tunit);
    //tdim = stringlist_delete(tdim);


    if (status != 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

// No error checking performed here
static
int write_string_column( 
        fitsfile *fits,  /* I - FITS file pointer                       */
        int  colnum,     /* I - number of column to write (1 = 1st col) */
        LONGLONG  firstrow,  /* I - first row to write (1 = 1st row)        */
        LONGLONG  firstelem, /* I - first vector element to write (1 = 1st) */
        LONGLONG  nelem,     /* I - number of strings to write              */
        char  *data,
        int  *status) {   /* IO - error status                           */

    LONGLONG i=0;
    LONGLONG twidth=0;
    // need to create a char** representation of the data, just point back
    // into the data array at string width offsets.  the fits_write_col_str
    // takes care of skipping between fields.
    char* cdata=NULL;
    char** strdata=NULL;

    // using struct def here, could cause problems
    twidth = fits->Fptr->tableptr[colnum-1].twidth;

    strdata = malloc(nelem*sizeof(char*));
    if (strdata == NULL) {
        PyErr_SetString(PyExc_MemoryError, "could not allocate temporary string pointers");
        *status = 99;
        return 1;
    }
    cdata = (char* ) data;
    for (i=0; i<nelem; i++) {
        strdata[i] = &cdata[twidth*i];
    }

    if( fits_write_col_str(fits, colnum, firstrow, firstelem, nelem, strdata, status)) {
        set_ioerr_string_from_status(*status);
        free(strdata);
        return 1;
    }


    free(strdata);

    return 0;
}


// write a column, starting at firstrow.  On the python side, the firstrow kwd
// should default to 1.
// You can append rows using firstrow = nrows+1
static PyObject *
PyFITSObject_write_column(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    int colnum=0;
    PyObject* array=NULL;

    void* data=NULL;
    PY_LONG_LONG firstrow_py=0;
    LONGLONG firstrow=1;
    LONGLONG firstelem=1;
    LONGLONG nelem=0;
    int npy_dtype=0;
    int fits_dtype=0;

    static char *kwlist[] = {"hdunum","colnum","array","firstrow", NULL};

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiOL", 
                  kwlist, &hdunum, &colnum, &array, &firstrow_py)) {
        return NULL;
    }
    firstrow = (LONGLONG) firstrow_py;

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_ValueError,"only arrays can be written to columns");
        return NULL;
    }

    npy_dtype = PyArray_TYPE(array);
    fits_dtype = npy_to_fits_table_type(npy_dtype);
    if (fits_dtype == -9999) {
        return NULL;
    }

    data = PyArray_DATA(array);
    nelem = PyArray_SIZE(array);

    if (fits_dtype == TSTRING) {

        // this is my wrapper for strings
        if (write_string_column(self->fits, colnum, firstrow, firstelem, nelem, data, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
        
    } else {
        if( fits_write_col(self->fits, fits_dtype, colnum, firstrow, firstelem, nelem, data, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    }

    // this is a full file close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    Py_RETURN_NONE;
}

// No error checking performed here
static
int write_var_string_column( 
        fitsfile *fits,  /* I - FITS file pointer                       */
        int  colnum,     /* I - number of column to write (1 = 1st col) */
        LONGLONG  firstrow,  /* I - first row to write (1 = 1st row)        */
        PyObject* array,
        int  *status) {   /* IO - error status                           */

    LONGLONG firstelem=1; // ignored
    LONGLONG nelem=1; // ignored
    npy_intp nrows=0;
    npy_intp i=0;
    char* ptr=NULL;
    int res=0;

    PyObject* format=NULL;
    PyObject* el=NULL;
    PyObject* el_string=NULL;
    char* strdata=NULL;
    char* strarr[1];
    int is_converted=0;

    format = PyString_FromString("%s");

    nrows = PyArray_SIZE(array);
    for (i=0; i<nrows; i++) {
        ptr = PyArray_GetPtr((PyArrayObject*) array, &i);
        el = PyArray_GETITEM(array, ptr);

        // convert to a string if needed
        if (PyString_Check(el)) {
            is_converted=0;
            // Don't free!
            strdata = PyString_AsString(el);
        } else {
            PyObject* args=PyTuple_New(1);

            PyTuple_SetItem(args,0,el);
            el_string = PyString_Format(format, args);

            Py_XDECREF(args);

            is_converted=1;
            // Don't free!
            strdata = PyString_AsString(el_string);
        }

        strarr[0] = strdata;
        res=fits_write_col_str(fits, colnum, 
                               firstrow+i, firstelem, nelem, 
                               strarr, status);
        if (is_converted) {
            Py_XDECREF(el_string);
        }

        if(res > 0) {
            goto write_var_string_column_cleanup;
        }
    }

write_var_string_column_cleanup:
    Py_XDECREF(format);
    if (*status > 0) {
        return 1;
    }

    return 0;
}

/* 
 * No error checking performed here
 */
static
int write_var_num_column( 
        fitsfile *fits,  /* I - FITS file pointer                       */
        int  colnum,     /* I - number of column to write (1 = 1st col) */
        LONGLONG  firstrow,  /* I - first row to write (1 = 1st row)        */
        int fits_dtype, 
        PyObject* array,
        int  *status) {   /* IO - error status                           */

    LONGLONG firstelem=1;
    npy_intp nelem=0;
    npy_intp nrows=0;
    npy_intp i=0;
    PyObject* el=NULL;
    PyObject* el_array=NULL;
    void* data=NULL;
    void* ptr=NULL;

    int npy_dtype=0, isvariable=0;

    int mindepth=1, maxdepth=0;
    PyObject* context=NULL;
    int requirements = 
        NPY_C_CONTIGUOUS 
        | NPY_ALIGNED 
        | NPY_NOTSWAPPED 
        | NPY_ELEMENTSTRIDES;

    int res=0;

    npy_dtype = fits_to_npy_table_type(fits_dtype, &isvariable);

    nrows = PyArray_SIZE(array);
    for (i=0; i<nrows; i++) {
        ptr = PyArray_GetPtr((PyArrayObject*) array, &i);
        el = PyArray_GETITEM(array, ptr);

        // a copy is only made if needed
        el_array = PyArray_CheckFromAny(el, PyArray_DescrFromType(npy_dtype), 
                                        mindepth, maxdepth, 
                                        requirements, context);
        if (el_array == NULL) {
            // error message will already be set
            return 1;
        }

        nelem=PyArray_SIZE(el);
        data=PyArray_DATA(el_array);
        res=fits_write_col(fits, abs(fits_dtype), colnum, 
                           firstrow+i, firstelem, (LONGLONG) nelem, data, status);
        Py_XDECREF(el_array);

        if(res > 0) {
            set_ioerr_string_from_status(*status);
            return 1;
        }
    }

    return 0;
}




/* 
 * write a variable length column, starting at firstrow.  On the python side,
 * the firstrow kwd should default to 1.  You can append rows using firstrow =
 * nrows+1
 *
 * The input array should be of type NPY_OBJECT, and the elements
 * should be either all strings or numpy arrays of the same type
 */

static PyObject *
PyFITSObject_write_var_column(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    int colnum=0;
    PyObject* array=NULL;

    PY_LONG_LONG firstrow_py=0;
    LONGLONG firstrow=1;
    int npy_dtype=0;
    int fits_dtype=0;

    static char *kwlist[] = {"hdunum","colnum","array","firstrow", NULL};

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiOL", 
                  kwlist, &hdunum, &colnum, &array, &firstrow_py)) {
        return NULL;
    }
    firstrow = (LONGLONG) firstrow_py;

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_ValueError,"only arrays can be written to columns");
        return NULL;
    }

    npy_dtype = PyArray_TYPE(array);
    if (npy_dtype != NPY_OBJECT) {
        PyErr_SetString(PyExc_TypeError,"only object arrays can be written to variable length columns");
        return NULL;
    }

    // determine the fits dtype for this column.  We will use this to get data
    // from the array for writing
    if (fits_get_eqcoltypell(self->fits, colnum, &fits_dtype, NULL, NULL, &status) > 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_dtype == -TSTRING) {
        if (write_var_string_column(self->fits, colnum, firstrow, array, &status)) {
            if (status != 0) {
                set_ioerr_string_from_status(status);
            }
            return NULL;
        }
    } else {
        if (write_var_num_column(self->fits, colnum, firstrow, fits_dtype, array, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    }

    // this is a full file close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    Py_RETURN_NONE;
}


 

// let python do the conversions
static PyObject *
PyFITSObject_write_string_key(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* keyname=NULL;
    char* value=NULL;
    char* comment=NULL;
    char* comment_in=NULL;
 
    if (!PyArg_ParseTuple(args, (char*)"isss", &hdunum, &keyname, &value, &comment_in)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (strlen(comment_in) > 0) {
        comment=comment_in;
    }

    if (ffukys(self->fits, keyname, value, comment, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this does not close and reopen
    if (fits_flush_buffer(self->fits, 0, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    Py_RETURN_NONE;
}
 
static PyObject *
PyFITSObject_write_double_key(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    int decimals=-15;

    char* keyname=NULL;
    double value=0;
    char* comment=NULL;
    char* comment_in=NULL;
 
    if (!PyArg_ParseTuple(args, (char*)"isds", &hdunum, &keyname, &value, &comment_in)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (strlen(comment_in) > 0) {
        comment=comment_in;
    }

    if (ffukyd(self->fits, keyname, value, decimals, comment, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this does not close and reopen
    if (fits_flush_buffer(self->fits, 0, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    Py_RETURN_NONE;
}
 
static PyObject *
PyFITSObject_write_long_key(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* keyname=NULL;
    long value=0;
    char* comment=NULL;
    char* comment_in=NULL;
 
    if (!PyArg_ParseTuple(args, (char*)"isls", &hdunum, &keyname, &value, &comment_in)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (strlen(comment_in) > 0) {
        comment=comment_in;
    }

    if (ffukyj(self->fits, keyname, (LONGLONG) value, comment, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this does not close and reopen
    if (fits_flush_buffer(self->fits, 0, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    Py_RETURN_NONE;
}
 
/*
 * read a single, entire column from the current HDU, which must be a binary
 * table, into a normal unstrided array.  Because of the internal fits
 * buffering, and since we will read multiple at a time, this should be more
 * efficient.  No error checking is done. No scaling is performed here, that is
 * done in python.
 */

static int read_column_bytes(fitsfile* fits, int colnum, void* data, int* status) {
    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0, row=0;

    long gsize=0; // number of bytes in column
    long ngroups=0; // number to read
    long offset=0; // gap between groups, not stride

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    colptr = hdu->tableptr + (colnum-1);

    // need to deal with array string columns
    gsize = get_groupsize(colptr);
    ngroups = hdu->numrows;
    offset = hdu->rowlength-gsize;

    file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;

    // need to use internal file-move code because of bookkeeping
    if (ffmbyt(fits, file_pos, REPORT_EOF, status)) {
        return 1;
    }

    // Here we use the function to read everything at once
    if (ffgbytoff(fits, gsize, ngroups, offset, data, status)) {
        return 1;
    }
    return 0;
}

/*
 * read a single, entire column from an ascii table into the input array.  This
 * version uses the standard read column instead of our by-bytes version.
 *
 * A number of assumptions are made, such as that columns are scalar, which
 * is true for ascii.
 */

static int read_ascii_column_all(fitsfile* fits, int colnum, PyObject* array, int* status) {

    int npy_dtype=0;
    int fits_dtype=0;

    npy_intp nelem=0;
    LONGLONG firstelem=1;
    LONGLONG firstrow=1;
    int* anynul=NULL;
    void* nulval=0;
    char* nulstr=" ";
    void* data=NULL;
    char* cdata=NULL;

    npy_intp i=0;

    npy_dtype = PyArray_TYPE(array);
    fits_dtype = npy_to_fits_table_type(npy_dtype);

    nelem = PyArray_SIZE(array);

    if (fits_dtype == TSTRING) {
        LONGLONG twidth=0;
        char** strdata=NULL;

        cdata = (char*) PyArray_DATA(array);

        strdata=malloc(nelem*sizeof(char*));
        if (NULL==strdata) {
            PyErr_SetString(PyExc_MemoryError, "could not allocate temporary string pointers");
            *status = 99;
            return 1;

        }

        twidth=fits->Fptr->tableptr[colnum-1].twidth;
        for (i=0; i<nelem; i++) {
            strdata[i] = &cdata[twidth*i];
        }

        if (fits_read_col_str(fits,colnum,firstrow,firstelem,nelem,nulstr,strdata,anynul,status) > 0) {
            free(strdata);
            return 1;
        }

        free(strdata);

    } else {
        data=PyArray_DATA(array);
        if (fits_read_col(fits,fits_dtype,colnum,firstrow,firstelem,nelem,nulval,data,anynul,status) > 0) {
            return 1;
        }
    }

    return 0;

}
static int read_ascii_column_byrow(
        fitsfile* fits, int colnum, PyObject* array, PyObject* rowsObj, int* status) {

    int npy_dtype=0;
    int fits_dtype=0;

    npy_intp nelem=0;
    LONGLONG firstelem=1;
    LONGLONG rownum=0;
    npy_intp nrows=-1;
    npy_intp stride=0;

    int* anynul=NULL;
    void* nulval=0;
    char* nulstr=" ";
    void* data=NULL;
    char* cdata=NULL;

    int dorows=0;

    npy_intp i=0;

    npy_dtype = PyArray_TYPE(array);
    fits_dtype = npy_to_fits_table_type(npy_dtype);

    nelem = PyArray_SIZE(array);


    if (rowsObj != Py_None) {
        dorows=1;
        nrows = PyArray_SIZE(rowsObj);
        if (nrows != nelem) {
            PyErr_Format(PyExc_ValueError, 
                    "input array[%ld] and rows[%ld] have different size", nelem,nrows);
            return 1;
        }
    }

    data = PyArray_GETPTR1(array, i);
    stride = PyArray_STRIDE(array,0);
    for (i=0; i<nrows; i++) {
        if (dorows) {
            rownum = (LONGLONG) (1 + *(npy_int64*) PyArray_GETPTR1(rowsObj, i));
        } else {
            rownum = (LONGLONG) (1+i);
        }
        // assuming 1-D
        data = PyArray_GETPTR1(array, i);
        if (fits_dtype==TSTRING) {
            cdata = (char* ) data;
            if (fits_read_col_str(fits,colnum,rownum,firstelem,1,nulstr,&cdata,anynul,status) > 0) {
                return 1;
            }
        } else {
            if (fits_read_col(fits,fits_dtype,colnum,rownum,firstelem,1,nulval,data,anynul,status) > 0) {
                return 1;
            }
        }
    }

    return 0;
}


static int read_ascii_column(fitsfile* fits, int colnum, PyObject* array, PyObject* rowsObj, int* status) {

    int ret=0;
    if (rowsObj != Py_None || !PyArray_ISCONTIGUOUS(array)) {
        ret = read_ascii_column_byrow(fits, colnum, array, rowsObj, status);
    } else {
        ret = read_ascii_column_all(fits, colnum, array, status);
    }

    return ret;
}




/*
 * Do we ever use this in practice? How much slower is this than above?
 */

static int read_column_bytes_strided(fitsfile* fits, int colnum, void* data, npy_intp stride, int* status) {
    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0, row=0;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    char* ptr;

    long gsize=0; // number of bytes in column
    long ngroups=0; // number to read
    long offset=0; // gap between groups, not stride

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    colptr = hdu->tableptr + (colnum-1);

    gsize = get_groupsize(colptr);
    ngroups = 1; // read one at a time
    offset = hdu->rowlength-gsize;

    ptr = (char*) data;
    for (row=0; row<hdu->numrows; row++) {
        file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;
        ffmbyt(fits, file_pos, REPORT_EOF, status);
        if (ffgbytoff(fits, gsize, ngroups, offset, (void*) ptr, status)) {
            return 1;
        }
        ptr += stride;
    }

    return 0;
}

// read a subset of rows for the input column
// the row array is assumed to be unique and sorted.
static int read_column_bytes_byrow(
        fitsfile* fits, 
        int colnum, 
        npy_intp nrows, 
        npy_int64* rows, 
        void* data, 
        npy_intp stride, 
        int* status) {

    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0, irow=0;
    npy_int64 row;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    char* ptr;

    long gsize=0; // number of bytes in column
    long ngroups=0; // number to read
    long offset=0; // gap between groups, not stride

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    colptr = hdu->tableptr + (colnum-1);

    // need to deal with array string columns
    gsize = get_groupsize(colptr);

    ngroups = 1; // read one at a time
    offset = hdu->rowlength-gsize;

    ptr = (char*) data;
    for (irow=0; irow<nrows; irow++) {
        row = rows[irow];
        file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;
        ffmbyt(fits, file_pos, REPORT_EOF, status);
        if (ffgbytoff(fits, gsize, ngroups, offset, (void*) ptr, status)) {
            return 1;
        }
        ptr += stride;
    }

    return 0;
}



/* 
 * read from a column into an input array
 */
static PyObject *
PyFITSObject_read_column(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;
    int colnum=0;

    FITSfile* hdu=NULL;
    int status=0;

    PyObject* array=NULL;
    void* data=NULL;
    npy_intp stride=0;

    PyObject* rowsObj;

    if (!PyArg_ParseTuple(args, (char*)"iiOO", &hdunum, &colnum, &array, &rowsObj)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // using struct defs here, could cause problems
    hdu = self->fits->Fptr;
    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot yet read columns from an IMAGE_HDU");
        return NULL;
    }
    if (colnum < 1 || colnum > hdu->tfield) {
        PyErr_SetString(PyExc_RuntimeError, "requested column is out of bounds");
        return NULL;
    }

    data = PyArray_DATA(array);
    
    if (hdutype == ASCII_TBL) {
        if (read_ascii_column(self->fits, colnum, array, rowsObj, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    } else {
        // we have some more optimizations for binary tables
        if (rowsObj == Py_None) {
            if (PyArray_ISCONTIGUOUS(array)) {
                if (read_column_bytes(self->fits, colnum, data, &status)) {
                    set_ioerr_string_from_status(status);
                    return NULL;
                }
            } else {
                stride = PyArray_STRIDE(array,0);
                if (read_column_bytes_strided(self->fits, colnum, data, stride, &status)) {
                    set_ioerr_string_from_status(status);
                    return NULL;
                }
            }
        } else {
            npy_intp nrows=0;
            npy_int64* rows=NULL;
            rows = get_int64_from_array(rowsObj, &nrows);
            if (rows == NULL) {
                return NULL;
            }
            stride = PyArray_STRIDE(array,0);
            if (read_column_bytes_byrow(self->fits, colnum, nrows, rows,
                        data, stride, &status)) {
                set_ioerr_string_from_status(status);
                return NULL;
            }

        }
    }
    Py_RETURN_NONE;
}




/*
 * Free all the elements in the python list as well as the list itself
 */
void free_all_python_list(PyObject* list) {
    if (PyList_Check(list)) {
        Py_ssize_t i=0;
        for (i=0; i<PyList_Size(list); i++) {
            Py_XDECREF(PyList_GetItem(list,i));
        }
    }
    Py_XDECREF(list);
}

PyObject* read_var_string(fitsfile* fits, int colnum, LONGLONG row, LONGLONG nchar, int* status) {
    LONGLONG firstelem=1;
    char* str=NULL;
    char* strarr[1];
    PyObject* stringObj=NULL;
    void* nulval=0;
    int* anynul=NULL;

    str=calloc(nchar,sizeof(char));
    if (str == NULL) {
        PyErr_Format(PyExc_MemoryError, 
                     "Could not allocate string of size %lld", nchar);
        return NULL;
    }

    strarr[0] = str;
    if (fits_read_col(fits,TSTRING,colnum,row,firstelem,nchar,nulval,strarr,anynul,status) > 0) {
        goto read_var_string_cleanup;
    }
    stringObj = PyString_FromString(str);
    if (NULL == stringObj) {
        PyErr_Format(PyExc_MemoryError, 
                     "Could not allocate py string of size %lld", nchar);
        goto read_var_string_cleanup;
    }

read_var_string_cleanup:
    free(str);

    return stringObj;
}
PyObject* read_var_nums(fitsfile* fits, int colnum, LONGLONG row, LONGLONG nelem, 
                        int fits_dtype, int npy_dtype, int* status) {
    LONGLONG firstelem=1;
    PyObject* arrayObj=NULL;
    void* nulval=0;
    int* anynul=NULL;
    npy_intp dims[1];
    int fortran=0;
    void* data=NULL;


    dims[0] = nelem;
    arrayObj=PyArray_ZEROS(1, dims, npy_dtype, fortran);
    if (arrayObj==NULL) {
        PyErr_Format(PyExc_MemoryError, 
                     "Could not allocate array type %d size %lld",npy_dtype,nelem);
        return NULL;
    }
    data = PyArray_DATA(arrayObj);
    if (fits_read_col(fits,abs(fits_dtype),colnum,row,firstelem,nelem,nulval,data,anynul,status) > 0) {
        Py_XDECREF(arrayObj);
        return NULL;
    }

    return arrayObj;
}
/*
 * read a variable length column as a list of arrays
 * what about strings?
 */
static PyObject *
PyFITSObject_read_var_column_as_list(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int colnum=0;
    PyObject* rowsObj=NULL;

    int hdutype=0;
    int ncols=0;
    const npy_int64* rows=NULL;
    LONGLONG nrows=0;
    int get_all_rows=0;

    int status=0, tstatus=0;

    int fits_dtype=0;
    int npy_dtype=0;
    int isvariable=0;
    LONGLONG repeat=0;
    LONGLONG width=0;
    LONGLONG offset=0;
    LONGLONG i=0;
    LONGLONG row=0;

    PyObject* listObj=NULL;
    PyObject* tempObj=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iiO", &hdunum, &colnum, &rowsObj)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot yet read columns from an IMAGE_HDU");
        return NULL;
    }
    // using struct defs here, could cause problems
    fits_get_num_cols(self->fits, &ncols, &status);
    if (colnum < 1 || colnum > ncols) {
        PyErr_SetString(PyExc_RuntimeError, "requested column is out of bounds");
        return NULL;
    }

    if (fits_get_coltypell(self->fits, colnum, &fits_dtype, &repeat, &width, &status) > 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    npy_dtype = fits_to_npy_table_type(fits_dtype, &isvariable);
    if (npy_dtype < 0) {
        return NULL;
    }
    if (!isvariable) {
        PyErr_Format(PyExc_TypeError,"Column %d not a variable length %d", colnum, fits_dtype); 
        return NULL;
    }
    
    if (rowsObj == Py_None) {
        fits_get_num_rowsll(self->fits, &nrows, &tstatus);
        get_all_rows=1;
    } else {
        npy_intp tnrows=0;
        rows = (const npy_int64*) get_int64_from_array(rowsObj, &tnrows);
        nrows=(LONGLONG) tnrows;
        get_all_rows=0;
    }

    listObj = PyList_New(0);

    for (i=0; i<nrows; i++) {
        if (get_all_rows) {
            row = i+1;
        } else {
            row = (LONGLONG) (rows[i]+1);
        }

        // repeat holds how many elements are in this row
        if (fits_read_descriptll(self->fits, colnum, row, &repeat, &offset, &status) > 0) {
            goto read_var_column_cleanup;
        }

        if (fits_dtype == -TSTRING) {
            tempObj = read_var_string(self->fits,colnum,row,repeat,&status);
        } else {
            tempObj = read_var_nums(self->fits,colnum,row,repeat,
                                    fits_dtype,npy_dtype,&status);
        }
        if (tempObj == NULL) {
            tstatus=1;
            goto read_var_column_cleanup;
        }
        PyList_Append(listObj, tempObj);
    }


read_var_column_cleanup:

    if (status != 0 || tstatus != 0) {
        Py_XDECREF(tempObj);
        free_all_python_list(listObj);
        if (status != 0) {
            set_ioerr_string_from_status(status);
        }
        return NULL;
    }

    return listObj;
}
 
// read the specified columns, and all rows, into the data array.  It is
// assumed the data match the requested columns perfectly, and that the column
// list is sorted

static int read_rec_column_bytes(fitsfile* fits, npy_intp ncols, npy_int64* colnums, void* data, int* status) {
    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0, row=0;
    npy_intp col=0;
    npy_int64 colnum=0;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    char* ptr;

    long groupsize=0; // number of bytes in column
    long ngroups=1; // number to read, one for row-by-row reading
    long offset=0; // gap between groups, not stride.  zero since we aren't using it

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    ptr = (char*) data;
    for (row=0; row<hdu->numrows; row++) {

        for (col=0; col < ncols; col++) {

            colnum = colnums[col];
            colptr = hdu->tableptr + (colnum-1);

            groupsize = get_groupsize(colptr);

            file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;

            // can just do one status check, since status are inherited.
            ffmbyt(fits, file_pos, REPORT_EOF, status);
            if (ffgbytoff(fits, groupsize, ngroups, offset, (void*) ptr, status)) {
                return 1;
            }
            ptr += groupsize;
        }
    }

    return 0;
}

// read specified columns and rows
static int read_rec_column_bytes_byrow(
        fitsfile* fits, 
        npy_intp ncols, npy_int64* colnums, 
        npy_intp nrows, npy_int64* rows,
        void* data, int* status) {
    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0;
    npy_intp col=0;
    npy_int64 colnum=0;

    npy_intp irow=0;
    npy_int64 row=0;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    char* ptr;

    long groupsize=0; // number of bytes in column
    long ngroups=1; // number to read, one for row-by-row reading
    long offset=0; // gap between groups, not stride.  zero since we aren't using it

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    ptr = (char*) data;
    for (irow=0; irow<nrows; irow++) {
        row = rows[irow];
        for (col=0; col < ncols; col++) {

            colnum = colnums[col];
            colptr = hdu->tableptr + (colnum-1);

            groupsize = get_groupsize(colptr);

            file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;

            // can just do one status check, since status are inherited.
            ffmbyt(fits, file_pos, REPORT_EOF, status);
            if (ffgbytoff(fits, groupsize, ngroups, offset, (void*) ptr, status)) {
                return 1;
            }
            ptr += groupsize;
        }
    }

    return 0;
}



// python method for reading specified columns and rows
static PyObject *
PyFITSObject_read_columns_as_rec(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;
    npy_intp ncols=0;
    npy_int64* colnums=NULL;

    int status=0;

    PyObject* columnsobj=NULL;
    PyObject* array=NULL;
    void* data=NULL;

    PyObject* rowsobj=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iOOO", &hdunum, &columnsobj, &array, &rowsobj)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        goto recread_columns_cleanup;
    }

    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read IMAGE_HDU into a recarray");
        return NULL;
    }
    
    colnums = get_int64_from_array(columnsobj, &ncols);
    if (colnums == NULL) {
        return NULL;
    }

    data = PyArray_DATA(array);
    if (rowsobj == Py_None) {
        if (read_rec_column_bytes(self->fits, ncols, colnums, data, &status)) {
            goto recread_columns_cleanup;
        }
    } else {
        npy_intp nrows;
        npy_int64* rows=NULL;
        rows = get_int64_from_array(rowsobj, &nrows);
        if (rows == NULL) {
            return NULL;
        }
        if (read_rec_column_bytes_byrow(self->fits, ncols, colnums, nrows, rows, data, &status)) {
            goto recread_columns_cleanup;
        }
    }

recread_columns_cleanup:

    if (status != 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    Py_RETURN_NONE;
}



/* 
 * read specified columns and rows
 *
 * Move by offset instead of just groupsize; this allows us to read into a
 * recarray while skipping some fields, e.g. variable length array fields, to
 * be read separately.
 *
 * If rows is NULL, then nrows are read consecutively.
 */

static int read_columns_as_rec_byoffset(
        fitsfile* fits, 
        npy_intp ncols, 
        const npy_int64* colnums,         // columns to read from file
        const npy_int64* field_offsets,   // offsets of corresponding fields within array
        npy_intp nrows, 
        const npy_int64* rows,
        char* data, 
        npy_intp recsize, 
        int* status) {

    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0;
    npy_intp col=0;
    npy_int64 colnum=0;

    char* ptr=NULL;

    int get_all_rows=1;
    npy_intp irow=0;
    npy_int64 row=0;

    long groupsize=0; // number of bytes in column
    long ngroups=1; // number to read, one for row-by-row reading
    long group_gap=0; // gap between groups, zero since we aren't using it

    if (rows != NULL) {
        get_all_rows=0;
    }

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    for (irow=0; irow<nrows; irow++) {
        if (get_all_rows) {
            row=irow;
        } else {
            row = rows[irow];
        }
        for (col=0; col < ncols; col++) {

            // point to this field in the array, allows for skipping
            ptr = data + irow*recsize + field_offsets[col];

            colnum = colnums[col];
            colptr = hdu->tableptr + (colnum-1);

            groupsize = get_groupsize(colptr);

            file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;

            // can just do one status check, since status are inherited.
            ffmbyt(fits, file_pos, REPORT_EOF, status);
            if (ffgbytoff(fits, groupsize, ngroups, group_gap, (void*) ptr, status)) {
                return 1;
            }
        }
    }

    return 0;
}







/* python method for reading specified columns and rows, moving by offset in
 * the array to allow some fields not read.
 *
 * columnsObj is the columns in the fits file to read.
 * offsetsObj is the offsets of the corresponding fields into the array.
 */
static PyObject *
PyFITSObject_read_columns_as_rec_byoffset(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    npy_intp ncols=0;
    npy_intp noffsets=0;
    npy_intp nrows=0;
    const npy_int64* colnums=NULL;
    const npy_int64* offsets=NULL;
    const npy_int64* rows=NULL;

    PyObject* columnsObj=NULL;
    PyObject* offsetsObj=NULL;
    PyObject* rowsObj=NULL;

    PyObject* array=NULL;
    void* data=NULL;
    npy_intp recsize=0;

    if (!PyArg_ParseTuple(args, (char*)"iOOOO", &hdunum, &columnsObj, &offsetsObj, &array, &rowsObj)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        goto recread_columns_byoffset_cleanup;
    }

    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read IMAGE_HDU into a recarray");
        return NULL;
    }
    
    colnums = (const npy_int64*) get_int64_from_array(columnsObj, &ncols);
    if (colnums == NULL) {
        return NULL;
    }
    offsets = (const npy_int64*) get_int64_from_array(offsetsObj, &noffsets);
    if (offsets == NULL) {
        return NULL;
    }
    if (noffsets != ncols) {
        PyErr_Format(PyExc_ValueError, 
                     "%ld columns requested but got %ld offsets", 
                     ncols, noffsets);
        return NULL;
    }

    if (rowsObj != Py_None) {
        rows = (const npy_int64*) get_int64_from_array(rowsObj, &nrows);
    } else {
        nrows = PyArray_SIZE(array);
    }

    data = PyArray_DATA(array);
    recsize = PyArray_ITEMSIZE(array);
    if (read_columns_as_rec_byoffset(
                self->fits, 
                ncols, colnums, offsets,
                nrows, 
                rows, 
                (char*) data, 
                recsize,
                &status) > 0) {
        goto recread_columns_byoffset_cleanup;
    }

recread_columns_byoffset_cleanup:

    if (status != 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    Py_RETURN_NONE;
}
 
 
// read specified rows, all columns
static int read_rec_bytes_byrow(
        fitsfile* fits, 
        npy_intp nrows, npy_int64* rows,
        void* data, int* status) {
    FITSfile* hdu=NULL;
    LONGLONG file_pos=0;

    npy_intp irow=0;
    npy_int64 row=0;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    char* ptr;

    long ngroups=1; // number to read, one for row-by-row reading
    long offset=0; // gap between groups, not stride.  zero since we aren't using it

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    ptr = (char*) data;

    for (irow=0; irow<nrows; irow++) {
        row = rows[irow];
        file_pos = hdu->datastart + row*hdu->rowlength;

        // can just do one status check, since status are inherited.
        ffmbyt(fits, file_pos, REPORT_EOF, status);
        if (ffgbytoff(fits, hdu->rowlength, ngroups, offset, (void*) ptr, status)) {
            return 1;
        }
        ptr += hdu->rowlength;
    }

    return 0;
}

// python method to read all columns but subset of rows
static PyObject *
PyFITSObject_read_rows_as_rec(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;

    int status=0;
    PyObject* array=NULL;
    void* data=NULL;

    PyObject* rowsObj=NULL;
    npy_intp nrows=0;
    npy_int64* rows=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iOO", &hdunum, &array, &rowsObj)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        goto recread_byrow_cleanup;
    }

    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read IMAGE_HDU into a recarray");
        return NULL;
    }

    data = PyArray_DATA(array);

    rows = get_int64_from_array(rowsObj, &nrows);
    if (rows == NULL) {
        return NULL;
    }
 
    if (read_rec_bytes_byrow(self->fits, nrows, rows, data, &status)) {
        goto recread_byrow_cleanup;
    }

recread_byrow_cleanup:

    if (status != 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    Py_RETURN_NONE;
}
 




/* Read the range of rows, 1-offset. It is assumed the data match the table
 * perfectly.
 */

static int read_rec_range(fitsfile* fits, LONGLONG firstrow, LONGLONG nrows, void* data, int* status) {
    // can also use this for reading row ranges
    LONGLONG firstchar=1;
    LONGLONG nchars=0;

    nchars = (fits->Fptr)->rowlength*nrows;

    if (fits_read_tblbytes(fits, firstrow, firstchar, nchars, (unsigned char*) data, status)) {
        return 1;
    }

    return 0;
}




static PyObject *
PyFITSObject_read_as_rec(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;

    int status=0;
    PyObject* array=NULL;
    void* data=NULL;

    PY_LONG_LONG firstrow=0;
    PY_LONG_LONG lastrow=0;
    PY_LONG_LONG nrows=0;

    if (!PyArg_ParseTuple(args, (char*)"iLLO", &hdunum, &firstrow, &lastrow, &array)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        goto recread_slice_cleanup;
    }

    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read IMAGE_HDU into a recarray");
        return NULL;
    }

    data = PyArray_DATA(array);

    nrows=lastrow-firstrow+1;
    if (read_rec_range(self->fits, (LONGLONG)firstrow, (LONGLONG)nrows, data, &status)) {
        goto recread_slice_cleanup;
    }

recread_slice_cleanup:

    if (status != 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    Py_RETURN_NONE;
}
 
 
// read an n-dimensional "image" into the input array.  Only minimal checking
// of the input array is done.
static PyObject *
PyFITSObject_read_image(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;
    int status=0;
    PyObject* array=NULL;
    void* data=NULL;
    int npy_dtype=0;
    int dummy=0, fits_read_dtype=0;

    int maxdim=10;
    int datatype=0; // type info for axis
    int naxis=0; // number of axes
    int i=0;
    LONGLONG naxes[]={0,0,0,0,0,0,0,0,0,0};  // size of each axis
    LONGLONG firstpixels[]={1,1,1,1,1,1,1,1,1,1};
    LONGLONG size=0;
    npy_intp arrsize=0;

    int anynul=0;

    if (!PyArg_ParseTuple(args, (char*)"iO", &hdunum, &array)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        return NULL;
    }

    if (fits_get_img_paramll(self->fits, maxdim, &datatype, &naxis, naxes, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // make sure dims match
    size=0;
    size = naxes[0];
    for (i=1; i< naxis; i++) {
        size *= naxes[i];
    }
    arrsize = PyArray_SIZE(array);
    data = PyArray_DATA(array);

    if (size != arrsize) {
        char mess[255];
        sprintf(mess,"Input array size is %ld but on disk array size is %lld", arrsize, size);
        PyErr_SetString(PyExc_RuntimeError, mess);
        return NULL;
    }

    npy_dtype = PyArray_TYPE(array);
    npy_to_fits_image_types(npy_dtype, &dummy, &fits_read_dtype);

    if (fits_read_pixll(self->fits, fits_read_dtype, firstpixels, size,
                        0, data, &anynul, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    Py_RETURN_NONE;
}

// read the entire header as list of dicts with name,value,comment and full
// card
static PyObject *
PyFITSObject_read_header(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char keyname[FLEN_KEYWORD];
    char value[FLEN_VALUE];
    char comment[FLEN_COMMENT];
    char card[FLEN_CARD];

    int nkeys=0, morekeys=0, i=0;

    PyObject* list=NULL;
    PyObject* dict=NULL;  // to hold the dict for each record


    if (!PyArg_ParseTuple(args, (char*)"i", &hdunum)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_get_hdrspace(self->fits, &nkeys, &morekeys, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    list=PyList_New(nkeys);
    for (i=0; i<nkeys; i++) {

        // the full card
        if (fits_read_record(self->fits, i+1, card, &status)) {
            // is this enough?
            Py_XDECREF(list);
            set_ioerr_string_from_status(status);
            return NULL;
        }

        // this just returns the character string stored in the header; we
        // can eval in python
        if (fits_read_keyn(self->fits, i+1, keyname, value, comment, &status)) {
            // is this enough?
            Py_XDECREF(list);
            set_ioerr_string_from_status(status);
            return NULL;
        }

        dict = PyDict_New();
        PyDict_SetItemString(dict, "card", PyString_FromString(card));
        PyDict_SetItemString(dict, "name", PyString_FromString(keyname));
        PyDict_SetItemString(dict, "value", PyString_FromString(value));
        PyDict_SetItemString(dict, "comment", PyString_FromString(comment));

        PyList_SetItem(list, i, dict);

    }

    return list;
}
 
static PyObject *
PyFITSObject_write_checksum(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    unsigned long datasum=0;
    unsigned long hdusum=0;

    PyObject* dict=NULL;

    if (!PyArg_ParseTuple(args, (char*)"i", &hdunum)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_write_chksum(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    if (fits_get_chksum(self->fits, &datasum, &hdusum, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    dict=PyDict_New();
    PyDict_SetItemString(dict, "datasum", PyLong_FromLongLong( (long long)datasum ));
    PyDict_SetItemString(dict, "hdusum", PyLong_FromLongLong( (long long)hdusum ));

    return dict;
}
static PyObject *
PyFITSObject_verify_checksum(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    int dataok=0, hduok=0;

    PyObject* dict=NULL;

    if (!PyArg_ParseTuple(args, (char*)"i", &hdunum)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_verify_chksum(self->fits, &dataok, &hduok, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    dict=PyDict_New();
    PyDict_SetItemString(dict, "dataok", PyLong_FromLong((long)dataok));
    PyDict_SetItemString(dict, "hduok", PyLong_FromLong((long)hduok));

    return dict;
}



static PyObject *
PyFITSObject_where(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    char* expression=NULL;

    LONGLONG nrows=0;

    long firstrow=1;
    long ngood=0;
    char* row_status=NULL;


    // Indices of rows for which expression is true
    PyObject* indicesObj=NULL;
    int ndim=1;
    npy_intp dims[1];
    npy_intp* data=NULL;
    long i=0;


    if (!PyArg_ParseTuple(args, (char*)"is", &hdunum, &expression)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_get_num_rowsll(self->fits, &nrows, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    row_status = malloc(nrows*sizeof(char));
    if (row_status==NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate row_status array");
        return NULL;
    }

    if (fits_find_rows(self->fits, expression, firstrow, (long) nrows, &ngood, row_status, &status)) {
        set_ioerr_string_from_status(status);
        goto where_function_cleanup;
    }

    dims[0] = ngood;
    indicesObj = PyArray_EMPTY(ndim, dims, NPY_INTP, 0);
    if (indicesObj == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate index array");
        goto where_function_cleanup;
    }

    if (ngood > 0) {
        data = PyArray_DATA(indicesObj);

        for (i=0; i<nrows; i++) {
            if (row_status[i]) {
                *data = (npy_intp) i;
                data++;
            }
        }
    }
where_function_cleanup:
    free(row_status);
    return indicesObj;
}

// generic functions, not tied to an object

static PyObject *
PyFITS_cfitsio_version(void) {
    float version=0;
    fits_get_version(&version);
    return PyFloat_FromDouble((double)version);
}



static PyMethodDef PyFITSObject_methods[] = {
    {"filename",             (PyCFunction)PyFITSObject_filename,             METH_VARARGS,  "filename\n\nReturn the name of the file."},

    {"where",           (PyCFunction)PyFITSObject_where,           METH_VARARGS,  "where\n\nReturn an index array where the input expression evaluates to true."},

    {"movabs_hdu",           (PyCFunction)PyFITSObject_movabs_hdu,           METH_VARARGS,  "movabs_hdu\n\nMove to the specified HDU."},
    {"movnam_hdu",           (PyCFunction)PyFITSObject_movnam_hdu,           METH_VARARGS,  "movnam_hdu\n\nMove to the specified HDU by name and return the hdu number."},

    {"get_hdu_info",         (PyCFunction)PyFITSObject_get_hdu_info,         METH_VARARGS,  "get_hdu_info\n\nReturn a dict with info about the specified HDU."},

    {"read_image",           (PyCFunction)PyFITSObject_read_image,           METH_VARARGS,  "read_image\n\nRead the entire n-dimensional image array.  No checking of array is done."},
    {"read_column",          (PyCFunction)PyFITSObject_read_column,          METH_VARARGS,  "read_column\n\nRead the column into the input array.  No checking of array is done."},
    {"read_var_column_as_list",          (PyCFunction)PyFITSObject_read_var_column_as_list,          METH_VARARGS,  "read_var_column_as_list\n\nRead the variable length column as a list of arrays."},
    {"read_columns_as_rec",  (PyCFunction)PyFITSObject_read_columns_as_rec,  METH_VARARGS,  "read_columns_as_rec\n\nRead the specified columns into the input rec array.  No checking of array is done."},
    {"read_columns_as_rec_byoffset",  (PyCFunction)PyFITSObject_read_columns_as_rec_byoffset,  METH_VARARGS,  "read_columns_as_rec_byoffset\n\nRead the specified columns into the input rec array at the specified offsets.  No checking of array is done."},
    {"read_rows_as_rec",     (PyCFunction)PyFITSObject_read_rows_as_rec,     METH_VARARGS,  "read_rows_as_rec\n\nRead the subset of rows into the input rec array.  No checking of array is done."},
    {"read_as_rec",          (PyCFunction)PyFITSObject_read_as_rec,          METH_VARARGS,  "read_as_rec\n\nRead a set of rows into the input rec array.  No significant checking of array is done."},
    {"read_header",          (PyCFunction)PyFITSObject_read_header,          METH_VARARGS,  "read_header\n\nRead the entire header as a list of dictionaries."},

    {"create_image_hdu",     (PyCFunction)PyFITSObject_create_image_hdu,     METH_KEYWORDS, "create_image_hdu\n\nWrite the input image to a new extension."},
    {"create_table_hdu",     (PyCFunction)PyFITSObject_create_table_hdu,     METH_KEYWORDS, "create_table_hdu\n\nCreate a new table with the input parameters."},

    {"write_checksum",       (PyCFunction)PyFITSObject_write_checksum,       METH_VARARGS,  "write_checksum\n\nCompute and write the checksums into the header."},
    {"verify_checksum",      (PyCFunction)PyFITSObject_verify_checksum,      METH_VARARGS,  "verify_checksum\n\nReturn a dict with dataok and hduok."},

    {"write_image",          (PyCFunction)PyFITSObject_write_image,          METH_VARARGS,  "write_image\n\nWrite the input image to a new extension."},
    {"write_column",         (PyCFunction)PyFITSObject_write_column,         METH_KEYWORDS, "write_column\n\nWrite a column into the specifed hdu."},
    {"write_var_column",         (PyCFunction)PyFITSObject_write_var_column,         METH_KEYWORDS, "write_var_column\n\nWrite a variable length column into the specifed hdu from an object array."},
    {"write_string_key",     (PyCFunction)PyFITSObject_write_string_key,     METH_VARARGS,  "write_string_key\n\nWrite a string key into the specified HDU."},
    {"write_double_key",     (PyCFunction)PyFITSObject_write_double_key,     METH_VARARGS,  "write_double_key\n\nWrite a double key into the specified HDU."},
    {"write_long_key",       (PyCFunction)PyFITSObject_write_long_key,       METH_VARARGS,  "write_long_key\n\nWrite a long key into the specified HDU."},
    {"close",                (PyCFunction)PyFITSObject_close,                METH_VARARGS,  "close\n\nClose the fits file."},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyFITSType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_fitsio.FITS",             /*tp_name*/
    sizeof(struct PyFITSObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyFITSObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyFITSObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "FITSIO Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyFITSObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyFITSObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyFITSObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef fitstype_methods[] = {
    {"cfitsio_version",      (PyCFunction)PyFITS_cfitsio_version,      METH_NOARGS,  "cfitsio_version\n\nReturn the cfitsio version."},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_fitsio_wrap",      /* m_name */
        "Defines the FITS class and some methods",  /* m_doc */
        -1,                  /* m_size */
        fitstype_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif


#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_fitsio_wrap(void) 
{
    PyObject* m;


    PyFITSType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyFITSType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyFITSType) < 0) {
        return;
    }
    m = Py_InitModule3("_fitsio_wrap", fitstype_methods, "Define FITS type and methods.");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyFITSType);
    PyModule_AddObject(m, "FITS", (PyObject *)&PyFITSType);

    import_array();
}
