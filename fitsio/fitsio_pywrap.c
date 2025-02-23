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
//#include "fitsio_pywrap_lists.h"
#include <numpy/arrayobject.h>

// this is not defined anywhere in cfitsio except in
// the fits file structure
#define CFITSIO_MAX_ARRAY_DIMS 99

// not sure where this is defined in numpy...
#define NUMPY_MAX_DIMS 32

struct PyFITSObject {
    PyObject_HEAD
    fitsfile* fits;
};

#ifdef FITSIO_PYWRAP_ALWAYS_NONSTANDARD_STRINGS
static int fits_use_standard_strings(void)
{
    return 0;
}
#else
#ifndef _FITSIO_H_FITS_USE_STANDARD_STRINGS
#define _FITSIO_H_FITS_USE_STANDARD_STRINGS
int CFITS_API fits_use_standard_strings(void);
#endif
#endif

// check unicode for python3, string for python2
static
int is_python_string(const PyObject* obj)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_Check(obj) || PyBytes_Check(obj);
#else
    return PyUnicode_Check(obj) || PyString_Check(obj);
#endif
}


/*
   Ensure all elements of the null terminated string are ascii, replacing
   non-ascii characters with a ?
*/

static void convert_to_ascii(char* str) {
    size_t size=0, i=0;
    int cval=0;

    size = strlen(str);
    for (i=0; i < size; i++) {
        cval = (int)str[i];
        if (cval < 0 || cval > 127) {
            str[i] = '?';
        }
    }
}

/*
   Replace non ascii characters with _
*/
static int convert_extname_to_ascii(char* str) {
    int was_converted=0;
    size_t size=0, i=0;
    int cval=0;

    size = strlen(str);
    for (i=0; i < size; i++) {
        cval = (int)str[i];

        if (cval < 0 || cval > 127) {
            was_converted = 1;
            str[i] = '_';
        }
    }
    return was_converted;
}


/*
   Replace bad keyword characters (?, *, #) or non-ascii with valid _ characters.
*/
static int convert_keyword_to_allowed_ascii_template_and_nonascii_only(char *str) {
    int isbad=0, was_converted=0;
    size_t size=0, i=0;
    int cval=0;

    size = strlen(str);
    for (i=0; i < size; i++) {
        cval = (int)str[i];

        isbad = (cval == '?') || (cval == '*') || (cval == '#');

        if (isbad || cval < 32 || cval > 126) {
            was_converted = 1;
            str[i] = '_';
        }
    }
    return was_converted;

}

/*
return 1 if a keyword has non-standard FITS keyword characters.
*/
static int has_invalid_keyword_chars(char *str) {
  int isbad=0;
  size_t size=0, i=0;
  int cval=0;

  size = strlen(str);
  for (i=0; i < size; i++) {
      cval = (int)str[i];

      isbad = !(
          (cval >= 'A' && cval <= 'Z')
          ||
          (cval >= 'a' && cval <= 'z')
          ||
          (cval >= '0' && cval <= '9')
          ||
          (cval == '-')
          ||
          (cval == '_')
      );
  }
  return isbad;
}

/*

   get a string version of the object. New memory
   is allocated and the receiver must clean it up.

*/

// unicode is common to python 2 and 3
static char* get_unicode_as_string(PyObject* obj)
{
    PyObject* tmp=NULL;
    char* strdata=NULL;
    tmp = PyObject_CallMethod(obj,"encode",NULL);

    strdata = strdup( PyBytes_AsString(tmp) );
    Py_XDECREF(tmp);

    return strdata;
}

static char* get_object_as_string(PyObject* obj)
{
    PyObject* format=NULL;
    PyObject* args=NULL;
    char* strdata=NULL;
    PyObject* tmpobj1=NULL;

    if (PyUnicode_Check(obj)) {

        strdata=get_unicode_as_string(obj);

    } else {

#if PY_MAJOR_VERSION >= 3

        if (PyBytes_Check(obj)) {
            strdata = strdup( PyBytes_AsString(obj) );
        } else {
            PyObject* tmpobj2=NULL;
            format = Py_BuildValue("s","%s");
            // this is not a string object
            args=PyTuple_New(1);

            PyTuple_SetItem(args,0,obj);
            tmpobj2 = PyUnicode_Format(format, args);
            tmpobj1 = PyObject_CallMethod(tmpobj2,"encode",NULL);

            Py_XDECREF(args);
            Py_XDECREF(tmpobj2);

            strdata = strdup( PyBytes_AsString(tmpobj1) );
            Py_XDECREF(tmpobj1);
            Py_XDECREF(format);
        }

#else
        // convert to a string as needed
        if (PyString_Check(obj)) {
            strdata = strdup( PyString_AsString(obj) );
        } else {
            format = Py_BuildValue("s","%s");
            args=PyTuple_New(1);

            PyTuple_SetItem(args,0,obj);
            tmpobj1= PyString_Format(format, args);

            strdata = strdup( PyString_AsString(tmpobj1) );
            Py_XDECREF(args);
            Py_XDECREF(tmpobj1);
            Py_XDECREF(format);
        }
#endif
    }

    return strdata;
}

static void set_ioerr_string_from_status(int status) {
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

/*
   string list helper functions
*/

struct stringlist {
    size_t size;
    char** data;
};

static struct stringlist* stringlist_new(void) {
    struct stringlist* slist=NULL;

    slist = malloc(sizeof(struct stringlist));
    slist->size = 0;
    slist->data=NULL;
    return slist;
}
// push a copy of the string onto the string list
static void stringlist_push(struct stringlist* slist, const char* str) {
    size_t newsize=0;
    size_t i=0;

    newsize = slist->size+1;
    slist->data = realloc(slist->data, sizeof(char*)*newsize);
    slist->size += 1;

    i = slist->size-1;

    slist->data[i] = strdup(str);
}

static void stringlist_push_size(struct stringlist* slist, size_t slen) {
    size_t newsize=0;
    size_t i=0;

    newsize = slist->size+1;
    slist->data = realloc(slist->data, sizeof(char*)*newsize);
    slist->size += 1;

    i = slist->size-1;

    slist->data[i] = calloc(slen+1,sizeof(char));
    //slist->data[i] = malloc(sizeof(char)*(slen+1));
    //memset(slist->data[i], 0, slen+1);
}
static struct stringlist* stringlist_delete(struct stringlist* slist) {
    if (slist != NULL) {
        size_t i=0;
        if (slist->data != NULL) {
            for (i=0; i < slist->size; i++) {
                free(slist->data[i]);
            }
        }
        free(slist->data);
        free(slist);
    }
    return NULL;
}


/*
static void stringlist_print(struct stringlist* slist) {
    size_t i=0;
    if (slist == NULL) {
        return;
    }
    for (i=0; i<slist->size; i++) {
        printf("  slist[%ld]: %s\n", i, slist->data[i]);
    }
}
*/


static int stringlist_addfrom_listobj(struct stringlist* slist,
                                      PyObject* listObj,
                                      const char* listname) {
    size_t size=0, i=0;
    char* tmpstr=NULL;

    if (!PyList_Check(listObj)) {
        PyErr_Format(PyExc_ValueError, "Expected a list for %s.", listname);
        return 1;
    }
    size = PyList_Size(listObj);

    for (i=0; i<size; i++) {
        PyObject* tmp = PyList_GetItem(listObj, i);
        if (!is_python_string(tmp)) {
            PyErr_Format(PyExc_ValueError,
                         "Expected only strings in %s list.", listname);
            return 1;
        }
        tmpstr = get_object_as_string(tmp);
        stringlist_push(slist, tmpstr);
        free(tmpstr);
    }
    return 0;
}

static
void add_double_to_dict(PyObject* dict, const char* key, double value) {
    PyObject* tobj=NULL;
    tobj=PyFloat_FromDouble(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}

static
void add_long_to_dict(PyObject* dict, const char* key, long value) {
    PyObject* tobj=NULL;
    tobj=PyLong_FromLong(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}

static
void add_long_long_to_dict(PyObject* dict, const char* key, long long value) {
    PyObject* tobj=NULL;
    tobj=PyLong_FromLongLong(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}

static
void add_string_to_dict(PyObject* dict, const char* key, const char* str) {
    PyObject* tobj=NULL;
    tobj=Py_BuildValue("s",str);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}

static
void add_none_to_dict(PyObject* dict, const char* key) {
    PyDict_SetItemString(dict, key, Py_None);
    Py_XINCREF(Py_None);
}
static
void add_true_to_dict(PyObject* dict, const char* key) {
    PyDict_SetItemString(dict, key, Py_True);
    Py_XINCREF(Py_True);
}
static
void add_false_to_dict(PyObject* dict, const char* key) {
    PyDict_SetItemString(dict, key, Py_False);
    Py_XINCREF(Py_False);
}


/*
static
void append_long_to_list(PyObject* list, long value) {
    PyObject* tobj=NULL;
    tobj=PyLong_FromLong(value);
    PyList_Append(list, tobj);
    Py_XDECREF(tobj);
}
*/

static
void append_long_long_to_list(PyObject* list, long long value) {
    PyObject* tobj=NULL;
    tobj=PyLong_FromLongLong(value);
    PyList_Append(list, tobj);
    Py_XDECREF(tobj);
}

/*
static
void append_string_to_list(PyObject* list, const char* str) {
    PyObject* tobj=NULL;
    tobj=Py_BuildValue("s",str);
    PyList_Append(list, tobj);
    Py_XDECREF(tobj);
}
*/



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
        return Py_BuildValue("s",repr);
    }  else {
        return Py_BuildValue("s","none");
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

        fnameObj = Py_BuildValue("s",filename);
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
static npy_int64* get_int64_from_array(PyArrayObject* arr, npy_intp* ncols) {

    npy_int64* colnums;
    int npy_type=0, check=0;

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "int64 array must be an array.");
        return NULL;
    }

    npy_type = PyArray_TYPE(arr);

    // on some platforms, creating an 'i8' array gives it a longlong
    // dtype.  Just make sure it is 8 bytes
    check=
        (npy_type == NPY_INT64)
        |
        (npy_type==NPY_LONGLONG && sizeof(npy_longlong)==sizeof(npy_int64));
	if (!check) {
        PyErr_Format(PyExc_TypeError,
                     "array must be an int64 array (%d), got %d.",
                     NPY_INT64,npy_type);
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
    int hdunum=0, hdutype=0, ext=0, ignore_scaling=FALSE;
    int status=0, tstatus=0, is_compressed=0;
    PyObject* dict=NULL;

    char extname[FLEN_VALUE];
    char hduname[FLEN_VALUE];
    int extver=0, hduver=0;

    long long header_start;
    long long data_start;
    long long data_end;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"ii", &hdunum, &ignore_scaling)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (ignore_scaling == TRUE
        && fits_set_bscale(self->fits, 1.0, 0.0, &status)) {
        return NULL;
    }

    dict = PyDict_New();
    ext=hdunum-1;

    add_long_to_dict(dict, "hdunum", (long)hdunum);
    add_long_to_dict(dict, "extnum", (long)ext);
    add_long_to_dict(dict, "hdutype", (long)hdutype);


    tstatus=0;
    if (fits_read_key(self->fits, TSTRING, "EXTNAME", extname, NULL, &tstatus)==0) {
        convert_extname_to_ascii(extname);
        add_string_to_dict(dict, "extname", extname);
    } else {
        add_string_to_dict(dict, "extname", "");
    }

    tstatus=0;
    if (fits_read_key(self->fits, TSTRING, "HDUNAME", hduname, NULL, &tstatus)==0) {
        convert_extname_to_ascii(hduname);
        add_string_to_dict(dict, "hduname", hduname);
    } else {
        add_string_to_dict(dict, "hduname", "");
    }

    tstatus=0;
    if (fits_read_key(self->fits, TINT, "EXTVER", &extver, NULL, &tstatus)==0) {
        add_long_to_dict(dict, "extver", (long)extver);
    } else {
        add_long_to_dict(dict, "extver", (long)0);
    }

    tstatus=0;
    if (fits_read_key(self->fits, TINT, "HDUVER", &hduver, NULL, &tstatus)==0) {
        add_long_to_dict(dict, "hduver", (long)hduver);
    } else {
        add_long_to_dict(dict, "hduver", (long)0);
    }

    tstatus=0;
    is_compressed=fits_is_compressed_image(self->fits, &tstatus);
    add_long_to_dict(dict, "is_compressed_image", (long)is_compressed);


    // get byte offsets
    if (0==fits_get_hduaddrll(self->fits, &header_start, &data_start, &data_end, &tstatus)) {
        add_long_long_to_dict(dict, "header_start", (long)header_start);
        add_long_long_to_dict(dict, "data_start", (long)data_start);
        add_long_long_to_dict(dict, "data_end", (long)data_end);
    } else {
        add_long_long_to_dict(dict, "header_start", -1);
        add_long_long_to_dict(dict, "data_start", -1);
        add_long_long_to_dict(dict, "data_end", -1);
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
            add_string_to_dict(dict,"error","could not determine image parameters");
        } else {
            add_long_to_dict(dict,"ndims",(long)ndims);
            add_long_to_dict(dict,"img_type",(long)bitpix);

            if (ignore_scaling == TRUE) {
                // Get the raw type if scaling is being ignored.
                fits_get_img_type(self->fits, &bitpix_equiv, &status);
            } else {
                fits_get_img_equivtype(self->fits, &bitpix_equiv, &status);
            }

            add_long_to_dict(dict,"img_equiv_type",(long)bitpix_equiv);

            tstatus=0;
            if (fits_read_key(self->fits, TSTRING, "ZCMPTYPE",
                              comptype, NULL, &tstatus)==0) {
                convert_to_ascii(comptype);
                add_string_to_dict(dict,"comptype",comptype);
            } else {
                add_none_to_dict(dict,"comptype");
            }

            for (i=0; i<ndims; i++) {
                append_long_long_to_list(dimsObj, (long long)dims[i]);
            }
            PyDict_SetItemString(dict, "dims", dimsObj);
            Py_XDECREF(dimsObj);

        }

    } else if (hdutype == BINARY_TBL) {
        int tstatus=0;
        LONGLONG nrows=0;
        int ncols=0;
        PyObject* colinfo = PyList_New(0);
        int i=0,j=0;

        fits_get_num_rowsll(self->fits, &nrows, &tstatus);
        fits_get_num_cols(self->fits, &ncols, &tstatus);
        add_long_long_to_dict(dict,"nrows",(long long)nrows);
        add_long_to_dict(dict,"ncols",(long)ncols);

        {
            PyObject* d = NULL;
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
            fits_read_btblhdrll(self->fits, ncols, NULL, NULL,
                                names->data, tforms->data,
                                NULL, NULL, NULL, &tstatus);

            for (i=0; i<ncols; i++) {
                d = PyDict_New();
                int type=0;
                LONGLONG repeat=0;
                LONGLONG width=0;

                convert_to_ascii(names->data[i]);
                add_string_to_dict(d,"name",names->data[i]);
                convert_to_ascii(tforms->data[i]);
                add_string_to_dict(d,"tform",tforms->data[i]);

                fits_get_coltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                add_long_to_dict(d,"type",(long)type);
                add_long_long_to_dict(d,"repeat",(long long)repeat);
                add_long_long_to_dict(d,"width",(long long)width);

                fits_get_eqcoltypell(self->fits,i+1,&type,&repeat,&width, &tstatus);
                add_long_to_dict(d,"eqtype",(long)type);

                tstatus=0;
                if (fits_read_tdimll(self->fits, i+1, maxdim, &ndims, dims,
                                     &tstatus)) {
                    add_none_to_dict(d,"tdim");
                } else {
                    PyObject* dimsObj=PyList_New(0);
                    for (j=0; j<ndims; j++) {
                        append_long_long_to_list(dimsObj, (long long)dims[j]);
                    }

                    PyDict_SetItemString(d, "tdim", dimsObj);
                    Py_XDECREF(dimsObj);
                }

                // using the struct, could cause problems
                // actually, we can use ffgcprll to get this info, but will
                // be redundant with some others above
                col = &self->fits->Fptr->tableptr[i];
                add_double_to_dict(d,"tscale",col->tscale);
                add_double_to_dict(d,"tzero",col->tzero);

                PyList_Append(colinfo, d);
                Py_XDECREF(d);
            }
            names=stringlist_delete(names);
            tforms=stringlist_delete(tforms);

            PyDict_SetItemString(dict, "colinfo", colinfo);
            Py_XDECREF(colinfo);
        }
    } else {
        int tstatus=0;
        LONGLONG nrows=0;
        int ncols=0;
        PyObject* colinfo = PyList_New(0);
        int i=0,j=0;

        fits_get_num_rowsll(self->fits, &nrows, &tstatus);
        fits_get_num_cols(self->fits, &ncols, &tstatus);
        add_long_long_to_dict(dict,"nrows",(long long)nrows);
        add_long_to_dict(dict,"ncols",(long)ncols);

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

                convert_to_ascii(names->data[i]);
                add_string_to_dict(d,"name",names->data[i]);
                convert_to_ascii(tforms->data[i]);
                add_string_to_dict(d,"tform",tforms->data[i]);

                fits_get_coltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                add_long_to_dict(d,"type",(long)type);
                add_long_long_to_dict(d,"repeat",(long long)repeat);
                add_long_long_to_dict(d,"width",(long long)width);

                fits_get_eqcoltypell(self->fits, i+1, &type, &repeat, &width, &tstatus);
                add_long_to_dict(d,"eqtype",(long)type);

                tstatus=0;
                if (fits_read_tdimll(self->fits, i+1, maxdim, &ndims, dims,
                                                      &tstatus)) {
                    add_none_to_dict(dict,"tdim");
                } else {
                    PyObject* dimsObj=PyList_New(0);
                    for (j=0; j<ndims; j++) {
                        append_long_long_to_list(dimsObj, (long long)dims[j]);
                    }

                    PyDict_SetItemString(d, "tdim", dimsObj);
                    Py_XDECREF(dimsObj);
                }

                // using the struct, could cause problems
                // actually, we can use ffgcprll to get this info, but will
                // be redundant with some others above
                col = &self->fits->Fptr->tableptr[i];
                add_double_to_dict(d,"tscale",col->tscale);
                add_double_to_dict(d,"tzero",col->tzero);

                PyList_Append(colinfo, d);
                Py_XDECREF(d);
            }
            names=stringlist_delete(names);
            tforms=stringlist_delete(tforms);

            PyDict_SetItemString(dict, "colinfo", colinfo);
            Py_XDECREF(colinfo);
        }

    }
    return dict;
}

// get info for the specified HDU
static PyObject *
PyFITSObject_get_hdu_name_version(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0, hdutype=0;
    int status=0;

    char extname[FLEN_VALUE];
    int extver=0;

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

    status=0;
    if (fits_read_key(self->fits, TINT, "EXTVER", &extver, NULL, &status)!=0) {
        extver=0;
    }

    status=0;
    if (fits_read_key(self->fits, TSTRING, "EXTNAME", extname, NULL, &status)==0) {
        return Py_BuildValue("si", extname, extver);
    } else {
        return Py_BuildValue("si", "", extver);
    }
}


// this is the parameter that goes in the type for fits_write_col
static int
npy_to_fits_table_type(int npy_dtype, int write_bitcols) {

    char mess[255];
    switch (npy_dtype) {
        case NPY_BOOL:
            if (write_bitcols) {
                return TBIT;
            } else {
                return TLOGICAL;
            }
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

        case NPY_COMPLEX64:
            return TCOMPLEX;
        case NPY_COMPLEX128:
            return TDBLCOMPLEX;

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
            if (sizeof(LONGLONG) == sizeof(npy_int64)) {
                *fits_img_type = LONGLONG_IMG;
                *fits_datatype = TLONGLONG;
            } else if (sizeof(long) == sizeof(npy_int64)) {
                *fits_img_type = LONG_IMG;
                *fits_datatype = TLONG;
            } else if (sizeof(int) == sizeof(npy_int64)) {
                // there is no UINT_IMG, so use ULONG_IMG
                *fits_img_type = LONG_IMG;
                *fits_datatype = TINT;
            } else if (sizeof(long long) == sizeof(npy_int64)) {
                // we don't expect to get here
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

        case TCOMPLEX:
            return NPY_COMPLEX64;
        case TDBLCOMPLEX:
            return NPY_COMPLEX128;


        case TSTRING:
            return NPY_STRING;

        default:
            PyErr_Format(PyExc_TypeError,"Unsupported FITS table datatype %d", fits_dtype);
            return -9999;
    }

    return 0;
}



static int create_empty_hdu(struct PyFITSObject* self)
{
    int status=0;
    int bitpix=SHORT_IMG;
    int naxis=0;
    long* naxes=NULL;
    if (fits_create_img(self->fits, bitpix, naxis, naxes, &status)) {
        set_ioerr_string_from_status(status);
        return 1;
    }

    return 0;
}


// follows fits convention that return value is true
// for failure
//
// exception strings are set internally
//
// length checking should happen in python
//
// note tile dims are written reverse order since
// python orders C and fits orders Fortran
static int set_compression(fitsfile *fits,
                           int comptype,
                           PyObject* tile_dims_obj,
                           int *status) {

    npy_int64 *tile_dims_py=NULL;
    long *tile_dims_fits=NULL;
    npy_intp ndims=0, i=0;

    // can be NOCOMPRESS (0)
    if (fits_set_compression_type(fits, comptype, status)) {
        set_ioerr_string_from_status(*status);
        goto _set_compression_bail;
        return 1;
    }

    if (tile_dims_obj != Py_None) {

        tile_dims_py=get_int64_from_array((PyArrayObject *) tile_dims_obj, &ndims);
        if (tile_dims_py==NULL) {
            *status=1;
        } else {
            tile_dims_fits = calloc(ndims,sizeof(long));
            if (!tile_dims_fits) {
                PyErr_Format(PyExc_MemoryError, "failed to allocate %ld longs",
                             ndims);
                goto _set_compression_bail;
            }

            for (i=0; i<ndims; i++) {
                tile_dims_fits[ndims-i-1] = tile_dims_py[i];
            }

            fits_set_tile_dim(fits, ndims, tile_dims_fits, status);

            free(tile_dims_fits);tile_dims_fits=NULL;
        }
    }

_set_compression_bail:
    return *status;
}

static int pyarray_get_ndim(PyArrayObject* arr) {
    return arr->nd;
}

/*
   Create an image extension, possible writing data as well.

   We allow creating from dimensions rather than from the input image shape,
   writing into the HDU later

   It is useful to create the extension first so we can write keywords into the
   header before adding data.  This avoids moving the data if the header grows
   too large.

   However, on distributed file systems it can be more efficient to write
   the data at this time due to slowness with updating the file in place.

 */

static PyObject *
PyFITSObject_create_image_hdu(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int ndims=0;
    long *dims=NULL;
    int image_datatype=0; // fits type for image, AKA bitpix
    int datatype=0; // type for the data we entered

    int comptype=0; // same as NOCOMPRESS in newer cfitsio

    PyObject *array_obj=NULL, *dims_obj=NULL, *tile_dims_obj=NULL;
    PyArrayObject *array=NULL, *dims_array=NULL;

    int npy_dtype=0, nkeys=0, write_data=0;
    int i=0;
    int status=0;

    char* extname=NULL;
    int extver=0;
    float qlevel=0;
    int qmethod=0;
    int dither_seed=0;
    float hcomp_scale=0;
    int hcomp_smooth=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    static char *kwlist[] = {
        "array","nkeys",
         "dims",
         "comptype",
         "tile_dims",

         "qlevel",
         "qmethod",
         "dither_seed",

         "hcomp_scale",
         "hcomp_smooth",

         "extname",
         "extver",
         NULL,
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oi|OiOfiifisi", kwlist,
                          &array_obj, &nkeys,
                          &dims_obj,
                          &comptype,
                          &tile_dims_obj,

                          &qlevel,
                          &qmethod,
                          &dither_seed,

                          &hcomp_scale,
                          &hcomp_smooth,

                          &extname,
                          &extver)) {
        goto create_image_hdu_cleanup;
    }


    if (array_obj == Py_None) {
        if (create_empty_hdu(self)) {
            return NULL;
        }
    } else {
        array = (PyArrayObject *) array_obj;
        if (!PyArray_Check(array)) {
            PyErr_SetString(PyExc_TypeError, "input must be an array.");
            goto create_image_hdu_cleanup;
        }

        npy_dtype = PyArray_TYPE(array);
        if (npy_to_fits_image_types(npy_dtype, &image_datatype, &datatype)) {
            goto create_image_hdu_cleanup;
        }

        if (PyArray_Check(dims_obj)) {
            // get dims from input, which must be of type 'i8'
            // this means we are not writing the array that was input,
            // it is only used to determine the data type
            dims_array = (PyArrayObject *) dims_obj;

            npy_int64 *tptr=NULL, tmp=0;
            ndims = PyArray_SIZE(dims_array);
            dims = calloc(ndims,sizeof(long));
            for (i=0; i<ndims; i++) {
                tptr = (npy_int64 *) PyArray_GETPTR1(dims_array, i);
                tmp = *tptr;
                dims[ndims-i-1] = (long) tmp;
            }
            write_data=0;
        } else {
            // we get the dimensions from the array, which means we are going
            // to write it as well
            ndims = pyarray_get_ndim(array);
            dims = calloc(ndims,sizeof(long));
            for (i=0; i<ndims; i++) {
                dims[ndims-i-1] = PyArray_DIM(array, i);
            }
            write_data=1;
        }

        if (comptype == 0) {
            comptype = (*(self->fits)->Fptr).request_compress_type;
        }

        // 0 means NOCOMPRESS but that wasn't defined in the bundled version of cfitsio
        // if (comptype >= 0) {
        if (comptype > 0) {
            // exception strings are set internally
            if (set_compression(self->fits, comptype, tile_dims_obj, &status)) {
                goto create_image_hdu_cleanup;
            }

            if (fits_set_quantize_level(self->fits, qlevel, &status)) {
                goto create_image_hdu_cleanup;
            }

            if (fits_set_quantize_method(self->fits, qmethod, &status)) {
                goto create_image_hdu_cleanup;
            }

            // zero means to use the default (system clock).
            if (dither_seed != 0) {
                if (fits_set_dither_seed(self->fits, dither_seed, &status)) {
                    goto create_image_hdu_cleanup;
                }
            }

            if (comptype == HCOMPRESS_1) {

                if (fits_set_hcomp_scale(self->fits, hcomp_scale, &status)) {
                    goto create_image_hdu_cleanup;
                }
                if (fits_set_hcomp_smooth(self->fits, hcomp_smooth, &status)) {
                    goto create_image_hdu_cleanup;
                }

            }
        }

        if (fits_create_img(self->fits, image_datatype, ndims, dims, &status)) {
            set_ioerr_string_from_status(status);
            goto create_image_hdu_cleanup;
        }


    }
    if (extname != NULL) {
        if (strlen(extname) > 0) {

            // comments are NULL
            if (fits_update_key_str(self->fits, "EXTNAME", extname, NULL, &status)) {
                set_ioerr_string_from_status(status);
                goto create_image_hdu_cleanup;
            }
            if (extver > 0) {
                if (fits_update_key_lng(self->fits, "EXTVER", (LONGLONG) extver, NULL, &status)) {
                    set_ioerr_string_from_status(status);
                    goto create_image_hdu_cleanup;
                }
            }
        }
    }

    if (nkeys > 0) {
        if (fits_set_hdrsize(self->fits, nkeys, &status) ) {
            set_ioerr_string_from_status(status);
            goto create_image_hdu_cleanup;
        }
    }

    if (write_data) {
        int firstpixel=1;
        LONGLONG nelements = 0;
        void* data=NULL;
        nelements = PyArray_SIZE(array);
        data = PyArray_DATA(array);
        if (fits_write_img(self->fits, datatype, firstpixel, nelements, data, &status)) {
            set_ioerr_string_from_status(status);
            goto create_image_hdu_cleanup;
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

    free(dims); dims=NULL;
    Py_RETURN_NONE;
}


// reshape the image to specified dims
// the input array must be of type int64
static PyObject *
PyFITSObject_reshape_image(struct PyFITSObject* self, PyObject* args) {

    int status=0;
    int hdunum=0, hdutype=0;
    PyObject *dims_obj=NULL;
    PyArrayObject *dims_array=NULL;
    LONGLONG dims[CFITSIO_MAX_ARRAY_DIMS]={0};
    LONGLONG dims_orig[CFITSIO_MAX_ARRAY_DIMS]={0};
    int ndims=0, ndims_orig=0;
    npy_int64 dim=0;
    npy_intp i=0;
    int bitpix=0, maxdim=CFITSIO_MAX_ARRAY_DIMS;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"iO", &hdunum, &dims_obj)) {
        return NULL;
    }
    dims_array = (PyArrayObject *) dims_obj;

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // existing image params, just to get bitpix
    if (fits_get_img_paramll(self->fits, maxdim, &bitpix, &ndims_orig, dims_orig, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    ndims = PyArray_SIZE(dims_array);
    for (i=0; i<ndims; i++) {
        dim= *(npy_int64 *) PyArray_GETPTR1(dims_array, i);
        dims[i] = (LONGLONG) dim;
    }

    if (fits_resize_imgll(self->fits, bitpix, ndims, dims, &status)) {
        set_ioerr_string_from_status(status);
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
    PY_LONG_LONG firstpixel_py=0;
    LONGLONG firstpixel=0;
    int image_datatype=0; // fits type for image, AKA bitpix
    int datatype=0; // type for the data we entered

    PyObject* array_obj=NULL;
    PyArrayObject* array=NULL;
    void* data=NULL;
    int npy_dtype=0;
    int status=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"iOL", &hdunum, &array_obj, &firstpixel_py)) {
        return NULL;
    }
    array = (PyArrayObject *) array_obj;

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
    firstpixel = (LONGLONG) firstpixel_py;
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
    int status=0, i=0;
    size_t size=0;
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
    if (size != (size_t)ncols) {
        PyErr_Format(PyExc_ValueError, "Expected %d elements in tdims list, got %ld", ncols, size);
        return 1;
    }

    for (i=0; i<ncols; i++) {
        colnum=i+1;
        tmp = PyList_GetItem(tdimObj, i);
        if (tmp != Py_None) {
            if (!is_python_string(tmp)) {
                PyErr_SetString(PyExc_ValueError, "Expected only strings or None for tdim");
                return 1;
            }

            sprintf(keyname, "TDIM%d", colnum);

            tdim = get_object_as_string(tmp);
            fits_write_key(fits, TSTRING, keyname, tdim, NULL, &status);
            free(tdim);

            if (status) {
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
    int table_type=0, nkeys=0;
    int nfields=0;
    LONGLONG nrows=0; // start empty

    static char *kwlist[] = {
        "table_type","nkeys", "ttyp","tform",
        "tunit", "tdim", "extname", "extver", NULL};
    // these are all strings
    PyObject* ttypObj=NULL;
    PyObject* tformObj=NULL;
    PyObject* tunitObj=NULL;    // optional
    PyObject* tdimObj=NULL;     // optional

    // these must be freed
    struct stringlist* ttyp=NULL;
    struct stringlist* tform=NULL;
    struct stringlist* tunit=NULL;
    //struct stringlist* tdim=stringlist_new();
    char* extname=NULL;
    char* extname_use=NULL;
    int extver=0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiOO|OOsi", kwlist,
                          &table_type, &nkeys, &ttypObj, &tformObj, &tunitObj, &tdimObj, &extname, &extver)) {
        return NULL;
    }

    ttyp=stringlist_new();
    tform=stringlist_new();
    tunit=stringlist_new();
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

            if (fits_update_key_lng(self->fits, "EXTVER", (LONGLONG) extver, NULL, &status)) {
                set_ioerr_string_from_status(status);
                goto create_table_cleanup;
            }
        }
    }

    if (nkeys > 0) {
        if (fits_set_hdrsize(self->fits, nkeys, &status) ) {
            set_ioerr_string_from_status(status);
            goto create_table_cleanup;
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




// create a new table structure.  No physical rows are added yet.
static PyObject *
PyFITSObject_insert_col(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;
    int colnum=0;

    int hdutype=0;

    static char *kwlist[] = {"hdunum","colnum","ttyp","tform","tdim", NULL};
    // these are all strings
    char* ttype=NULL; // field name
    char* tform=NULL; // format
    PyObject* tdimObj=NULL;     // optional, a list of len 1

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiss|O", kwlist,
                          &hdunum, &colnum, &ttype, &tform, &tdimObj)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_insert_col(self->fits, colnum, ttype, tform, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // OK if dims are not sent
    if (tdimObj != NULL && tdimObj != Py_None) {
        PyObject* tmp=NULL;
        char* tdim=NULL;
        char keyname[20];

        sprintf(keyname, "TDIM%d", colnum);
        tmp = PyList_GetItem(tdimObj, 0);

        tdim = get_object_as_string(tmp);
        fits_write_key(self->fits, TSTRING, keyname, tdim, NULL, &status);
        free(tdim);

        if (status) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    }

    // this does a full close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
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
/*
static PyObject *
PyFITSObject_write_column(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    int colnum=0;
    int write_bitcols=0;
    PyObject* array=NULL;

    void* data=NULL;
    PY_LONG_LONG firstrow_py=0;
    LONGLONG firstrow=1;
    LONGLONG firstelem=1;
    LONGLONG nelem=0;
    int npy_dtype=0;
    int fits_dtype=0;

    static char *kwlist[] = {"hdunum","colnum","array","firstrow","write_bitcols", NULL};

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiOLi",
                                     kwlist, &hdunum, &colnum, &array, &firstrow_py, &write_bitcols)) {
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
    fits_dtype = npy_to_fits_table_type(npy_dtype, write_bitcols);
    if (fits_dtype == -9999) {
        return NULL;
    }
    if (fits_dtype == TLOGICAL) {
        int tstatus=0, ttype=0;
        LONGLONG trepeat=0, twidth=0;
        // if the column exists and is declared TBIT we will write
        // that way instead
        if (fits_get_coltypell(self->fits, colnum,
                               &ttype, &trepeat, &twidth, &tstatus)==0) {
            // if we don't get here its because the column doesn't exist
            // yet and that's ok
            if (ttype==TBIT) {
                fits_dtype=TBIT;
            }
        }
    }



    data = PyArray_DATA(array);
    nelem = PyArray_SIZE(array);

    if (fits_dtype == TSTRING) {

        // this is my wrapper for strings
        if (write_string_column(self->fits, colnum, firstrow, firstelem, nelem, data, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    } else if (fits_dtype == TBIT) {
        if (fits_write_col_bit(self->fits, colnum, firstrow, firstelem, nelem, data, &status)) {
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
*/

static PyObject *
PyFITSObject_write_columns(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    int write_bitcols=0;
    //void **data_ptrs=NULL;
    PyObject* colnum_list=NULL;
    PyObject* array_list=NULL;
    PyObject *tmp_obj=NULL;
    PyArrayObject *tmp_array=NULL;

    Py_ssize_t ncols=0;

    void* data=NULL;
    PY_LONG_LONG firstrow_py=0;
    LONGLONG firstrow=1, thisrow=0;
    LONGLONG firstelem=1;
    LONGLONG nelem=0;
    LONGLONG *nperrow=NULL;
    int npy_dtype=0;
    int *fits_dtypes=NULL;
    int *is_string=NULL, *colnums=NULL;
    void **array_ptrs=NULL;

    npy_intp ndim=0, *dims=NULL;
    Py_ssize_t irow=0, icol=0, j=0;;

    static char *kwlist[] = {"hdunum","colnums","arraylist","firstrow","write_bitcols", NULL};

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iOOLi",
                                     kwlist, &hdunum, &colnum_list, &array_list, &firstrow_py, &write_bitcols)) {
        return NULL;
    }
    firstrow = (LONGLONG) firstrow_py;

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    if (!PyList_Check(colnum_list)) {
        PyErr_SetString(PyExc_ValueError,"colnums must be a list");
        return NULL;
    }
    if (!PyList_Check(array_list)) {
        PyErr_SetString(PyExc_ValueError,"colnums must be a list");
        return NULL;
    }
    ncols = PyList_Size(colnum_list);
    if (ncols == 0) {
        goto _fitsio_pywrap_write_columns_bail;
    }
    if (ncols != PyList_Size(array_list)) {
        PyErr_Format(PyExc_ValueError,"colnum and array lists not same size: %ld/%ld",
                     ncols, PyList_Size(array_list));
    }

    // from here on we'll have some temporary arrays we have to free
    is_string = calloc(ncols, sizeof(int));
    colnums = calloc(ncols, sizeof(int));
    array_ptrs = calloc(ncols, sizeof(void*));
    nperrow = calloc(ncols, sizeof(LONGLONG));
    fits_dtypes = calloc(ncols, sizeof(int));

    for (icol=0; icol<ncols; icol++) {

        tmp_obj = PyList_GetItem(colnum_list,icol);
#if PY_MAJOR_VERSION >= 3
        colnums[icol] = 1+(int) PyLong_AsLong(tmp_obj);
#else
        colnums[icol] = 1+(int) PyInt_AsLong(tmp_obj);
#endif

        tmp_array = (PyArrayObject *) PyList_GetItem(array_list, icol);
        npy_dtype = PyArray_TYPE(tmp_array);

        fits_dtypes[icol] = npy_to_fits_table_type(npy_dtype, write_bitcols);
        if (fits_dtypes[icol] == -9999) {
            status=1;
            goto _fitsio_pywrap_write_columns_bail;
        }
        if (fits_dtypes[icol] == TLOGICAL) {
            int tstatus=0, ttype=0;
            LONGLONG trepeat=0, twidth=0;
            // if the column exists and is declared TBIT we will write
            // that way instead
            if (fits_get_coltypell(self->fits, colnums[icol],
                                   &ttype, &trepeat, &twidth, &tstatus)==0) {
                // if we don't get here its because the column doesn't exist
                // yet and that's ok
                if (ttype==TBIT) {
                    fits_dtypes[icol]=TBIT;
                }
            }
        }

        if (fits_dtypes[icol]==TSTRING) {
            is_string[icol] = 1;
        }
        ndim = PyArray_NDIM(tmp_array);
        dims = PyArray_DIMS(tmp_array);
        if (icol==0) {
            nelem = dims[0];
        } else {
            if (dims[0] != nelem) {
                PyErr_Format(PyExc_ValueError,
                        "not all entries have same row count, "
                        "%lld/%ld", nelem,dims[0]);
                status=1;
                goto _fitsio_pywrap_write_columns_bail;
            }
        }

        array_ptrs[icol] = tmp_array;

        nperrow[icol] = 1;
        for (j=1; j<ndim; j++) {
            nperrow[icol] *= dims[j];
        }
    }

    for (irow=0; irow<nelem; irow++) {
        thisrow = firstrow + irow;
        for (icol=0; icol<ncols; icol++) {
            data=PyArray_GETPTR1(array_ptrs[icol], irow);
            if (is_string[icol]) {
                if (write_string_column(self->fits,
                                        colnums[icol],
                                        thisrow,
                                        firstelem,
                                        nperrow[icol],
                                        (char*)data,
                                        &status)) {
                    set_ioerr_string_from_status(status);
                    goto _fitsio_pywrap_write_columns_bail;
                }

            } else if (fits_dtypes[icol] == TBIT) {
                if (fits_write_col_bit(self->fits,
                                       colnums[icol],
                                       thisrow,
                                       firstelem,
                                       nperrow[icol],
                                       data,
                                       &status)) {
                    set_ioerr_string_from_status(status);
                    goto _fitsio_pywrap_write_columns_bail;
                }
            } else {
                if( fits_write_col(self->fits,
                                   fits_dtypes[icol],
                                   colnums[icol],
                                   thisrow,
                                   firstelem,
                                   nperrow[icol],
                                   data,
                                   &status)) {
                    set_ioerr_string_from_status(status);
                    goto _fitsio_pywrap_write_columns_bail;
                }
            }
        }
    }
    /*
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
    */

_fitsio_pywrap_write_columns_bail:
    free(is_string); is_string=NULL;
    free(colnums); colnums=NULL;
    free(array_ptrs); array_ptrs=NULL;
    free(nperrow); nperrow=NULL;
    free(fits_dtypes); fits_dtypes=NULL;
    if (status != 0) {
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
        PyArrayObject* array,
        int  *status) {   /* IO - error status                           */

    LONGLONG firstelem=1; // ignored
    LONGLONG nelem=1; // ignored
    npy_intp nrows=0;
    npy_intp i=0;
    char* ptr=NULL;
    int res=0;

    PyObject* el=NULL;
    char* strdata=NULL;
    char* strarr[1];


    nrows = PyArray_SIZE(array);
    for (i=0; i<nrows; i++) {
        ptr = PyArray_GetPtr(array, &i);
        el = PyArray_GETITEM(array, ptr);

        strdata=get_object_as_string(el);

        // just a container
        strarr[0] = strdata;
        res=fits_write_col_str(fits, colnum,
                               firstrow+i, firstelem, nelem,
                               strarr, status);

        free(strdata);
        if(res > 0) {
            goto write_var_string_column_cleanup;
        }
    }

write_var_string_column_cleanup:

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
        PyArrayObject* array,
        int  *status) {   /* IO - error status                           */

    LONGLONG firstelem=1;
    npy_intp nelem=0;
    npy_intp nrows=0;
    npy_intp i=0;
    PyObject* el=NULL;
    PyArrayObject* el_array=NULL;
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
        el_array = (PyArrayObject *) PyArray_CheckFromAny(
            el, PyArray_DescrFromType(npy_dtype),
            mindepth, maxdepth,
            requirements, context
        );
        if (el_array == NULL) {
            // error message will already be set
            return 1;
        }

        nelem = PyArray_SIZE(el_array);
        data = PyArray_DATA(el_array);
        res = fits_write_col(fits, abs(fits_dtype), colnum,
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
    PyObject* array_obj=NULL;
    PyArrayObject* array=NULL;

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
                  kwlist, &hdunum, &colnum, &array_obj, &firstrow_py)) {
        return NULL;
    }
    firstrow = (LONGLONG) firstrow_py;

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    if (!PyArray_Check(array_obj)) {
        PyErr_SetString(PyExc_ValueError,"only arrays can be written to columns");
        return NULL;
    }
    array = (PyArrayObject *) array_obj;

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


/*
    case for writing an entire record
*/
static PyObject *
PyFITSObject_write_record(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* cardin=NULL;
    char card[FLEN_CARD];

    if (!PyArg_ParseTuple(args, (char*)"is", &hdunum, &cardin)) {
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
    // card not null terminated, so we copy everything.  GCC will
    // warn about this
    strncpy(card, cardin, FLEN_CARD);

    if (fits_write_record(self->fits, card, &status)) {
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

    if (fits_update_key_longstr(self->fits, keyname, value, comment, &status)) {
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

    if (fits_update_key_dbl(self->fits, keyname, value, decimals, comment, &status)) {
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
PyFITSObject_write_long_long_key(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* keyname=NULL;
    long long value=0;
    char* comment=NULL;
    char* comment_in=NULL;

    if (!PyArg_ParseTuple(args, (char*)"isLs", &hdunum, &keyname, &value, &comment_in)) {
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

    if (fits_update_key_lng(self->fits, keyname, (LONGLONG) value, comment, &status)) {
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
PyFITSObject_write_logical_key(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* keyname=NULL;
    int value=0;
    char* comment=NULL;
    char* comment_in=NULL;

    if (!PyArg_ParseTuple(args, (char*)"isis", &hdunum, &keyname, &value, &comment_in)) {
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

    if (fits_update_key_log(self->fits, keyname, value, comment, &status)) {
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

// let python do the conversions
static PyObject *
PyFITSObject_write_comment(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* comment=NULL;

    if (!PyArg_ParseTuple(args, (char*)"is", &hdunum, &comment)) {
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

    if (fits_write_comment(self->fits, comment, &status)) {
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

// let python do the conversions
static PyObject *
PyFITSObject_write_history(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* history=NULL;

    if (!PyArg_ParseTuple(args, (char*)"is", &hdunum, &history)) {
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

    if (fits_write_history(self->fits, history, &status)) {
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

//   ADW: Adapted from ffpcom and ffphis in putkey.c
static
int fits_write_continue( fitsfile *fptr,      /* I - FITS file pointer  */
                         const char *cont,    /* I - continue string    */
                         int   *status)       /* IO - error status      */
/*
  Write 1 or more CONTINUE keywords.  If the history string is too
  long to fit on a single keyword (72 chars) then it will automatically
  be continued on multiple CONTINUE keywords.
*/
{
    char card[FLEN_CARD];
    int len, ii;

    if (*status > 0)           /* inherit input status value if > 0 */
        return(*status);

    len = strlen(cont);
    ii = 0;

    for (; len > 0; len -= 72)
    {
        strcpy(card, "CONTINUE");
        strncat(card, &cont[ii], 72);
        ffprec(fptr, card, status);
        ii += 72;
    }

    return(*status);
}

// let python do the conversions
static PyObject *
PyFITSObject_write_continue(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* value=NULL;

    if (!PyArg_ParseTuple(args, (char*)"is", &hdunum, &value)) {
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

    if (fits_write_continue(self->fits, value, &status)) {
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
PyFITSObject_write_undefined_key(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;

    char* keyname=NULL;
    char* comment=NULL;
    char* comment_in=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iss", &hdunum, &keyname, &comment_in)) {
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

    if (fits_update_key_null(self->fits, keyname, comment, &status)) {
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
   insert a set of rows
*/

static PyObject *
PyFITSObject_insert_rows(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;

    int hdutype=0;
    PY_LONG_LONG firstrow_py=0, nrows_py=0;
    LONGLONG firstrow=0, nrows=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"iLL",
                          &hdunum, &firstrow_py, &nrows_py)) {
        return NULL;
    }

    firstrow = (LONGLONG) firstrow_py;
    nrows = (LONGLONG) nrows_py;

    if (nrows <= 0) {
        // nothing to do, just return
        Py_RETURN_NONE;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_insert_rows(self->fits, firstrow, nrows, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this does a full close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    Py_RETURN_NONE;
}


/*

   delete a range of rows

   input stop is like a python slice, so exclusive, but 1-offset
   rather than 0-offset
*/

static PyObject *
PyFITSObject_delete_row_range(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;

    int hdutype=0;
    PY_LONG_LONG slice_start_py=0, slice_stop_py=0;
    LONGLONG slice_start=0, slice_stop=0, nrows=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"iLL",
                          &hdunum, &slice_start_py, &slice_stop_py)) {
        return NULL;
    }

    slice_start = (LONGLONG) slice_start_py;
    slice_stop = (LONGLONG) slice_stop_py;
    nrows = slice_stop - slice_start;

    if (nrows <= 0) {
        // nothing to do, just return
        Py_RETURN_NONE;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_delete_rows(self->fits, slice_start, nrows, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this does a full close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    Py_RETURN_NONE;
}

/*

   delete a specific set of rows, 1-offset

   no type checking is applied to the rows
*/

static PyObject *
PyFITSObject_delete_rows(struct PyFITSObject* self, PyObject* args, PyObject* kwds) {
    int status=0;
    int hdunum=0;

    int hdutype=0;
    PyObject *rows_obj=NULL;
    PyArrayObject *rows_array=NULL;
    LONGLONG *rows=NULL, nrows=0;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_ValueError, "fits file is NULL");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, (char*)"iO",
                          &hdunum, &rows_obj)) {
        return NULL;
    }
    rows_array = (PyArrayObject *) rows_obj;

    rows = (LONGLONG *) PyArray_DATA(rows_array);
    nrows = PyArray_SIZE(rows_array);
    if (nrows <= 0) {
        Py_RETURN_NONE;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    if (fits_delete_rowlistll(self->fits, rows, nrows, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    // this does a full close and reopen
    if (fits_flush_file(self->fits, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    Py_RETURN_NONE;
}




/*
 * read a single, entire column from an ascii table into the input array.  This
 * version uses the standard read column instead of our by-bytes version.
 *
 * A number of assumptions are made, such as that columns are scalar, which
 * is true for ascii.
 */

static int read_ascii_column_all(fitsfile* fits, int colnum, PyArrayObject* array, int* status) {

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

    npy_dtype = PyArray_TYPE(array);
    fits_dtype = npy_to_fits_table_type(npy_dtype,0);

    nelem = PyArray_SIZE(array);

    if (fits_dtype == TSTRING) {
        npy_intp i=0;
        LONGLONG rownum=0;

        for (i=0; i<nelem; i++) {
            cdata = PyArray_GETPTR1(array, i);
            rownum = (LONGLONG) (1+i);
            if (fits_read_col_str(fits,colnum,rownum,firstelem,1,nulstr,&cdata,anynul,status) > 0) {
                return 1;
            }
        }

        /*

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
            //strdata[i] = &cdata[twidth*i];
            // this 1-d assumption works because array fields are not allowedin ascii
            strdata[i] = (char*) PyArray_GETPTR1(array, i);
        }

        if (fits_read_col_str(fits,colnum,firstrow,firstelem,nelem,nulstr,strdata,anynul,status) > 0) {
            free(strdata);
            return 1;
        }

        free(strdata);
        */

    } else {
        data=PyArray_DATA(array);
        if (fits_read_col(fits,fits_dtype,colnum,firstrow,firstelem,nelem,nulval,data,anynul,status) > 0) {
            return 1;
        }
    }

    return 0;

}
static int read_ascii_column_byrow(
        fitsfile* fits,
        int colnum,
        PyArrayObject* array,
        PyArrayObject* rows,
        PyArrayObject* sortind,
        int* status
)
{

    int npy_dtype=0;
    int fits_dtype=0;

    npy_intp nelem=0;
    LONGLONG firstelem=1;
    LONGLONG rownum=0;
    npy_int64 si=0;
    npy_intp nrows=-1;

    int* anynul=NULL;
    void* nulval=0;
    char* nulstr=" ";
    void* data=NULL;
    char* cdata=NULL;

    int dorows=0;

    npy_intp i=0;

    npy_dtype = PyArray_TYPE(array);
    fits_dtype = npy_to_fits_table_type(npy_dtype,0);

    nelem = PyArray_SIZE(array);


    if ((PyObject *) rows != Py_None) {
        dorows=1;
        nrows = PyArray_SIZE(rows);
        if (nrows != nelem) {
            PyErr_Format(PyExc_ValueError,
                    "input array[%ld] and rows[%ld] have different size", nelem,nrows);
            return 1;
        }
    }

    for (i=0; i<nrows; i++) {
        if (dorows) {
            si = *(npy_int64*) PyArray_GETPTR1(sortind, i);
            rownum = (LONGLONG)( *(npy_int64*) PyArray_GETPTR1(rows, si) );
            rownum += 1;
        } else {
            rownum = (LONGLONG) (1+i);
        }
        // assuming 1-D
        data = PyArray_GETPTR1(array, si);
        if (fits_dtype == TSTRING) {
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


static int read_ascii_column(
    fitsfile* fits,
    int colnum,
    PyArrayObject* array,
    PyArrayObject* rows,
    PyArrayObject* sortind,
    int* status
)
{

    int ret=0;
    if ((PyObject *) rows != Py_None || !PyArray_ISCONTIGUOUS(array)) {
        ret = read_ascii_column_byrow(
            fits, colnum, array, rows, sortind, status
        );
    } else {
        ret = read_ascii_column_all(fits, colnum, array, status);
    }

    return ret;
}





// read a subset of rows for the input column
// the row array is assumed to be unique and sorted.
static int read_binary_column(
        fitsfile* fits,
        int colnum,
        npy_intp nrows,
        npy_int64* rows,
        npy_int64* sortind,
        void* vdata,
        npy_intp stride,
        int* status) {

    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0, irow=0;
    npy_int64 row=0, si=0;

    LONGLONG repeat=0;
    LONGLONG width=0;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    char *data=NULL, *ptr=NULL;

    data = (char *) vdata;

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    colptr = hdu->tableptr + (colnum-1);

    repeat = colptr->trepeat;
    width = colptr->tdatatype == TSTRING ? 1 : colptr->twidth;

    for (irow=0; irow<nrows; irow++) {
        if (rows != NULL) {
            si = sortind[irow];
            row = rows[si];
        } else {
            row = irow;
        }

        ptr = data + si * stride;

        file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;
        ffmbyt(fits, file_pos, REPORT_EOF, status);
        if (ffgbytoff(fits, width, repeat, 0, (void*)ptr, status)) {
            return 1;
        }
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

    PyObject* array_obj=NULL, *rows_obj=NULL, *sortind_obj=NULL;
    PyArrayObject *array=NULL, *rows_array=NULL, *sortind_array=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iiOOO",
            &hdunum, &colnum, &array_obj, &rows_obj, &sortind_obj)) {
        return NULL;
    }

    array = (PyArrayObject *) array_obj;
    rows_array = (PyArrayObject *) rows_obj;
    sortind_array = (PyArrayObject *) sortind_obj;

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


    if (hdutype == ASCII_TBL) {
        if (read_ascii_column(self->fits, colnum, array, rows_array, sortind_array, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    } else {
        void* data=PyArray_DATA(array);
        npy_intp nrows=0, nsortind=0;
        npy_int64* rows=NULL, *sortind=NULL;
        npy_intp stride=PyArray_STRIDE(array,0);
        if (rows_obj == Py_None) {
            nrows = hdu->numrows;
        } else {
            rows = get_int64_from_array(rows_array, &nrows);
            sortind = get_int64_from_array(sortind_array, &nsortind);
        }

        if (read_binary_column(self->fits, colnum, nrows, rows, sortind, data, stride, &status)) {
            set_ioerr_string_from_status(status);
            return NULL;
        }
    }
    Py_RETURN_NONE;
}




/*
 * Free all the elements in the python list as well as the list itself
 */
static void free_all_python_list(PyObject* list) {
    if (PyList_Check(list)) {
        Py_ssize_t i=0;
        for (i=0; i<PyList_Size(list); i++) {
            Py_XDECREF(PyList_GetItem(list,i));
        }
    }
    Py_XDECREF(list);
}

static PyObject*
read_var_string(fitsfile* fits, int colnum, LONGLONG row, LONGLONG nchar, int* status) {
    LONGLONG firstelem=1;
    char* str=NULL;
    char* strarr[1];
    PyObject* stringObj=NULL;
    void* nulval=0;
    int* anynul=NULL;

    str=calloc(nchar+1,sizeof(char));
    if (str == NULL) {
        PyErr_Format(PyExc_MemoryError,
                     "Could not allocate string of size %lld", nchar);
        return NULL;
    }

    strarr[0] = str;
    if (fits_read_col(fits,TSTRING,colnum,row,firstelem,nchar,nulval,strarr,anynul,status) > 0) {
        goto read_var_string_cleanup;
    }
#if PY_MAJOR_VERSION >= 3
    // bytes
    stringObj = Py_BuildValue("y",str);
#else
    stringObj = Py_BuildValue("s",str);
#endif
    if (NULL == stringObj) {
        PyErr_Format(PyExc_MemoryError,
                     "Could not allocate py string of size %lld", nchar);
        goto read_var_string_cleanup;
    }

read_var_string_cleanup:
    free(str);

    return stringObj;
}
static PyObject*
read_var_nums(fitsfile* fits, int colnum, LONGLONG row, LONGLONG nelem,
              int fits_dtype, int npy_dtype, int* status) {
    LONGLONG firstelem=1;
    PyArrayObject* array=NULL;
    void* nulval=0;
    int* anynul=NULL;
    npy_intp dims[1];
    int fortran=0;
    void* data=NULL;


    dims[0] = nelem;
    array = (PyArrayObject *) PyArray_ZEROS(1, dims, npy_dtype, fortran);
    if (array == NULL) {
        PyErr_Format(PyExc_MemoryError,
                     "Could not allocate array type %d size %lld",npy_dtype,nelem);
        return NULL;
    }
    data = PyArray_DATA(array);
    if (fits_read_col(fits,abs(fits_dtype),colnum,row,firstelem,nelem,nulval,data,anynul,status) > 0) {
        Py_XDECREF(array);
        return NULL;
    }

    return (PyObject *) array;
}
/*
 * read a variable length column as a list of arrays
 * what about strings?
 */
static PyObject *
PyFITSObject_read_var_column_as_list(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int colnum=0;
    PyObject* rows_obj=NULL, *sortind_obj=NULL;
    PyArrayObject* rows_array=NULL, *sortind_array=NULL;

    int hdutype=0;
    int ncols=0;
    const npy_int64* rows=NULL, *sortind=NULL;
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
    npy_int64 si = 0;

    PyObject* listObj=NULL;
    PyObject* tempObj=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iiOO", &hdunum, &colnum, &rows_obj, &sortind_obj)) {
        return NULL;
    }
    rows_array = (PyArrayObject *) rows_obj;
    sortind_array = (PyArrayObject *) sortind_obj;

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

    if (rows_obj == Py_None) {
        fits_get_num_rowsll(self->fits, &nrows, &tstatus);
        get_all_rows=1;
    } else {
        npy_intp tnrows=0, nsortind=0;
        rows = (const npy_int64*) get_int64_from_array(rows_array, &tnrows);
        sortind = (const npy_int64*) get_int64_from_array(sortind_array, &nsortind);
        nrows = (LONGLONG) tnrows;
        get_all_rows = 0;
    }

    listObj = PyList_New(0);

    for (i=0; i<nrows; i++) {
        tempObj=NULL;

        if (get_all_rows) {
            row = i+1;
        } else {
            si = sortind[i];
            row = (LONGLONG) (rows[si]+1);
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
        Py_XDECREF(tempObj);
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


// read specified columns and rows
static int read_binary_rec_columns(
        fitsfile* fits,
        npy_intp ncols,
        npy_int64* colnums,
        npy_intp nrows,
        npy_int64* rows,
        npy_int64* sortind,
        PyArrayObject *array,
        int* status
) {
    FITSfile* hdu=NULL;
    tcolumn* colptr=NULL;
    LONGLONG file_pos=0;
    npy_intp col=0;
    npy_int64 colnum=0;
    char* ptr=NULL;

    npy_intp irow=0;
    npy_int64 row=0, si=0;

    LONGLONG gsize=0; // number of bytes in column
    LONGLONG repeat=0;
    LONGLONG width=0;

    // using struct defs here, could cause problems
    hdu = fits->Fptr;

    for (irow=0; irow < nrows; irow++) {
        if (rows != NULL) {
            si = sortind[irow];
            row = rows[si];
        } else {
            si = irow;
            row = irow;
        }

        ptr = (char *) PyArray_GETPTR1(array, si);
        for (col=0; col < ncols; col++) {

            colnum = colnums[col];
            colptr = hdu->tableptr + (colnum-1);

            repeat = colptr->trepeat;
            width = colptr->tdatatype == TSTRING ? 1 : colptr->twidth;

            file_pos = hdu->datastart + row*hdu->rowlength + colptr->tbcol;

            if (colptr->tdatatype == TBIT) {
                if (fits_read_col_bit(fits, colnum, row+1, 1, repeat, ptr, status)) {
                    return 1;
                }
            } else {
                // can just do one status check, since status are inherited.
                ffmbyt(fits, file_pos, REPORT_EOF, status);
                if (ffgbytoff(fits, width, repeat, 0, (void*)ptr, status)) {
                    return 1;
                }
            }

            gsize = repeat * width;
            ptr += gsize;

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
    FITSfile* hdu=NULL;

    int status=0;

    PyObject *columns_obj=NULL, *array_obj=NULL, *rows_obj=NULL, *sortind_obj=NULL;

    npy_intp nrows, nsortind;
    npy_int64* rows=NULL, *sortind=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iOOOO",
                          &hdunum, &columns_obj, &array_obj, &rows_obj, &sortind_obj)) {
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

    colnums = get_int64_from_array((PyArrayObject *) columns_obj, &ncols);
    if (colnums == NULL) {
        return NULL;
    }

    hdu = self->fits->Fptr;

    if (rows_obj == Py_None) {
        nrows = hdu->numrows;
    } else {
        rows = get_int64_from_array((PyArrayObject *) rows_obj, &nrows);
        if (rows == NULL) {
            return NULL;
        }
        sortind = get_int64_from_array((PyArrayObject *) sortind_obj, &nsortind);
        if (sortind == NULL) {
            return NULL;
        }
    }
    if (read_binary_rec_columns(
            self->fits, ncols, colnums,
            nrows, rows, sortind, (PyArrayObject *) array_obj, &status)) {
        goto recread_columns_cleanup;
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
        const npy_int64* sortind,
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
    npy_int64 row=0, si=0;

    long groupsize=0; // number of bytes in column
    long ngroups=1; // number to read, one for row-by-row reading
    long group_gap=0; // gap between groups, zero since we aren't using it

    if (rows != NULL) {
        get_all_rows=0;
    }

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    for (irow=0; irow < nrows; irow++) {
        if (get_all_rows) {
            row = irow;
            si = irow;
        } else {
            si = sortind[irow];
            row = rows[si];
        }
        for (col=0; col < ncols; col++) {

            // point to this field in the array, allows for skipping
            ptr = data + si*recsize + field_offsets[col];

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
    npy_intp nrows=0, nsortind=0;
    const npy_int64* colnums=NULL;
    const npy_int64* offsets=NULL;
    const npy_int64* rows=NULL, *sortind=NULL;

    PyObject* columns_obj=NULL;
    PyObject* offsets_obj=NULL;
    PyObject* rows_obj=NULL;
    PyObject* sortind_obj=NULL;

    PyObject* array_obj=NULL;
    PyArrayObject *array=NULL;
    void* data=NULL;
    npy_intp recsize=0;

    if (!PyArg_ParseTuple(args, (char*)"iOOOOO",
            &hdunum, &columns_obj, &offsets_obj, &array_obj, &rows_obj, &sortind_obj)) {
        return NULL;
    }

    array = (PyArrayObject *) array_obj;

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

    colnums = (const npy_int64*) get_int64_from_array((PyArrayObject *) columns_obj, &ncols);
    if (colnums == NULL) {
        return NULL;
    }
    offsets = (const npy_int64*) get_int64_from_array((PyArrayObject *) offsets_obj, &noffsets);
    if (offsets == NULL) {
        return NULL;
    }
    if (noffsets != ncols) {
        PyErr_Format(PyExc_ValueError,
                     "%ld columns requested but got %ld offsets",
                     ncols, noffsets);
        return NULL;
    }

    if (rows_obj != Py_None) {
        rows = (const npy_int64*) get_int64_from_array((PyArrayObject *) rows_obj, &nrows);
        sortind = (const npy_int64*) get_int64_from_array((PyArrayObject *) sortind_obj, &nsortind);
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
                sortind,
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
        npy_intp nrows, npy_int64* rows, npy_int64* sortind,
        void* vdata, int* status) {

    FITSfile* hdu=NULL;

    npy_intp irow=0, si=0;
    LONGLONG firstrow=1;
    LONGLONG firstchar=1;

    // use char for pointer arith.  It's actually ok to use void as char but
    // this is just in case.
    unsigned char* ptr, *data;

    // using struct defs here, could cause problems
    hdu = fits->Fptr;
    // ptr = (unsigned char*) data;
    data = (unsigned char*) vdata;

    for (irow=0; irow < nrows; irow++) {

        si = sortind[irow];

        // Input is zero-offset
        firstrow = 1 + (LONGLONG) rows[si];

        ptr = data + si * hdu->rowlength;

        if (fits_read_tblbytes(fits, firstrow, firstchar, hdu->rowlength, ptr, status)) {
            return 1;
        }

        // ptr += hdu->rowlength;
    }

    return 0;
}
// read specified rows, all columns
/*
static int read_rec_bytes_byrowold(
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
*/


// python method to read all columns but subset of rows
static PyObject *
PyFITSObject_read_rows_as_rec(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;

    int status=0;
    void* data=NULL;

    PyObject *array_obj=NULL, *rows_obj=NULL, *sortind_obj=NULL;
    npy_intp nrows=0, nsortind=0;
    npy_int64* rows=NULL;
    npy_int64* sortind=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iOOO", &hdunum, &array_obj, &rows_obj, &sortind_obj)) {
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

    data = PyArray_DATA((PyArrayObject *) array_obj);

    rows = get_int64_from_array((PyArrayObject *) rows_obj, &nrows);
    if (rows == NULL) {
        return NULL;
    }
    sortind = get_int64_from_array((PyArrayObject *) sortind_obj, &nsortind);
    if (sortind == NULL) {
        return NULL;
    }

    if (read_rec_bytes_byrow(self->fits, nrows, rows, sortind, data, &status)) {
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




/* here rows are 1-offset, unlike when reading a specific subset of rows */
static PyObject *
PyFITSObject_read_as_rec(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;

    int status=0;
    PyObject* array_obj=NULL;
    void* data=NULL;

    PY_LONG_LONG firstrow=0;
    PY_LONG_LONG lastrow=0;
    PY_LONG_LONG nrows=0;

    if (!PyArg_ParseTuple(args, (char*)"iLLO", &hdunum, &firstrow, &lastrow, &array_obj)) {
        return NULL;
    }

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        goto recread_asrec_cleanup;
    }

    if (hdutype == IMAGE_HDU) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot read IMAGE_HDU into a recarray");
        return NULL;
    }

    data = PyArray_DATA((PyArrayObject *) array_obj);

    nrows=lastrow-firstrow+1;
    if (read_rec_range(self->fits, (LONGLONG)firstrow, (LONGLONG)nrows, data, &status)) {
        goto recread_asrec_cleanup;
    }

recread_asrec_cleanup:

    if (status != 0) {
        set_ioerr_string_from_status(status);
        return NULL;
    }
    Py_RETURN_NONE;
}


// read an n-dimensional "image" into the input array.  Only minimal checking
// of the input array is done.
// Note numpy allows a maximum of 32 dimensions
static PyObject *
PyFITSObject_read_image(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;
    int status=0;
    PyObject* array_obj=NULL;
    PyArrayObject* array=NULL;
    void* data=NULL;
    int npy_dtype=0;
    int dummy=0, fits_read_dtype=0;

    int maxdim=NUMPY_MAX_DIMS; // numpy maximum
    int datatype=0; // type info for axis
    int naxis=0; // number of axes
    int i=0;
    LONGLONG naxes[NUMPY_MAX_DIMS];;  // size of each axis
    LONGLONG firstpixels[NUMPY_MAX_DIMS];
    LONGLONG size=0;
    npy_intp arrsize=0;

    int anynul=0;

    if (!PyArg_ParseTuple(args, (char*)"iO", &hdunum, &array_obj)) {
        return NULL;
    }

    array = (PyArrayObject *) array_obj;

    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        return NULL;
    }

    if (fits_get_img_paramll(self->fits, maxdim, &datatype, &naxis,
                             naxes, &status)) {
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
        PyErr_Format(PyExc_RuntimeError,
          "Input array size is %ld but on disk array size is %lld",
          arrsize, size);
        return NULL;
    }

    npy_dtype = PyArray_TYPE(array);
    npy_to_fits_image_types(npy_dtype, &dummy, &fits_read_dtype);

    for (i=0; i<naxis; i++) {
        firstpixels[i] = 1;
    }
    if (fits_read_pixll(self->fits, fits_read_dtype, firstpixels, size,
                        0, data, &anynul, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }


    Py_RETURN_NONE;
}

static PyObject *
PyFITSObject_read_raw(struct PyFITSObject* self, PyObject* args) {
    if (self->fits == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "FITS file is NULL");
        return NULL;
    }
    //fitsfile* fits = self->fits;
    FITSfile* FITS = self->fits->Fptr;
    int status = 0;
    char* filedata;
    LONGLONG sz;
    LONGLONG io_pos;
    PyObject *stringobj;

    // Flush (close & reopen HDU) to make everything consistent
    ffflus(self->fits, &status);
    if (status) {
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to flush FITS file data to disk; CFITSIO code %i",
                     status);
        return NULL;
    }
    // Allocate buffer for string
    sz = FITS->filesize;
    // Create python string object of requested size, unitialized
    stringobj = PyBytes_FromStringAndSize(NULL, sz);
    if (!stringobj) {
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to allocate python string object to hold FITS file data: %i bytes",
                     (int)sz);
        return NULL;
    }
    // Grab pointer to the memory buffer of the python string object
    filedata = PyBytes_AsString(stringobj);
    if (!filedata) {
        Py_DECREF(stringobj);
        return NULL;
    }
    // Remember old file position
    io_pos = FITS->io_pos;
    // Seek to beginning of file
    if (ffseek(FITS, 0)) {
        Py_DECREF(stringobj);
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to seek to beginning of FITS file");
        return NULL;
    }
    // Read into filedata
    if (ffread(FITS, sz, filedata, &status)) {
        Py_DECREF(stringobj);
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to read file data into memory: CFITSIO code %i",
                     status);
        return NULL;
    }
    // Seek back to where we were
    if (ffseek(FITS, io_pos)) {
        Py_DECREF(stringobj);
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to seek back to original FITS file position");
        return NULL;
    }
    return stringobj;
}

static int get_long_slices(PyArrayObject* fpix_arr,
                           PyArrayObject* lpix_arr,
                           PyArrayObject* step_arr,
                           long** fpix,
                           long** lpix,
                           long** step) {

    int i=0;
    npy_int64* ptr=NULL;
    npy_intp fsize=0, lsize=0, ssize=0;

    fsize=PyArray_SIZE(fpix_arr);
    lsize=PyArray_SIZE(lpix_arr);
    ssize=PyArray_SIZE(step_arr);

    if (lsize != fsize || ssize != fsize) {
        PyErr_SetString(PyExc_RuntimeError,
                        "start/end/step must be same len");
        return 1;
    }

    *fpix=calloc(fsize, sizeof(long));
    *lpix=calloc(fsize, sizeof(long));
    *step=calloc(fsize, sizeof(long));

    for (i=0;i<fsize;i++) {
        ptr=PyArray_GETPTR1(fpix_arr, i);
        (*fpix)[i] = *ptr;
        ptr=PyArray_GETPTR1(lpix_arr, i);
        (*lpix)[i] = *ptr;
        ptr=PyArray_GETPTR1(step_arr, i);
        (*step)[i] = *ptr;
    }
    return 0;
}

// read an n-dimensional "image" into the input array.  Only minimal checking
// of the input array is done.
static PyObject *
PyFITSObject_read_image_slice(struct PyFITSObject* self, PyObject* args) {
    int hdunum=0;
    int hdutype=0;
    int status=0;
    PyObject* fpix_obj=NULL;
    PyObject* lpix_obj=NULL;
    PyObject* step_obj=NULL;
    int ignore_scaling=FALSE;
    PyObject* array=NULL;
    long* fpix=NULL;
    long* lpix=NULL;
    long* step=NULL;
    void* data=NULL;
    int npy_dtype=0;
    int dummy=0, fits_read_dtype=0;

    int anynul=0;

    if (!PyArg_ParseTuple(args, (char*)"iOOOiO",
                &hdunum, &fpix_obj, &lpix_obj, &step_obj, &ignore_scaling,
                &array)) {
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        return NULL;
    }

    if (ignore_scaling == TRUE
        && fits_set_bscale(self->fits, 1.0, 0.0, &status)) {
        return NULL;
    }

    if (get_long_slices(
          (PyArrayObject *) fpix_obj,
          (PyArrayObject *) lpix_obj,
          (PyArrayObject *) step_obj,
          &fpix,&lpix,&step)) {
        return NULL;
    }
    data = PyArray_DATA((PyArrayObject *) array);

    npy_dtype = PyArray_TYPE((PyArrayObject *) array);
    npy_to_fits_image_types(npy_dtype, &dummy, &fits_read_dtype);

    if (fits_read_subset(self->fits, fits_read_dtype, fpix, lpix, step,
                         0, data, &anynul, &status)) {
        set_ioerr_string_from_status(status);
        goto read_image_slice_cleanup;
    }

read_image_slice_cleanup:
    free(fpix);
    free(lpix);
    free(step);

    if (status != 0) {
        return NULL;
    }

    Py_RETURN_NONE;
}



static int hierarch_is_string(const char* card)
{
    int i=0, is_string_value=1;

    for (i=0; i<78; i++) {
        if (card[i] == '=') {
            // we found the equals, now if it is a string we
            // now exactly where the quote must be
            if (card[i+2] == '\'') {
                is_string_value = 1;
            } else  {
                is_string_value = 0;
            }
        }
    }
    return is_string_value;
}

// read the entire header as list of dicts with name,value,comment and full
// card
static PyObject *
PyFITSObject_read_header(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    int lcont=0, lcomm=0, ls=0;
    int tocomp=0;
    int is_comment_or_history=0, is_blank_key=0;
    char *longstr=NULL;

    char keyname[FLEN_KEYWORD];
    char value[FLEN_VALUE];
    char comment[FLEN_COMMENT];
    char scomment[FLEN_COMMENT];
    char card[FLEN_CARD];
    long is_string_value=0;

    LONGLONG lval=0;
    double dval=0;

    int nkeys=0, morekeys=0, i=0;
    int has_equals=0, has_quote=0, was_converted=0, is_hierarch=0;

    PyObject* list=NULL;
    PyObject* dict=NULL;  // to hold the dict for each record

    lcont=strlen("CONTINUE");
    lcomm=strlen("COMMENT");

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

    list=PyList_New(0);
    for (i=0; i<nkeys; i++) {

        // the full card
        if (fits_read_record(self->fits, i+1, card, &status)) {
            Py_XDECREF(list);
            set_ioerr_string_from_status(status);
            return NULL;
        }

        // this just returns the character string stored in the header; we
        // can eval in python
        if (fits_read_keyn(self->fits, i+1, keyname, value, scomment, &status)) {
            Py_XDECREF(list);
            set_ioerr_string_from_status(status);
            return NULL;
        }

        ls = strlen(keyname);
        tocomp = (ls < lcont) ? ls : lcont;

        is_blank_key = 0;
        is_hierarch = 0;
        if (ls == 0) {
            is_blank_key = 1;
        } else {

            // skip CONTINUE, we already read the data
            if (strncmp(keyname,"CONTINUE",tocomp)==0) {
                continue;
            }

            if (strncmp(keyname, "COMMENT", tocomp) ==0
                    || strncmp(keyname, "HISTORY", tocomp )==0) {
                is_comment_or_history = 1;

            } else {
                is_comment_or_history = 0;

                if (fits_read_key_longstr(self->fits, keyname, &longstr, comment, &status)) {
                    Py_XDECREF(list);
                    set_ioerr_string_from_status(status);
                    return NULL;
                }

                if (strncmp(card,"HIERARCH",8)==0) {
                    is_hierarch = 1;
                    if (hierarch_is_string(card)) {
                        is_string_value=1;
                    } else {
                        is_string_value=0;
                    }
                } else {
                    int j;
                    has_equals = (card[8] == '=') ? 1 : 0;
                    has_quote = 0;
                    // Look for 0 or more space characters followed by a quote character.
                    for (j=10; j<FLEN_CARD; j++) {
                        if (card[j] == '\'') {
                            has_quote = 1;
                            break;
                        } else if (card[j] != ' ') {
                            break;
                        }
                    }
                    if (has_equals && has_quote) {
                        is_string_value=1;
                    } else {
                        is_string_value=0;
                    }
                }
            }
        }

        dict = PyDict_New();
        convert_to_ascii(card);
        add_string_to_dict(dict, "card_string", card);

        if (is_blank_key) {
            add_none_to_dict(dict, "name");
            add_string_to_dict(dict, "value", "");
            convert_to_ascii(scomment);
            add_string_to_dict(dict, "comment", scomment);

        } else if (is_comment_or_history) {
            // comment or history
            convert_to_ascii(scomment);
            add_string_to_dict(dict, "name", keyname);
            add_string_to_dict(dict, "value", scomment);
            add_string_to_dict(dict, "comment", scomment);

        } else {

            if (is_hierarch) {
              // if a key is hierarch, then any ascii character is allowed except
              // *, ? and #. Thus we convert any of those (and any chars we find
              // that don't correspond to something written, ascii <= 32 or 127)
              // to an underscore
              // if the key is converted, then we cannot parse it further with
              // cfitsio
              was_converted = convert_keyword_to_allowed_ascii_template_and_nonascii_only(keyname);
            } else {
              // for non-hierach keys, we cannot use the cfitsio functions if any
              // character besides those in the fits conventions (A-Z,a-z,0-9,_,-)
              // are present. Thus we flag those and store their values as a string
              // if this happens.
              was_converted = has_invalid_keyword_chars(keyname);

              // in order to actually store the key in the python dict, we have to cut
              // out any non-ascii chars
              // we additionally convert the template chars to that the fits data
              // we make can be written back without error
              // note that the check by has_invalid_keyword_chars is more stringent
              // than the checks done here, so if any conversion is done it has already
              // been flagged above.
              convert_keyword_to_allowed_ascii_template_and_nonascii_only(keyname);
            }
            add_string_to_dict(dict,"name",keyname);
            convert_to_ascii(comment);
            add_string_to_dict(dict,"comment",comment);

            // if not a comment but empty value, put in None
            tocomp = (ls < lcomm) ? ls : lcomm;
            // if (!is_string_value && 0==strlen(longstr) && !is_comment) {
            if (!is_string_value && 0==strlen(longstr)) {

                add_none_to_dict(dict, "value");

            } else {

                // if it's a string we just store it.
                if (is_string_value) {
                    convert_to_ascii(longstr);
                    add_string_to_dict(dict,"value",longstr);
                } else if ( longstr[0]=='T' ) {
                    add_true_to_dict(dict, "value");
                } else if (longstr[0]=='F') {
                    add_false_to_dict(dict, "value");
                } else if (was_converted) {
                    // if we had to convert bad characters in the keyword name
                    // we can't attempt to get a numerical value using
                    // fits_read_key because some characters in a keyword name
                    // cause a seg fault
                    convert_to_ascii(longstr);
                    add_string_to_dict(dict,"value",longstr);
                } else if (
                           (strchr(longstr,'.') != NULL)
                           || (strchr(longstr,'E') != NULL)
                           || (strchr(longstr,'e') != NULL) ) {
                    // we found a floating point value
                    fits_read_key(self->fits, TDOUBLE, keyname, &dval, comment, &status);
                    add_double_to_dict(dict,"value",dval);
                } else {

                    // we might have found an integer
                    if (fits_read_key(self->fits,
                                      TLONGLONG,
                                      keyname,
                                      &lval,
                                      comment,
                                      &status)) {

                        // something non standard, just store it as a string
                        convert_to_ascii(longstr);
                        add_string_to_dict(dict,"value",longstr);
                        status=0;

                    } else {
                        add_long_long_to_dict(dict,"value",(long long)lval);
                    }
                }

            }
        }

        free(longstr); longstr=NULL;

        PyList_Append(list, dict);
        Py_XDECREF(dict);

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
    add_long_long_to_dict(dict,"datasum",(long long)datasum);
    add_long_long_to_dict(dict,"hdusum",(long long)hdusum);

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
    add_long_to_dict(dict,"dataok",(long)dataok);
    add_long_to_dict(dict,"hduok",(long)hduok);

    return dict;
}



static PyObject *
PyFITSObject_where(struct PyFITSObject* self, PyObject* args) {
    int status=0;
    int hdunum=0;
    int hdutype=0;
    char* expression=NULL;

    long firstrow;
    long nrows;
    long ngood=0;
    char* row_status=NULL;


    // Indices of rows for which expression is true
    PyObject* indices_obj=NULL;
    int ndim=1;
    npy_intp dims[1];
    npy_intp* data=NULL;
    long i=0;


    if (!PyArg_ParseTuple(args, (char*)"isll", &hdunum, &expression,
                          &firstrow, &nrows)) {
        return NULL;
    }

    if (firstrow < 1 || nrows < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "firstrow and nrows must be positive integers");
        return NULL;
    }

    if (fits_movabs_hdu(self->fits, hdunum, &hdutype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    row_status = malloc(nrows*sizeof(char));
    if (row_status==NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate row_status array");
        return NULL;
    }

    if (fits_find_rows(self->fits, expression, firstrow, nrows, &ngood, row_status, &status)) {
        set_ioerr_string_from_status(status);
        goto where_function_cleanup;
    }

    dims[0] = ngood;
    indices_obj = PyArray_EMPTY(ndim, dims, NPY_INTP, 0);
    if (indices_obj == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate index array");
        goto where_function_cleanup;
    }

    if (ngood > 0) {
        data = PyArray_DATA((PyArrayObject *) indices_obj);

        for (i=0; i<nrows; i++) {
            if (row_status[i]) {
                *data = (npy_intp) i;
                data++;
            }
        }
    }
where_function_cleanup:
    free(row_status);
    return indices_obj;
}

// generic functions, not tied to an object

static PyObject *
PyFITS_cfitsio_version(void) {
    float version=0;
    fits_get_version(&version);
    return PyFloat_FromDouble((double)version);
}

static PyObject *
PyFITS_cfitsio_use_standard_strings(void) {
    if ( fits_use_standard_strings() ) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}


/*

'C',              'L',     'I',     'F'             'X'
character string, logical, integer, floating point, complex

*/

static PyObject *
PyFITS_get_keytype(PyObject* self, PyObject* args) {

    int status=0;
    char* card=NULL;
    char dtype[2]={0};

    if (!PyArg_ParseTuple(args, (char*)"s", &card)) {
        return NULL;
    }


    if (fits_get_keytype(card, dtype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    } else {
        return Py_BuildValue("s", dtype);
    }
}
static PyObject *
PyFITS_get_key_meta(PyObject* self, PyObject* args) {

    int status=0;
    char* card=NULL;
    char dtype[2]={0};
    int keyclass=0;

    if (!PyArg_ParseTuple(args, (char*)"s", &card)) {
        return NULL;
    }


    keyclass=fits_get_keyclass(card);

    if (fits_get_keytype(card, dtype, &status)) {
        set_ioerr_string_from_status(status);
        return NULL;
    }

    return Py_BuildValue("is", keyclass, dtype);

}

/*

    note the special first four comment fields will not be called comment but
    structural!  That will cause an exception to be raised, so the card should
    be checked before calling this function

*/

static PyObject *
PyFITS_parse_card(PyObject* self, PyObject* args) {

    int status=0;
    char name[FLEN_VALUE]={0};
    char value[FLEN_VALUE]={0};
    char comment[FLEN_COMMENT]={0};
    int keylen=0;
    int keyclass=0;
    int is_undefined=0;

    char* card=NULL;
    char dtype[2]={0};
    PyObject* output=NULL;

    if (!PyArg_ParseTuple(args, (char*)"s", &card)) {
        goto bail;
    }

    keyclass=fits_get_keyclass(card);

    // only proceed if not comment or history, but note the special first four
    // comment fields will not be called comment but structural!  That will
    // cause an exception to be raised, so the card should be checked before
    // calling this function

    if (keyclass != TYP_COMM_KEY && keyclass != TYP_CONT_KEY) {

        if (fits_get_keyname(card, name, &keylen, &status)) {
            set_ioerr_string_from_status(status);
            goto bail;
        }
        if (fits_parse_value(card, value, comment, &status)) {
            set_ioerr_string_from_status(status);
            goto bail;
        }
        if (fits_get_keytype(value, dtype, &status)) {

            if (status == VALUE_UNDEFINED) {
                is_undefined=1;
                status=0;
            } else {
                set_ioerr_string_from_status(status);
                goto bail;
            }
        }
    }

bail:
    if (status != 0) {
        return NULL;
    }

    if (is_undefined) {
        output=Py_BuildValue("isss", keyclass, name, dtype, comment);
    } else {
        output=Py_BuildValue("issss", keyclass, name, value, dtype, comment);
    }
    return output;
}



static PyMethodDef PyFITSObject_methods[] = {
    {"filename",             (PyCFunction)PyFITSObject_filename,             METH_VARARGS,  "filename\n\nReturn the name of the file."},

    {"where",                (PyCFunction)PyFITSObject_where,                METH_VARARGS,  "where\n\nReturn an index array where the input expression evaluates to true."},

    {"movabs_hdu",           (PyCFunction)PyFITSObject_movabs_hdu,           METH_VARARGS,  "movabs_hdu\n\nMove to the specified HDU."},
    {"movnam_hdu",           (PyCFunction)PyFITSObject_movnam_hdu,           METH_VARARGS,  "movnam_hdu\n\nMove to the specified HDU by name and return the hdu number."},

    {"get_hdu_name_version",         (PyCFunction)PyFITSObject_get_hdu_name_version,         METH_VARARGS,  "get_hdu_name_version\n\nReturn a tuple (extname,extvers)."},
    {"get_hdu_info",         (PyCFunction)PyFITSObject_get_hdu_info,         METH_VARARGS,  "get_hdu_info\n\nReturn a dict with info about the specified HDU."},
    {"read_raw",             (PyCFunction)PyFITSObject_read_raw,             METH_NOARGS,  "read_raw\n\nRead the entire raw contents of the FITS file, returning a python string."},
    {"read_image",           (PyCFunction)PyFITSObject_read_image,           METH_VARARGS,  "read_image\n\nRead the entire n-dimensional image array.  No checking of array is done."},
    {"read_image_slice",     (PyCFunction)PyFITSObject_read_image_slice,     METH_VARARGS,  "read_image_slice\n\nRead an image slice."},
    {"read_column",          (PyCFunction)PyFITSObject_read_column,          METH_VARARGS,  "read_column\n\nRead the column into the input array.  No checking of array is done."},
    {"read_var_column_as_list",          (PyCFunction)PyFITSObject_read_var_column_as_list,          METH_VARARGS,  "read_var_column_as_list\n\nRead the variable length column as a list of arrays."},
    {"read_columns_as_rec",  (PyCFunction)PyFITSObject_read_columns_as_rec,  METH_VARARGS,  "read_columns_as_rec\n\nRead the specified columns into the input rec array.  No checking of array is done."},
    {"read_columns_as_rec_byoffset",  (PyCFunction)PyFITSObject_read_columns_as_rec_byoffset,  METH_VARARGS,  "read_columns_as_rec_byoffset\n\nRead the specified columns into the input rec array at the specified offsets.  No checking of array is done."},
    {"read_rows_as_rec",     (PyCFunction)PyFITSObject_read_rows_as_rec,     METH_VARARGS,  "read_rows_as_rec\n\nRead the subset of rows into the input rec array.  No checking of array is done."},
    {"read_as_rec",          (PyCFunction)PyFITSObject_read_as_rec,          METH_VARARGS,  "read_as_rec\n\nRead a set of rows into the input rec array.  No significant checking of array is done."},
    {"read_header",          (PyCFunction)PyFITSObject_read_header,          METH_VARARGS | METH_VARARGS,  "read_header\n\nRead the entire header as a list of dictionaries."},

    {"create_image_hdu",     (PyCFunction)PyFITSObject_create_image_hdu,     METH_VARARGS | METH_KEYWORDS, "create_image_hdu\n\nWrite the input image to a new extension."},
    {"create_table_hdu",     (PyCFunction)PyFITSObject_create_table_hdu,     METH_VARARGS | METH_KEYWORDS, "create_table_hdu\n\nCreate a new table with the input parameters."},
    {"insert_col",           (PyCFunction)PyFITSObject_insert_col,           METH_VARARGS | METH_KEYWORDS, "insert_col\n\nInsert a new column."},

    {"write_checksum",       (PyCFunction)PyFITSObject_write_checksum,       METH_VARARGS,  "write_checksum\n\nCompute and write the checksums into the header."},
    {"verify_checksum",      (PyCFunction)PyFITSObject_verify_checksum,      METH_VARARGS,  "verify_checksum\n\nReturn a dict with dataok and hduok."},

    {"reshape_image",          (PyCFunction)PyFITSObject_reshape_image,          METH_VARARGS,  "reshape_image\n\nReshape the image."},
    {"write_image",          (PyCFunction)PyFITSObject_write_image,          METH_VARARGS,  "write_image\n\nWrite the input image to a new extension."},
    //{"write_column",         (PyCFunction)PyFITSObject_write_column,         METH_VARARGS | METH_KEYWORDS, "write_column\n\nWrite a column into the specified hdu."},
    {"write_columns",        (PyCFunction)PyFITSObject_write_columns,        METH_VARARGS | METH_KEYWORDS, "write_columns\n\nWrite columns into the specified hdu."},
    {"write_var_column",     (PyCFunction)PyFITSObject_write_var_column,     METH_VARARGS | METH_KEYWORDS, "write_var_column\n\nWrite a variable length column into the specified hdu from an object array."},
    {"write_record",         (PyCFunction)PyFITSObject_write_record,     METH_VARARGS,  "write_record\n\nWrite a header card."},
    {"write_string_key",     (PyCFunction)PyFITSObject_write_string_key,     METH_VARARGS,  "write_string_key\n\nWrite a string key into the specified HDU."},
    {"write_double_key",     (PyCFunction)PyFITSObject_write_double_key,     METH_VARARGS,  "write_double_key\n\nWrite a double key into the specified HDU."},

    {"write_long_long_key",       (PyCFunction)PyFITSObject_write_long_long_key,       METH_VARARGS,  "write_long_long_key\n\nWrite a long long key into the specified HDU."},
    {"write_logical_key",    (PyCFunction)PyFITSObject_write_logical_key,    METH_VARARGS,  "write_logical_key\n\nWrite a logical key into the specified HDU."},

    {"write_comment",        (PyCFunction)PyFITSObject_write_comment,        METH_VARARGS,  "write_comment\n\nWrite a comment into the header of the specified HDU."},
    {"write_history",        (PyCFunction)PyFITSObject_write_history,        METH_VARARGS,  "write_history\n\nWrite history into the header of the specified HDU."},
    {"write_continue",       (PyCFunction)PyFITSObject_write_continue,        METH_VARARGS,  "write_continue\n\nWrite contineu into the header of the specified HDU."},

    {"write_undefined_key",       (PyCFunction)PyFITSObject_write_undefined_key,        METH_VARARGS,  "write_undefined_key\n\nWrite a key without a value field into the header of the specified HDU."},

    {"insert_rows",        (PyCFunction)PyFITSObject_insert_rows,        METH_VARARGS,  "Insert blank rows"},

    {"delete_row_range",        (PyCFunction)PyFITSObject_delete_row_range,        METH_VARARGS,  "Delete a range of rows"},
    {"delete_rows",        (PyCFunction)PyFITSObject_delete_rows,        METH_VARARGS,  "Delete a set of rows"},

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
    {"cfitsio_use_standard_strings",      (PyCFunction)PyFITS_cfitsio_use_standard_strings,      METH_NOARGS,  "cfitsio_use_standard_strings\n\nReturn True if using string code that matches the FITS standard."},
    {"parse_card",      (PyCFunction)PyFITS_parse_card,      METH_VARARGS,  "parse_card\n\nparse the card to get the key name, value (as a string), data type and comment."},
    {"get_keytype",      (PyCFunction)PyFITS_get_keytype,      METH_VARARGS,  "get_keytype\n\nparse the card to get the key type."},
    {"get_key_meta",      (PyCFunction)PyFITS_get_key_meta,      METH_VARARGS,  "get_key_meta\n\nparse the card to get key metadata (keyclass,dtype)."},
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
#if PY_MAJOR_VERSION >= 3
PyInit__fitsio_wrap(void)
#else
init_fitsio_wrap(void)
#endif
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
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
