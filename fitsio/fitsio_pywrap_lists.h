#ifndef _FITSIO_PYWRAP_LISTS_H
#define _FITSIO_PYWRAP_LISTS_H

#include <stdint.h>

struct i64list {
    size_t size;
    int64_t* data;
};

struct i64list* i64list_new(void);
struct i64list* i64list_delete(struct i64list* list);
void i64list_push(struct i64list* list, int64_t val);
struct i64list* i64list_fromarray(PyObject* arrayObj);
void i64list_print(struct i64list* list);


struct stringlist {
    size_t size;
    char** data;
};

struct stringlist* stringlist_new(void);
void stringlist_push(struct stringlist* slist, const char* str);
void stringlist_push_size(struct stringlist* slist, size_t slen);
struct stringlist* stringlist_delete(struct stringlist* slist);
int stringlist_addfrom_listobj(struct stringlist* slist, PyObject* listObj, const char* listname);
void stringlist_print(struct stringlist* slist);




#endif
