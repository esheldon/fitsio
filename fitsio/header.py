"""
header classes for fitslib, part of the fitsio package.

See the main docs at https://github.com/esheldon/fitsio

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

"""
from __future__ import with_statement, print_function
import warnings

from . import _fitsio_wrap
from .util import isstring, FITSRuntimeWarning, IS_PY3

# for python3 compat
if IS_PY3:
    xrange = range

TYP_STRUC_KEY = 10
TYP_CMPRS_KEY = 20
TYP_SCAL_KEY = 30
TYP_NULL_KEY = 40
TYP_DIM_KEY = 50
TYP_RANG_KEY = 60
TYP_UNIT_KEY = 70
TYP_DISP_KEY = 80
TYP_HDUID_KEY = 90
TYP_CKSUM_KEY = 100
TYP_WCS_KEY = 110
TYP_REFSYS_KEY = 120
TYP_COMM_KEY = 130
TYP_CONT_KEY = 140
TYP_USER_KEY = 150


class FITSHDR(object):
    """
    A class representing a FITS header.

    parameters
    ----------
    record_list: optional
        A list of dicts, or dict, or another FITSHDR
          - list of dictionaries containing 'name','value' and optionally
            a 'comment' field; the order is preserved.
          - a dictionary of keyword-value pairs; no comments are written
            in this case, and the order is arbitrary.
          - another FITSHDR object; the order is preserved.

    examples:

        hdr=FITSHDR()

        # set a simple value
        hdr['blah'] = 35

        # set from a dict to include a comment.
        rec={'name':'fromdict', 'value':3, 'comment':'my comment'}
        hdr.add_record(rec)

        # can do the same with a full FITSRecord
        rec=FITSRecord( {'name':'temp', 'value':35, 'comment':'temp in C'} )
        hdr.add_record(rec)

        # in the above, the record is replaced if one with the same name
        # exists, except for COMMENT and HISTORY, which can exist as
        # duplicates

        # print the header
        print(hdr)

        # print a single record
        print(hdr['fromdict'])


        # can also set from a card
        hdr.add_record('test    =                   77')
        # using a FITSRecord object (internally uses FITSCard)
        card=FITSRecord('test    =                   77')
        hdr.add_record(card)

        # can also construct with a record list
        recs=[{'name':'test', 'value':35, 'comment':'a comment'},
              {'name':'blah', 'value':'some string'}]
        hdr=FITSHDR(recs)

        # if you have no comments, you can construct with a simple dict
        recs={'day':'saturday',
              'telescope':'blanco'}
        hdr=FITSHDR(recs)

    """
    def __init__(self, record_list=None):

        self._record_list = []
        self._record_map = {}
        self._index_map = {}

        if isinstance(record_list, FITSHDR):
            for r in record_list.records():
                self.add_record(r)
        elif isinstance(record_list, dict):
            for k in record_list:
                r = {'name': k, 'value': record_list[k]}
                self.add_record(r)
        elif isinstance(record_list, list):
            for r in record_list:
                self.add_record(r)
        elif record_list is not None:
            raise ValueError("expected a dict or list of dicts or FITSHDR")

    def add_record(self, record_in):
        """
        Add a new record.  Strip quotes from around strings.

        This will over-write if the key already exists, except
        for COMMENT and HISTORY fields

        parameters
        -----------
        record:
            The record, either a dict or a header card string
            or a FITSRecord or FITSCard
        """
        if (isinstance(record_in, dict) and
                'name' in record_in and 'value' in record_in):
            record = {}
            record.update(record_in)
        else:
            record = FITSRecord(record_in)

        # only append when this name already exists if it is
        # a comment or history field, otherwise simply over-write
        key = record['name']
        if key is not None:
            key = key.upper()

        key_exists = key in self._record_map

        if not key_exists or key in ('COMMENT', 'HISTORY', 'CONTINUE', None):
            # append new record
            self._record_list.append(record)
            index = len(self._record_list)-1
            self._index_map[key] = index
        else:
            # over-write existing
            index = self._index_map[key]
            self._record_list[index] = record

        self._record_map[key] = record

    def _add_to_map(self, record):
        key = record['name'].upper()
        self._record_map[key] = record

    def get_comment(self, item):
        """
        Get the comment for the requested entry
        """
        key = item.upper()
        if key not in self._record_map:
            raise KeyError("unknown record: %s" % key)

        if 'comment' not in self._record_map[key]:
            return None
        else:
            return self._record_map[key]['comment']

    def records(self):
        """
        Return the list of full records as a list of dictionaries.
        """
        return self._record_list

    def keys(self):
        """
        Return a copy of the current key list.
        """
        return [e['name'] for e in self._record_list]

    def delete(self, name):
        """
        Delete the specified entry if it exists.
        """
        if isinstance(name, (list, tuple)):
            for xx in name:
                self.delete(xx)
        else:
            if name in self._record_map:
                del self._record_map[name]
                self._record_list = [
                    r for r in self._record_list if r['name'] != name]

    def clean(self, is_table=False):
        """
        Remove reserved keywords from the header.

        These are keywords that the fits writer must write in order
        to maintain consistency between header and data.

        keywords
        --------
        is_table: bool, optional
            Set True if this is a table, so extra keywords will be cleaned
        """

        rmnames = [
            'SIMPLE', 'EXTEND', 'XTENSION', 'BITPIX', 'PCOUNT', 'GCOUNT',
            'THEAP',
            'EXTNAME',
            # 'BLANK',
            'ZQUANTIZ', 'ZDITHER0', 'ZIMAGE', 'ZCMPTYPE',
            'ZSIMPLE', 'ZTENSION', 'ZPCOUNT', 'ZGCOUNT',
            'ZBITPIX', 'ZEXTEND',
            # 'FZTILELN','FZALGOR',
            'CHECKSUM', 'DATASUM']

        if is_table:
            # these are not allowed in tables
            rmnames += [
                'BUNIT', 'BSCALE', 'BZERO',
            ]

        self.delete(rmnames)

        r = self._record_map.get('NAXIS', None)
        if r is not None:
            naxis = int(r['value'])
            self.delete('NAXIS')

            rmnames = ['NAXIS%d' % i for i in xrange(1, naxis+1)]
            self.delete(rmnames)

        r = self._record_map.get('ZNAXIS', None)
        self.delete('ZNAXIS')
        if r is not None:

            znaxis = int(r['value'])

            rmnames = ['ZNAXIS%d' % i for i in xrange(1, znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZTILE%d' % i for i in xrange(1, znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZNAME%d' % i for i in xrange(1, znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZVAL%d' % i for i in xrange(1, znaxis+1)]
            self.delete(rmnames)

        r = self._record_map.get('TFIELDS', None)
        if r is not None:
            tfields = int(r['value'])
            self.delete('TFIELDS')

            if tfields > 0:

                nbase = [
                    'TFORM', 'TTYPE', 'TDIM', 'TUNIT', 'TSCAL', 'TZERO',
                    'TNULL', 'TDISP', 'TDMIN', 'TDMAX', 'TDESC', 'TROTA',
                    'TRPIX', 'TRVAL', 'TDELT', 'TCUNI',
                    # 'FZALG'
                ]
                for i in xrange(1, tfields+1):
                    names = ['%s%d' % (n, i) for n in nbase]
                    self.delete(names)

    def get(self, item, default_value=None):
        """
        Get the requested header entry by keyword name
        """

        found, name = self._contains_and_name(item)
        if found:
            return self._record_map[name]['value']
        else:
            return default_value

    def __len__(self):
        return len(self._record_list)

    def __contains__(self, item):
        found, _ = self._contains_and_name(item)
        return found

    def _contains_and_name(self, item):

        if isinstance(item, FITSRecord):
            name = item['name']
        elif isinstance(item, dict):
            name = item.get('name', None)
            if name is None:
                raise ValueError("dict record must have 'name' field")
        else:
            name = item

        found = False
        if name is None:
            if None in self._record_map:
                found = True
        else:
            name = name.upper()
            if name in self._record_map:
                found = True
            elif name[0:8] == 'HIERARCH':
                if len(name) > 9:
                    name = name[9:]
                    if name in self._record_map:
                        found = True

        return found, name

    def __setitem__(self, item, value):
        if isinstance(value, (dict, FITSRecord)):
            if item.upper() != value['name'].upper():
                raise ValueError("when setting using a FITSRecord, the "
                                 "name field must match")
            rec = value
        else:
            rec = {'name': item, 'value': value}

        try:
            # the entry may already exist; if so, preserve the comment
            comment = self.get_comment(item)
            rec['comment'] = comment
        except KeyError:
            pass

        self.add_record(rec)

    def __getitem__(self, item):
        if item not in self:
            raise KeyError("unknown record: %s" % item)

        return self.get(item)

    def __iter__(self):
        self._current = 0
        return self

    def next(self):
        """
        for iteration over the header entries
        """
        if self._current < len(self._record_list):
            rec = self._record_list[self._current]
            key = rec['name']
            self._current += 1
            return key
        else:
            raise StopIteration
    __next__ = next

    def _record2card(self, record):
        """
        when we add new records they don't have a card,
        this sort of fakes it up similar to what cfitsio
        does, just for display purposes.  e.g.

            DBL     =            23.299843
            LNG     =              3423432
            KEYSNC  = 'hello   '
            KEYSC   = 'hello   '           / a comment for string
            KEYDC   =     3.14159265358979 / a comment for pi
            KEYLC   =            323423432 / a comment for long

        basically,
            - 8 chars, left aligned, for the keyword name
            - a space
            - 20 chars for value, left aligned for strings, right aligned for
              numbers
            - if there is a comment, one space followed by / then another space
              then the comment out to 80 chars

        """
        name = record['name']
        value = record['value']
        comment = record.get('comment', '')

        v_isstring = isstring(value)

        if name is None:
            card = '         %s' % comment
        elif name == 'COMMENT':
            card = 'COMMENT %s' % comment
        elif name == 'CONTINUE':
            card = 'CONTINUE   %s' % value
        elif name == 'HISTORY':
            card = 'HISTORY   %s' % value
        else:
            if len(name) > 8:
                card = 'HIERARCH %s= ' % name
            else:
                card = '%-8s= ' % name[0:8]

            # these may be string representations of data, or actual strings
            if v_isstring:
                value = str(value)
                if len(value) > 0:
                    if value[0] != "'":
                        # this is a string representing a string header field
                        # make it look like it will look in the header
                        value = "'" + value + "'"
                        vstr = '%-20s' % value
                    else:
                        vstr = "%20s" % value
                else:
                    vstr = "''"
            else:
                if value is True:
                    value = 'T'
                elif value is False:
                    value = 'F'

                vstr = '%20s' % value

            card += vstr

            if 'comment' in record:
                card += ' / %s' % record['comment']

        if v_isstring and len(card) > 80:
            card = card[0:79] + "'"
        else:
            card = card[0:80]

        return card

    def __repr__(self):
        rep = ['']
        for r in self._record_list:
            card = self._record2card(r)
            # if 'card_string' not in r:
            #     card = self._record2card(r)
            # else:
            #     card = r['card_string']

            rep.append(card)
        return '\n'.join(rep)


class FITSRecord(dict):
    """
    Class to represent a FITS header record

    parameters
    ----------
    record: string or dict
        If a string, it should represent a FITS header card

        If a dict it should have 'name' and 'value' fields.
        Can have a 'comment' field.

    examples
    --------

    # from a dict.  Can include a comment
    rec=FITSRecord( {'name':'temp', 'value':35, 'comment':'temperature in C'} )

    # from a card
    card=FITSRecord('test    =                   77 / My comment')

    """
    def __init__(self, record):
        self.set_record(record)

    def set_record(self, record, **keys):
        """
        check the record is valid and set keys in the dict

        parameters
        ----------
        record: string
            Dict representing a record or a string representing a FITS header
            card
        """

        if keys:
            import warnings
            warnings.warn(
                "The keyword arguments '%s' are being ignored! This warning "
                "will be an error in a future version of `fitsio`!" % keys,
                DeprecationWarning, stacklevel=2)

        if isstring(record):
            card = FITSCard(record)
            self.update(card)

            self.verify()

        else:

            if isinstance(record, FITSRecord):
                self.update(record)
            elif isinstance(record, dict):
                if 'name' in record and 'value' in record:
                    self.update(record)

                elif 'card_string' in record:
                    self.set_record(record['card_string'])

                else:
                    raise ValueError('record must have name,value fields '
                                     'or a card_string field')
            else:
                raise ValueError("record must be a string card or "
                                 "dictionary or FITSRecord")

    def verify(self):
        """
        make sure name,value exist
        """
        if 'name' not in self:
            raise ValueError("each record must have a 'name' field")
        if 'value' not in self:
            raise ValueError("each record must have a 'value' field")


_BLANK = '       '


class FITSCard(FITSRecord):
    """
    class to represent ordinary FITS cards.

    CONTINUE not supported

    examples
    --------

    # from a card
    card=FITSRecord('test    =                   77 / My comment')
    """
    def __init__(self, card_string):
        self.set_card(card_string)

    def set_card(self, card_string):
        self['card_string'] = card_string

        self._check_hierarch()

        if self._is_hierarch:
            self._set_as_key()
        else:
            self._check_equals()

            self._check_type()
            self._check_len()

            front = card_string[0:7]
            if (not self.has_equals() or
                    front in ['COMMENT', 'HISTORY', 'CONTINU', _BLANK]):

                if front == 'HISTORY':
                    self._set_as_history()
                elif front == 'CONTINU':
                    self._set_as_continue()
                elif front == _BLANK:
                    self._set_as_blank()
                else:
                    # note anything without an = and not history and not blank
                    # key comment is treated as COMMENT; this is built into
                    # cfitsio as well
                    self._set_as_comment()

                if self.has_equals():
                    mess = (
                        "warning: It is not FITS-compliant for a %s header "
                        "card to include an = sign. There may be slight "
                        "inconsistencies if you write this back out to a "
                        "file.")
                    mess = mess % (card_string[:8])
                    warnings.warn(mess, FITSRuntimeWarning)
            else:
                self._set_as_key()

    def has_equals(self):
        """
        True if = is in position 8
        """
        return self._has_equals

    def _check_hierarch(self):
        card_string = self['card_string']
        if card_string[0:8].upper() == 'HIERARCH':
            self._is_hierarch = True
        else:
            self._is_hierarch = False

    def _check_equals(self):
        """
        check for = in position 8, set attribute _has_equals
        """
        card_string = self['card_string']
        if len(card_string) < 9:
            self._has_equals = False
        elif card_string[8] == '=':
            self._has_equals = True
        else:
            self._has_equals = False

    def _set_as_key(self):
        card_string = self['card_string']
        res = _fitsio_wrap.parse_card(card_string)
        if len(res) == 5:
            keyclass, name, value, dtype, comment = res
        else:
            keyclass, name, dtype, comment = res
            value = None

        if keyclass == TYP_CONT_KEY:
            raise ValueError("bad card '%s'.  CONTINUE not "
                             "supported" % card_string)

        self['class'] = keyclass
        self['name'] = name
        self['value_orig'] = value
        self['value'] = self._convert_value(value)
        self['dtype'] = dtype
        self['comment'] = comment

    def _set_as_blank(self):
        self['class'] = TYP_USER_KEY
        self['name'] = None
        self['value'] = None
        self['comment'] = self['card_string'][8:]

    def _set_as_comment(self):
        comment = self._extract_comm_or_hist_value()

        self['class'] = TYP_COMM_KEY
        self['name'] = 'COMMENT'
        self['value'] = comment

    def _set_as_history(self):
        history = self._extract_comm_or_hist_value()

        self['class'] = TYP_COMM_KEY
        self['name'] = 'HISTORY'
        self['value'] = history

    def _set_as_continue(self):
        value = self._extract_comm_or_hist_value()

        self['class'] = TYP_CONT_KEY
        self['name'] = 'CONTINUE'
        self['value'] = value

    def _convert_value(self, value_orig):
        """
        things like 6 and 1.25 are converted with ast.literal_value

        Things like 'hello' are stripped of quotes
        """
        import ast
        if value_orig is None:
            return value_orig

        if value_orig.startswith("'") and value_orig.endswith("'"):
            value = value_orig[1:-1]
        else:

            try:
                avalue = ast.parse(value_orig).body[0].value
                if isinstance(avalue, ast.BinOp):
                    # this is probably a string that happens to look like
                    # a binary operation, e.g. '25-3'
                    value = value_orig
                else:
                    value = ast.literal_eval(value_orig)
            except Exception:
                value = self._convert_string(value_orig)

            if isinstance(value, int) and '_' in value_orig:
                value = value_orig

        return value

    def _convert_string(self, s):
        if s == 'T':
            return True
        elif s == 'F':
            return False
        else:
            return s

    def _extract_comm_or_hist_value(self):
        card_string = self['card_string']
        if self._has_equals:
            if len(card_string) >= 9:
                value = card_string[9:]
            else:
                value = ''
        else:
            if len(card_string) >= 8:
                # value=card_string[7:]
                value = card_string[8:]
            else:
                value = ''
        return value

    def _check_type(self):
        card_string = self['card_string']
        if not isstring(card_string):
            raise TypeError(
                "card must be a string, got type %s" % type(card_string))

    def _check_len(self):
        ln = len(self['card_string'])
        if ln > 80:
            mess = "len(card) is %d.  cards must have length < 80"
            raise ValueError(mess)
