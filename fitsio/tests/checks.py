def check_header(header, rh):
    for k in header:
        v = header[k]
        rv = rh[k]

        if isinstance(rv, str):
            v = v.strip()
            rv = rv.strip()

        assert v == rv, "testing equal key '%s'" % k


def compare_headerlist_header(header_list, header):
    """
    The first is a list of dicts, second a FITSHDR
    """
    for entry in header_list:
        name = entry['name'].upper()
        value = entry['value']
        hvalue = header[name]

        if isinstance(hvalue, str):
            hvalue = hvalue.strip()

        assert value == hvalue, (
            "testing header key '%s'" % name
        )

        if 'comment' in entry:
            assert (
                entry['comment'].strip() ==
                header.get_comment(name).strip()
            ), "testing comment for header key '%s'" % name
