def check_header(header, rh):
    for k in header:
        v = header[k]
        rv = rh[k]

        if isinstance(rv, str):
            v = v.strip()
            rv = rv.strip()

        assert v == rv, "testing equal key '%s'" % k
