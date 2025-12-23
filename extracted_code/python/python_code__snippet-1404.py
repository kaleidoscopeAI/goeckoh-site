if sys.version_info[0] < 3:
    mode += 'b'
else:
    kwargs['newline'] = ''
    # Python 3 determines encoding from locale. Force 'utf-8'
    # file encoding to match other forced utf-8 encoding
    kwargs['encoding'] = 'utf-8'
return open(fn, mode, **kwargs)


