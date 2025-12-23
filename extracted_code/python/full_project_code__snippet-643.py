"""
If the key `optname` from the dictionary `options` is a string,
split it at whitespace and return it. If it is already a list
or a tuple, it is returned as a list.
"""
val = options.get(optname, default)
if isinstance(val, str):
    return val.split()
elif isinstance(val, (list, tuple)):
    return list(val)
else:
    raise OptionError('Invalid type %r for option %s; you '
                      'must give a list value' % (
                          val, optname))


