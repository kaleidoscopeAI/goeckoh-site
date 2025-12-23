"""
Intuitively, this is `options.get(optname, default)`, but restricted to
Boolean value. The Booleans can be represented as string, in order to accept
Boolean value from the command line arguments. If the key `optname` is
present in the dictionary `options` and is not associated with a Boolean,
raise an `OptionError`. If it is absent, `default` is returned instead.

The valid string values for ``True`` are ``1``, ``yes``, ``true`` and
``on``, the ones for ``False`` are ``0``, ``no``, ``false`` and ``off``
(matched case-insensitively).
"""
string = options.get(optname, default)
if isinstance(string, bool):
    return string
elif isinstance(string, int):
    return bool(string)
elif not isinstance(string, str):
    raise OptionError('Invalid type %r for option %s; use '
                      '1/0, yes/no, true/false, on/off' % (
                          string, optname))
elif string.lower() in ('1', 'yes', 'true', 'on'):
    return True
elif string.lower() in ('0', 'no', 'false', 'off'):
    return False
else:
    raise OptionError('Invalid value %r for option %s; use '
                      '1/0, yes/no, true/false, on/off' % (
                          string, optname))


