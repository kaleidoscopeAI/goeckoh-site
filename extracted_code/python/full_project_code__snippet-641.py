"""
If the key `optname` from the dictionary is not in the sequence
`allowed`, raise an error, otherwise return it.
"""
string = options.get(optname, default)
if normcase:
    string = string.lower()
if string not in allowed:
    raise OptionError('Value for option %s must be one of %s' %
                      (optname, ', '.join(map(str, allowed))))
return string


