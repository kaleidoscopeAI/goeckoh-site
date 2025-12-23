"""
A utility method used to get name and version from a string.

From e.g. a Provides-Dist value.

:param p: A value in a form 'foo (1.0)'
:return: The name and version as a tuple.
"""
m = NAME_VERSION_RE.match(p)
if not m:
    raise DistlibException('Ill-formed name/version string: \'%s\'' % p)
d = m.groupdict()
return d['name'].strip().lower(), d['ver']


