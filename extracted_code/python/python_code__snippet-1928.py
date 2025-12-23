"""Parse the version string from a <package>+<version> filename
"fragment" (stem) or egg fragment.

:param fragment: The string to parse. E.g. foo-2.1
:param canonical_name: The canonicalized name of the package this
    belongs to.
"""
try:
    version_start = _find_name_version_sep(fragment, canonical_name) + 1
except ValueError:
    return None
version = fragment[version_start:]
if not version:
    return None
return version


