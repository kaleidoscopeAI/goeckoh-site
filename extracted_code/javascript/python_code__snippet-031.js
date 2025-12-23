"""Find the separator's index based on the package's canonical name.

:param fragment: A <package>+<version> filename "fragment" (stem) or
    egg fragment.
:param canonical_name: The package's canonical name.

This function is needed since the canonicalized name does not necessarily
have the same length as the egg info's name part. An example::

>>> fragment = 'foo__bar-1.0'
>>> canonical_name = 'foo-bar'
>>> _find_name_version_sep(fragment, canonical_name)
8
"""
# Project name and version must be separated by one single dash. Find all
# occurrences of dashes; if the string in front of it matches the canonical
# name, this is the one separating the name and version parts.
for i, c in enumerate(fragment):
    if c != "-":
        continue
    if canonicalize_name(fragment[:i]) == canonical_name:
        return i
raise ValueError(f"{fragment} does not match {canonical_name}")


