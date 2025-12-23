"""
Return a single named information item from the distro release file
data source of the current OS distribution.

Parameters:

* ``attribute`` (string): Key of the information item.

Returns:

* (string): Value of the information item, if the item exists.
  The empty string, if the item does not exist.

See `distro release file`_ for details about these information items.
"""
return _distro.distro_release_attr(attribute)


