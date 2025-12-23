"""
Return a single named information item from the os-release file data source
of the current OS distribution.

Parameters:

* ``attribute`` (string): Key of the information item.

Returns:

* (string): Value of the information item, if the item exists.
  The empty string, if the item does not exist.

See `os-release file`_ for details about these information items.
"""
return _distro.os_release_attr(attribute)


