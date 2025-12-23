"""
Return a dictionary containing key-value pairs for the information items
from the os-release file data source of the current OS distribution.

See `os-release file`_ for details about these information items.
"""
return _distro.os_release_info()


