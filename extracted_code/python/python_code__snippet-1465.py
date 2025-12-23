"""
Return a dictionary containing key-value pairs for the information items
from the lsb_release command data source of the current OS distribution.

See `lsb_release command output`_ for details about these information
items.
"""
return _distro.lsb_release_info()


