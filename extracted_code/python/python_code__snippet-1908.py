"""The resolution to bpo-44860 will change this incorrect platlib.

See <https://bugs.python.org/issue44860>.
"""
from distutils.command.install import INSTALL_SCHEMES

try:
    unix_user_platlib = INSTALL_SCHEMES["unix_user"]["platlib"]
except KeyError:
    return False
return unix_user_platlib == "$usersite"


