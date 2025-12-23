"""Debian adds two additional schemes."""
from distutils.command.install import INSTALL_SCHEMES

return "deb_system" in INSTALL_SCHEMES and "unix_local" in INSTALL_SCHEMES


