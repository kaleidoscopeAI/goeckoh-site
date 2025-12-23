# Import distutils lazily to avoid deprecation warnings,
# but import it soon enough that it is in memory and available during
# a pip reinstall.
from . import _distutils

