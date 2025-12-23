"""Return the default platform-shared lib location."""
new = _sysconfig.get_platlib()
if _USE_SYSCONFIG:
    return new

from . import _distutils

old = _distutils.get_platlib()
if _looks_like_deb_system_dist_packages(old):
    return old
if _warn_if_mismatch(pathlib.Path(old), pathlib.Path(new), key="platlib"):
    _log_context()
return old


