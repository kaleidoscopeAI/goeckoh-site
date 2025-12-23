sys_glibc = _get_glibc_version()
if sys_glibc < version:
    return False
# Check for presence of _manylinux module.
try:
    import _manylinux  # noqa
except ImportError:
    return True
if hasattr(_manylinux, "manylinux_compatible"):
    result = _manylinux.manylinux_compatible(version[0], version[1], arch)
    if result is not None:
        return bool(result)
    return True
if version == _GLibCVersion(2, 5):
    if hasattr(_manylinux, "manylinux1_compatible"):
        return bool(_manylinux.manylinux1_compatible)
if version == _GLibCVersion(2, 12):
    if hasattr(_manylinux, "manylinux2010_compatible"):
        return bool(_manylinux.manylinux2010_compatible)
if version == _GLibCVersion(2, 17):
    if hasattr(_manylinux, "manylinux2014_compatible"):
        return bool(_manylinux.manylinux2014_compatible)
return True


