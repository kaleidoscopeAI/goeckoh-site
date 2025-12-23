try:
    import _ssl  # noqa: F401  # ignore unused

    return True
except ImportError:
    pass

from pip._vendor.urllib3.util import IS_PYOPENSSL

return IS_PYOPENSSL


