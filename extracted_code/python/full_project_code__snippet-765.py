# Note: This logic prevents upgrading cryptography on Windows, if imported
#       as part of pip.
from pip._internal.utils.compat import WINDOWS
if not WINDOWS:
    raise ImportError("pip internals: don't import cryptography on Windows")
try:
    import ssl
except ImportError:
    ssl = None

if not getattr(ssl, "HAS_SNI", False):
    from pip._vendor.urllib3.contrib import pyopenssl

    pyopenssl.inject_into_urllib3()

    # Check cryptography version
    from cryptography import __version__ as cryptography_version

    _check_cryptography(cryptography_version)
