"""
Verifies that PyOpenSSL's package-level dependencies have been met.
Throws `ImportError` if they are not met.
"""
# Method added in `cryptography==1.1`; not available in older versions
from cryptography.x509.extensions import Extensions

if getattr(Extensions, "get_extension_for_class", None) is None:
    raise ImportError(
        "'cryptography' module missing required functionality.  "
        "Try upgrading to v1.3.4 or newer."
    )

# pyOpenSSL 0.14 and above use cryptography for OpenSSL bindings. The _x509
# attribute is only present on those versions.
from OpenSSL.crypto import X509

x509 = X509()
if getattr(x509, "_x509", None) is None:
    raise ImportError(
        "'pyOpenSSL' module missing required functionality. "
        "Try upgrading to v0.14 or newer."
    )


