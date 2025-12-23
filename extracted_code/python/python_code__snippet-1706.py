"""Injects the :class:`truststore.SSLContext` into the ``ssl``
module by replacing :class:`ssl.SSLContext`.
"""
setattr(ssl, "SSLContext", SSLContext)
# urllib3 holds on to its own reference of ssl.SSLContext
# so we need to replace that reference too.
try:
    import pip._vendor.urllib3.util.ssl_ as urllib3_ssl

    setattr(urllib3_ssl, "SSLContext", SSLContext)
except ImportError:
    pass


