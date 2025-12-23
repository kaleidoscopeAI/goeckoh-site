"""Restores the :class:`ssl.SSLContext` class to its original state"""
setattr(ssl, "SSLContext", _original_SSLContext)
try:
    import pip._vendor.urllib3.util.ssl_ as urllib3_ssl

    urllib3_ssl.SSLContext = _original_SSLContext
except ImportError:
    pass


