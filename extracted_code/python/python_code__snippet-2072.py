if sys.version_info < (3, 10):
    raise CommandError("The truststore feature is only available for Python 3.10+")

try:
    import ssl
except ImportError:
    logger.warning("Disabling truststore since ssl support is missing")
    return None

try:
    from pip._vendor import truststore
except ImportError as e:
    raise CommandError(f"The truststore feature is unavailable: {e}")

return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


