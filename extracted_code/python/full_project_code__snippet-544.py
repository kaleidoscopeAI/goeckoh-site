try:
    from ssl import PROTOCOL_SSLv23 as PROTOCOL_TLS

    PROTOCOL_SSLv23 = PROTOCOL_TLS
except ImportError:
    PROTOCOL_SSLv23 = PROTOCOL_TLS = 2

