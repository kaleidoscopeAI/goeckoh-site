"""
Verifies the peer certificates from an SSLSocket or SSLObject
against the certificates in the OS trust store.
"""
sslobj: ssl.SSLObject = sock_or_sslobj  # type: ignore[assignment]
try:
    while not hasattr(sslobj, "get_unverified_chain"):
        sslobj = sslobj._sslobj  # type: ignore[attr-defined]
except AttributeError:
    pass

# SSLObject.get_unverified_chain() returns 'None'
# if the peer sends no certificates. This is common
# for the server-side scenario.
unverified_chain: typing.Sequence[_ssl.Certificate] = (
    sslobj.get_unverified_chain() or ()  # type: ignore[attr-defined]
)
cert_bytes = [cert.public_bytes(_ssl.ENCODING_DER) for cert in unverified_chain]
_verify_peercerts_impl(
    sock_or_sslobj.context, cert_bytes, server_hostname=server_hostname
)


