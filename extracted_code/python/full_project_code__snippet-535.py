"""Connect to *address* and return the socket object.

Convenience function.  Connect to *address* (a 2-tuple ``(host,
port)``) and return the socket object.  Passing the optional
*timeout* parameter will set the timeout on the socket instance
before attempting to connect.  If no *timeout* is supplied, the
global default timeout setting returned by :func:`socket.getdefaulttimeout`
is used.  If *source_address* is set it must be a tuple of (host, port)
for the socket to bind as a source address before making the connection.
An host of '' or port 0 tells the OS to use the default.
"""

host, port = address
if host.startswith("["):
    host = host.strip("[]")
err = None

# Using the value from allowed_gai_family() in the context of getaddrinfo lets
# us select whether to work with IPv4 DNS records, IPv6 records, or both.
# The original create_connection function always returns all records.
family = allowed_gai_family()

try:
    host.encode("idna")
except UnicodeError:
    return six.raise_from(
        LocationParseError(u"'%s', label empty or too long" % host), None
    )

for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
    af, socktype, proto, canonname, sa = res
    sock = None
    try:
        sock = socket.socket(af, socktype, proto)

        # If provided, set socket level options before connecting.
        _set_socket_options(sock, socket_options)

        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(timeout)
        if source_address:
            sock.bind(source_address)
        sock.connect(sa)
        return sock

    except socket.error as e:
        err = e
        if sock is not None:
            sock.close()
            sock = None

if err is not None:
    raise err

raise socket.error("getaddrinfo returns an empty list")


