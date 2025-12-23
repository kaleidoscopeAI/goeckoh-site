"""This function is designed to work in the context of
getaddrinfo, where family=socket.AF_UNSPEC is the default and
will perform a DNS search for both IPv6 and IPv4 records."""

family = socket.AF_INET
if HAS_IPV6:
    family = socket.AF_UNSPEC
return family


