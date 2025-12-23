"""Exact matching of IP addresses.

RFC 6125 explicitly doesn't define an algorithm for this
(section 1.7.2 - "Out of Scope").
"""
# OpenSSL may add a trailing newline to a subjectAltName's IP address
# Divergence from upstream: ipaddress can't handle byte str
ip = ipaddress.ip_address(_to_unicode(ipname).rstrip())
return ip == host_ip


