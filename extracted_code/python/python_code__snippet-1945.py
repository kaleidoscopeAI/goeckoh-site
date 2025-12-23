"""
Build a netloc from a host-port pair
"""
if port is None:
    return host
if ":" in host:
    # Only wrap host with square brackets when it is IPv6
    host = f"[{host}]"
return f"{host}:{port}"


