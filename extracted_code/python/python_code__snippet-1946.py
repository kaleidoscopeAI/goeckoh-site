"""
Build a full URL from a netloc.
"""
if netloc.count(":") >= 2 and "@" not in netloc and "[" not in netloc:
    # It must be a bare IPv6 address, so wrap it with brackets.
    netloc = f"[{netloc}]"
return f"{scheme}://{netloc}"


