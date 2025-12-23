# This is taken from PEP 503.
value = _canonicalize_regex.sub("-", name).lower()
return cast(NormalizedName, value)


