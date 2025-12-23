"""A decorator to make `parse_float` safe.

`parse_float` must not return dicts or lists, because these types
would be mixed with parsed TOML tables and arrays, thus confusing
the parser. The returned decorated callable raises `ValueError`
instead of returning illegal types.
"""
# The default `float` callable never returns illegal types. Optimize it.
if parse_float is float:  # type: ignore[comparison-overlap]
    return float

def safe_parse_float(float_str: str) -> Any:
    float_value = parse_float(float_str)
    if isinstance(float_value, (dict, list)):
        raise ValueError("parse_float must not return dicts or lists")
    return float_value

return safe_parse_float


