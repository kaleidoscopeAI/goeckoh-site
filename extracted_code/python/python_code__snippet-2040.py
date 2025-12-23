"""Get value from dictionary and verify expected type."""
if key not in d:
    return default
value = d[key]
if not isinstance(value, expected_type):
    raise DirectUrlValidationError(
        f"{value!r} has unexpected type for {key} (expected {expected_type})"
    )
return value


