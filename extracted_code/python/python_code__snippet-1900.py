target = os.environ

# Save values from the target and change them.
non_existent_marker = object()
saved_values: Dict[str, Union[object, str]] = {}
for name, new_value in changes.items():
    try:
        saved_values[name] = target[name]
    except KeyError:
        saved_values[name] = non_existent_marker
    target[name] = new_value

try:
    yield
finally:
    # Restore original values in the target.
    for name, original_value in saved_values.items():
        if original_value is non_existent_marker:
            del target[name]
        else:
            assert isinstance(original_value, str)  # for mypy
            target[name] = original_value


