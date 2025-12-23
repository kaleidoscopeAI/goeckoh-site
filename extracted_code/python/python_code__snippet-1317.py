pos += 2  # Skip "[["
pos = skip_chars(src, pos, TOML_WS)
pos, key = parse_key(src, pos)

if out.flags.is_(key, Flags.FROZEN):
    raise suffixed_err(src, pos, f"Cannot mutate immutable namespace {key}")
# Free the namespace now that it points to another empty list item...
out.flags.unset_all(key)
# ...but this key precisely is still prohibited from table declaration
out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
try:
    out.data.append_nest_to_list(key)
except KeyError:
    raise suffixed_err(src, pos, "Cannot overwrite a value") from None

if not src.startswith("]]", pos):
    raise suffixed_err(src, pos, "Expected ']]' at the end of an array declaration")
return pos + 2, key


