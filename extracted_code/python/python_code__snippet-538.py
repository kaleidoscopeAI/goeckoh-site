        raise suffixed_err(src, pos, "Cannot overwrite a value") from None
    if key_stem in nest:
        raise suffixed_err(src, pos, "Cannot overwrite a value")
    # Mark inline table and array namespaces recursively immutable
    if isinstance(value, (dict, list)):
        out.flags.set(header + key, Flags.FROZEN, recursive=True)
    nest[key_stem] = value
    return pos


def parse_key_value_pair(
    src: str, pos: Pos, parse_float: ParseFloat
