@functools.wraps(fn)
def wrapped(self: "Specifier", prospective: ParsedVersion, spec: str) -> bool:
    if not isinstance(prospective, Version):
        return False
    return fn(self, prospective, spec)

return wrapped


