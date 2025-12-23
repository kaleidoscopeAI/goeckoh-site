@functools.wraps(fn)
def unique(*args: Any, **kw: Any) -> Generator[Any, None, None]:
    seen: Set[Any] = set()
    for item in fn(*args, **kw):
        if item not in seen:
            seen.add(item)
            yield item

return unique


