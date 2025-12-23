def is_consecutive(c):
    c_int = ord(c)
    is_consecutive.prev, prev = c_int, is_consecutive.prev
    if c_int - prev > 1:
        is_consecutive.value = next(is_consecutive.counter)
    return is_consecutive.value

is_consecutive.prev = 0  # type: ignore [attr-defined]
is_consecutive.counter = itertools.count()  # type: ignore [attr-defined]
is_consecutive.value = -1  # type: ignore [attr-defined]

def escape_re_range_char(c):
    return "\\" + c if c in r"\^-][" else c

def no_escape_re_range_char(c):
    return c

if not re_escape:
    escape_re_range_char = no_escape_re_range_char

ret = []
s = "".join(sorted(set(s)))
if len(s) > 3:
    for _, chars in itertools.groupby(s, key=is_consecutive):
        first = last = next(chars)
        last = collections.deque(
            itertools.chain(iter([last]), chars), maxlen=1
        ).pop()
        if first == last:
            ret.append(escape_re_range_char(first))
        else:
            sep = "" if ord(last) == ord(first) + 1 else "-"
            ret.append(
                f"{escape_re_range_char(first)}{sep}{escape_re_range_char(last)}"
            )
else:
    ret = [escape_re_range_char(c) for c in s]

return "".join(ret)


