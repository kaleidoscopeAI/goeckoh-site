def make_tuple(s, absent):
    if s is None:
        result = (absent,)
    else:
        parts = s[1:].split('.')
        # We can't compare ints and strings on Python 3, so fudge it
        # by zero-filling numeric values so simulate a numeric comparison
        result = tuple([p.zfill(8) if p.isdigit() else p for p in parts])
    return result

m = is_semver(s)
if not m:
    raise UnsupportedVersionError(s)
groups = m.groups()
major, minor, patch = [int(i) for i in groups[:3]]
# choose the '|' and '*' so that versions sort correctly
pre, build = make_tuple(groups[3], '|'), make_tuple(groups[5], '*')
return (major, minor, patch), pre, build


