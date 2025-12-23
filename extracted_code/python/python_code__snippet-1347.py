"""Suggest a normalized version close to the given version string.

If you have a version string that isn't rational (i.e. NormalizedVersion
doesn't like it) then you might be able to get an equivalent (or close)
rational version from this function.

This does a number of simple normalizations to the given string, based
on observation of versions currently in use on PyPI. Given a dump of
those version during PyCon 2009, 4287 of them:
- 2312 (53.93%) match NormalizedVersion without change
  with the automatic suggestion
- 3474 (81.04%) match when using this suggestion method

@param s {str} An irrational version string.
@returns A rational version string, or None, if couldn't determine one.
"""
try:
    _normalized_key(s)
    return s   # already rational
except UnsupportedVersionError:
    pass

rs = s.lower()

# part of this could use maketrans
for orig, repl in (('-alpha', 'a'), ('-beta', 'b'), ('alpha', 'a'),
                   ('beta', 'b'), ('rc', 'c'), ('-final', ''),
                   ('-pre', 'c'),
                   ('-release', ''), ('.release', ''), ('-stable', ''),
                   ('+', '.'), ('_', '.'), (' ', ''), ('.final', ''),
                   ('final', '')):
    rs = rs.replace(orig, repl)

# if something ends with dev or pre, we add a 0
rs = re.sub(r"pre$", r"pre0", rs)
rs = re.sub(r"dev$", r"dev0", rs)

# if we have something like "b-2" or "a.2" at the end of the
# version, that is probably beta, alpha, etc
# let's remove the dash or dot
rs = re.sub(r"([abc]|rc)[\-\.](\d+)$", r"\1\2", rs)

# 1.0-dev-r371 -> 1.0.dev371
# 0.1-dev-r79 -> 0.1.dev79
rs = re.sub(r"[\-\.](dev)[\-\.]?r?(\d+)$", r".\1\2", rs)

# Clean: 2.0.a.3, 2.0.b1, 0.9.0~c1
rs = re.sub(r"[.~]?([abc])\.?", r"\1", rs)

# Clean: v0.3, v1.0
if rs.startswith('v'):
    rs = rs[1:]

# Clean leading '0's on numbers.
# TODO: unintended side-effect on, e.g., "2003.05.09"
# PyPI stats: 77 (~2%) better
rs = re.sub(r"\b0+(\d+)(?!\d)", r"\1", rs)

# Clean a/b/c with no version. E.g. "1.0a" -> "1.0a0". Setuptools infers
# zero.
# PyPI stats: 245 (7.56%) better
rs = re.sub(r"(\d+[abc])$", r"\g<1>0", rs)

# the 'dev-rNNN' tag is a dev tag
rs = re.sub(r"\.?(dev-r|dev\.r)\.?(\d+)$", r".dev\2", rs)

# clean the - when used as a pre delimiter
rs = re.sub(r"-(a|b|c)(\d+)$", r"\1\2", rs)

# a terminal "dev" or "devel" can be changed into ".dev0"
rs = re.sub(r"[\.\-](dev|devel)$", r".dev0", rs)

# a terminal "dev" can be changed into ".dev0"
rs = re.sub(r"(?![\.\-])dev$", r".dev0", rs)

# a terminal "final" or "stable" can be removed
rs = re.sub(r"(final|stable)$", "", rs)

# The 'r' and the '-' tags are post release tags
#   0.4a1.r10       ->  0.4a1.post10
#   0.9.33-17222    ->  0.9.33.post17222
#   0.9.33-r17222   ->  0.9.33.post17222
rs = re.sub(r"\.?(r|-|-r)\.?(\d+)$", r".post\2", rs)

# Clean 'r' instead of 'dev' usage:
#   0.9.33+r17222   ->  0.9.33.dev17222
#   1.0dev123       ->  1.0.dev123
#   1.0.git123      ->  1.0.dev123
#   1.0.bzr123      ->  1.0.dev123
#   0.1a0dev.123    ->  0.1a0.dev123
# PyPI stats:  ~150 (~4%) better
rs = re.sub(r"\.?(dev|git|bzr)\.?(\d+)$", r".dev\2", rs)

# Clean '.pre' (normalized from '-pre' above) instead of 'c' usage:
#   0.2.pre1        ->  0.2c1
#   0.2-c1         ->  0.2c1
#   1.0preview123   ->  1.0c123
# PyPI stats: ~21 (0.62%) better
rs = re.sub(r"\.?(pre|preview|-c)(\d+)$", r"c\g<2>", rs)

# Tcl/Tk uses "px" for their post release markers
rs = re.sub(r"p(\d+)$", r".post\1", rs)

try:
    _normalized_key(rs)
except UnsupportedVersionError:
    rs = None
return rs

