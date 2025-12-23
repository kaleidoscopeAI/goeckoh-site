"""Get system site packages

Usually from site.getsitepackages,
but fallback on `get_purelib()/get_platlib()` if unavailable
(e.g. in a virtualenv created by virtualenv<20)

Returns normalized set of strings.
"""
if hasattr(site, "getsitepackages"):
    system_sites = site.getsitepackages()
else:
    # virtualenv < 20 overwrites site.py without getsitepackages
    # fallback on get_purelib/get_platlib.
    # this is known to miss things, but shouldn't in the cases
    # where getsitepackages() has been removed (inside a virtualenv)
    system_sites = [get_purelib(), get_platlib()]
return {os.path.normcase(path) for path in system_sites}


