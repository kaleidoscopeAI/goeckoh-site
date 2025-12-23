"""
Return True if path is within sys.prefix, if we're running in a virtualenv.

If we're not in a virtualenv, all paths are considered "local."

Caution: this function assumes the head of path has been normalized
with normalize_path.
"""
if not running_under_virtualenv():
    return True
return path.startswith(normalize_path(sys.prefix))


