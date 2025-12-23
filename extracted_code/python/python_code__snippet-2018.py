"""Checks whether the string "looks like" a path on the filesystem.

This does not check whether the target actually exists, only judge from the
appearance.

Returns true if any of the following conditions is true:
* a path separator is found (either os.path.sep or os.path.altsep);
* a dot is found (which represents the current directory).
"""
if os.path.sep in name:
    return True
if os.path.altsep is not None and os.path.altsep in name:
    return True
if name.startswith("."):
    return True
return False


