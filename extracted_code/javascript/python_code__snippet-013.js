"""Extended globbing function that supports ** and {opt1,opt2,opt3}."""
if _CHECK_RECURSIVE_GLOB.search(path_glob):
    msg = """invalid glob %r: recursive glob "**" must be used alone"""
    raise ValueError(msg % path_glob)
if _CHECK_MISMATCH_SET.search(path_glob):
    msg = """invalid glob %r: mismatching set marker '{' or '}'"""
    raise ValueError(msg % path_glob)
return _iglob(path_glob)


