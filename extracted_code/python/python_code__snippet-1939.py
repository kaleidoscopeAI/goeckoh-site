"""
`rmtree` error handler to 'force' a file remove (i.e. like `rm -f`).

* If a file is readonly then it's write flag is set and operation is
  retried.

* `onerror` is the original callback from `rmtree(... onerror=onerror)`
  that is chained at the end if the "rm -f" still fails.
"""
try:
    st_mode = os.stat(path).st_mode
except OSError:
    # it's equivalent to os.path.exists
    return

if not st_mode & stat.S_IWRITE:
    # convert to read/write
    try:
        os.chmod(path, st_mode | stat.S_IWRITE)
    except OSError:
        pass
    else:
        # use the original function to repeat the operation
        try:
            func(path)
            return
        except OSError:
            pass

if not isinstance(exc_info, BaseException):
    _, exc_info, _ = exc_info
onexc(func, path, exc_info)


