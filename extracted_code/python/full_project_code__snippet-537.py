# Old and broken Pythons.
def _retry_on_intr(fn, timeout):
    if timeout is None:
        deadline = float("inf")
    else:
        deadline = monotonic() + timeout

    while True:
        try:
            return fn(timeout)
        # OSError for 3 <= pyver < 3.5, select.error for pyver <= 2.7
        except (OSError, select.error) as e:
            # 'e.args[0]' incantation works for both OSError and select.error
            if e.args[0] != errno.EINTR:
                raise
            else:
                timeout = deadline - monotonic()
                if timeout < 0:
                    timeout = 0
                if timeout == float("inf"):
                    timeout = None
                continue


