# Modern Python, that retries syscalls by default
def _retry_on_intr(fn, timeout):
    return fn(timeout)

