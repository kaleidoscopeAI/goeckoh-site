    def _retry_on_intr(fn, timeout):
        return fn(timeout)

