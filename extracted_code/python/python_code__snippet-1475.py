# Ugly hack...
RecursionError = RuntimeError

def _is_recursionerror(e):
    return (
        len(e.args) == 1
        and isinstance(e.args[0], str)
        and e.args[0].startswith("maximum recursion depth exceeded")
    )

