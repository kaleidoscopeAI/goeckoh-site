class BrokenStdoutLoggingError(Exception):
    """
    Raised if BrokenPipeError occurs for the stdout stream while logging.
    """


def _is_broken_pipe_error(exc_class: Type[BaseException], exc: BaseException) -> bool:
    if exc_class is BrokenPipeError:
        return True

    # On Windows, a broken pipe can show up as EINVAL rather than EPIPE:
    # https://bugs.python.org/issue19612
    # https://bugs.python.org/issue30418
    if not WINDOWS:
        return False

    return isinstance(exc, OSError) and exc.errno in (errno.EINVAL, errno.EPIPE)


