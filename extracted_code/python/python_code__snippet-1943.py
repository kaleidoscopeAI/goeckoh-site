"""Return a context manager used by captured_stdout/stdin/stderr
that temporarily replaces the sys stream *stream_name* with a StringIO.

Taken from Lib/support/__init__.py in the CPython repo.
"""
orig_stdout = getattr(sys, stream_name)
setattr(sys, stream_name, StreamWrapper.from_stream(orig_stdout))
try:
    yield getattr(sys, stream_name)
finally:
    setattr(sys, stream_name, orig_stdout)


