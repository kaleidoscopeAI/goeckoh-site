# We delay choosing which implementation to use until the first time we're
# called. We could do it at import time, but then we might make the wrong
# decision if someone goes wild with monkeypatching select.poll after
# we're imported.
global wait_for_socket
if _have_working_poll():
    wait_for_socket = poll_wait_for_socket
elif hasattr(select, "select"):
    wait_for_socket = select_wait_for_socket
else:  # Platform-specific: Appengine.
    wait_for_socket = null_wait_for_socket
return wait_for_socket(*args, **kwargs)


