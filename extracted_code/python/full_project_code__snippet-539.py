# Apparently some systems have a select.poll that fails as soon as you try
# to use it, either due to strange configuration or broken monkeypatching
# from libraries like eventlet/greenlet.
try:
    poll_obj = select.poll()
    _retry_on_intr(poll_obj.poll, 0)
except (AttributeError, OSError):
    return False
else:
    return True


