if _should_use_importlib_metadata():
    from . import importlib

    return cast(Backend, importlib)
from . import pkg_resources

return cast(Backend, pkg_resources)


