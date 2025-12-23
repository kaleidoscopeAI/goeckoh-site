import os
import types
from typing import Union

Package = Union[types.ModuleType, str]
Resource = Union[str, "os.PathLike"]

# This fallback will work for Python versions prior to 3.7 that lack the
# importlib.resources module but relies on the existing `where` function
# so won't address issues with environments like PyOxidizer that don't set
# __file__ on modules.
def read_text(
    package: Package,
    resource: Resource,
    encoding: str = 'utf-8',
    errors: str = 'strict'
) -> str:
    with open(where(), encoding=encoding) as data:
        return data.read()

# If we don't have importlib.resources, then we will just do the old logic
# of assuming we're on the filesystem and munge the path directly.
def where() -> str:
    f = os.path.dirname(__file__)

    return os.path.join(f, "cacert.pem")

def contents() -> str:
    return read_text("pip._vendor.certifi", "cacert.pem", encoding="ascii")


