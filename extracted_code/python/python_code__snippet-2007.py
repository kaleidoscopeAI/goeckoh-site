# Module name can be uppercase in vendor.txt for some reason...
module_name = module_name.lower().replace("-", "_")
# PATCH: setuptools is actually only pkg_resources.
if module_name == "setuptools":
    module_name = "pkg_resources"

try:
    __import__(f"pip._vendor.{module_name}", globals(), locals(), level=0)
    return getattr(pip._vendor, module_name)
except ImportError:
    # We allow 'truststore' to fail to import due
    # to being unavailable on Python 3.9 and earlier.
    if module_name == "truststore" and sys.version_info < (3, 10):
        return None
    raise


