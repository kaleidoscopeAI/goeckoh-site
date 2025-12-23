"Fallback implementation of glibc_version_string using ctypes."

try:
    import ctypes
except ImportError:
    return None

# ctypes.CDLL(None) internally calls dlopen(NULL), and as the dlopen
# manpage says, "If filename is NULL, then the returned handle is for the
# main program". This way we can let the linker do the work to figure out
# which libc our process is actually using.
process_namespace = ctypes.CDLL(None)
try:
    gnu_get_libc_version = process_namespace.gnu_get_libc_version
except AttributeError:
    # Symbol doesn't exist -> therefore, we are not linked to
    # glibc.
    return None

# Call gnu_get_libc_version, which returns a string like "2.5"
gnu_get_libc_version.restype = ctypes.c_char_p
version_str = gnu_get_libc_version()
# py2 / py3 compatibility:
if not isinstance(version_str, str):
    version_str = version_str.decode("ascii")

return version_str


