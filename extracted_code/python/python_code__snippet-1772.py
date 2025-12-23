"""Return this platform's string for platform-specific distributions

XXX Currently this is the same as ``distutils.util.get_platform()``, but it
needs some hacks for Linux and macOS.
"""
from sysconfig import get_platform

plat = get_platform()
if sys.platform == "darwin" and not plat.startswith('macosx-'):
    try:
        version = _macos_vers()
        machine = os.uname()[4].replace(" ", "_")
        return "macosx-%d.%d-%s" % (
            int(version[0]),
            int(version[1]),
            _macos_arch(machine),
        )
    except ValueError:
        # if someone is running a non-Mac darwin system, this will fall
        # through to the default implementation
        pass
return plat


