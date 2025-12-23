"""Return a dict with the Python implementation and version.

Provide both the name and the version of the Python implementation
currently running. For example, on CPython 3.10.3 it will return
{'name': 'CPython', 'version': '3.10.3'}.

This function works best on CPython and PyPy: in particular, it probably
doesn't work for Jython or IronPython. Future investigation should be done
to work out the correct shape of the code for those platforms.
"""
implementation = platform.python_implementation()

if implementation == "CPython":
    implementation_version = platform.python_version()
elif implementation == "PyPy":
    implementation_version = "{}.{}.{}".format(
        sys.pypy_version_info.major,
        sys.pypy_version_info.minor,
        sys.pypy_version_info.micro,
    )
    if sys.pypy_version_info.releaselevel != "final":
        implementation_version = "".join(
            [implementation_version, sys.pypy_version_info.releaselevel]
        )
elif implementation == "Jython":
    implementation_version = platform.python_version()  # Complete Guess
elif implementation == "IronPython":
    implementation_version = platform.python_version()  # Complete Guess
else:
    implementation_version = "Unknown"

return {"name": implementation, "version": implementation_version}


