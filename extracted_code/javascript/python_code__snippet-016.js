"""
Return the codename for the release of the current OS distribution,
as a string.

If the distribution does not have a codename, an empty string is returned.

Note that the returned codename is not always really a codename. For
example, openSUSE returns "x86_64". This function does not handle such
cases in any special way and just returns the string it finds, if any.

**Lookup hierarchy:**

* the codename within the "VERSION" attribute of the os-release file, if
  provided,

* the value of the "Codename" attribute returned by the lsb_release
  command,

* the value of the "<codename>" field of the distro release file.
"""
return _distro.codename()


