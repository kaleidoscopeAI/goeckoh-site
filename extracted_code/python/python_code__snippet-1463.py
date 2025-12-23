"""
Return the name of the current OS distribution, as a human-readable
string.

If *pretty* is false, the name is returned without version or codename.
(e.g. "CentOS Linux")

If *pretty* is true, the version and codename are appended.
(e.g. "CentOS Linux 7.1.1503 (Core)")

**Lookup hierarchy:**

The name is obtained from the following sources, in the specified order.
The first available and non-empty value is used:

* If *pretty* is false:

  - the value of the "NAME" attribute of the os-release file,

  - the value of the "Distributor ID" attribute returned by the lsb_release
    command,

  - the value of the "<name>" field of the distro release file.

* If *pretty* is true:

  - the value of the "PRETTY_NAME" attribute of the os-release file,

  - the value of the "Description" attribute returned by the lsb_release
    command,

  - the value of the "<name>" field of the distro release file, appended
    with the value of the pretty version ("<version_id>" and "<codename>"
    fields) of the distro release file, if available.
"""
return _distro.name(pretty)


