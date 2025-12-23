"""
.. deprecated:: 1.6.0

    :func:`distro.linux_distribution()` is deprecated. It should only be
    used as a compatibility shim with Python's
    :py:func:`platform.linux_distribution()`. Please use :func:`distro.id`,
    :func:`distro.version` and :func:`distro.name` instead.

Return information about the current OS distribution as a tuple
``(id_name, version, codename)`` with items as follows:

* ``id_name``:  If *full_distribution_name* is false, the result of
  :func:`distro.id`. Otherwise, the result of :func:`distro.name`.

* ``version``:  The result of :func:`distro.version`.

* ``codename``:  The extra item (usually in parentheses) after the
  os-release version number, or the result of :func:`distro.codename`.

The interface of this function is compatible with the original
:py:func:`platform.linux_distribution` function, supporting a subset of
its parameters.

The data it returns may not exactly be the same, because it uses more data
sources than the original function, and that may lead to different data if
the OS distribution is not consistent across multiple data sources it
provides (there are indeed such distributions ...).

Another reason for differences is the fact that the :func:`distro.id`
method normalizes the distro ID string to a reliable machine-readable value
for a number of popular OS distributions.
"""
warnings.warn(
    "distro.linux_distribution() is deprecated. It should only be used as a "
    "compatibility shim with Python's platform.linux_distribution(). Please use "
    "distro.id(), distro.version() and distro.name() instead.",
    DeprecationWarning,
    stacklevel=2,
)
return _distro.linux_distribution(full_distribution_name)


