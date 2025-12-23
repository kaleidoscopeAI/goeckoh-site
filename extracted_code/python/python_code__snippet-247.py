        # Mac OS 11 on x86_64 is compatible with binaries from previous releases.
        # Arm64 support was introduced in 11.0, so no Arm binaries from previous
        # releases exist.
        #
        # However, the "universal2" binary format can have a
        # macOS version earlier than 11.0 when the x86_64 part of the binary supports
        # that version of macOS.
        if arch == "x86_64":
            for minor_version in range(16, 3, -1):
                compat_version = 10, minor_version
                binary_formats = _mac_binary_formats(compat_version, arch)
                for binary_format in binary_formats:
                    yield "macosx_{major}_{minor}_{binary_format}".format(
                        major=compat_version[0],
                        minor=compat_version[1],
                        binary_format=binary_format,
                    )
        else:
            for minor_version in range(16, 3, -1):
                compat_version = 10, minor_version
                binary_format = "universal2"
                yield "macosx_{major}_{minor}_{binary_format}".format(
                    major=compat_version[0],
                    minor=compat_version[1],
                    binary_format=binary_format,
                )


def _linux_platforms(is_32bit: bool = _32_BIT_INTERPRETER) -> Iterator[str]:
    linux = _normalize_string(sysconfig.get_platform())
    if is_32bit:
        if linux == "linux_x86_64":
            linux = "linux_i686"
        elif linux == "linux_aarch64":
            linux = "linux_armv7l"
    _, arch = linux.split("_", 1)
    yield from _manylinux.platform_tags(linux, arch)
    yield from _musllinux.platform_tags(arch)
    yield linux


def _generic_platforms() -> Iterator[str]:
    yield _normalize_string(sysconfig.get_platform())


def platform_tags() -> Iterator[str]:
    """
    Provides the platform tags for this installation.
    """
    if platform.system() == "Darwin":
        return mac_platforms()
    elif platform.system() == "Linux":
        return _linux_platforms()
    else:
        return _generic_platforms()


def interpreter_name() -> str:
    """
    Returns the name of the running interpreter.
    """
    name = sys.implementation.name
    return INTERPRETER_SHORT_NAMES.get(name) or name


def interpreter_version(*, warn: bool = False) -> str:
    """
    Returns the version of the running interpreter.
    """
    version = _get_config_var("py_version_nodot", warn=warn)
    if version:
        version = str(version)
    else:
        version = _version_nodot(sys.version_info[:2])
    return version


def _version_nodot(version: PythonVersion) -> str:
    return "".join(map(str, version))


def sys_tags(*, warn: bool = False) -> Iterator[Tag]:
    """
    Returns the sequence of tag triples for the running interpreter.

    The order of the sequence corresponds to priority order for the
    interpreter, from most to least important.
    """

    interp_name = interpreter_name()
    if interp_name == "cp":
        yield from cpython_tags(warn=warn)
    else:
        yield from generic_tags()

    if interp_name == "pp":
        yield from compatible_tags(interpreter="pp3")
    else:
        yield from compatible_tags()


