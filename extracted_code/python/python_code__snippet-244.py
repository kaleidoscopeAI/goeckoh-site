        yield from (Tag(interpreter, "abi3", platform_) for platform_ in platforms)
    yield from (Tag(interpreter, "none", platform_) for platform_ in platforms)

    if _abi3_applies(python_version):
        for minor_version in range(python_version[1] - 1, 1, -1):
            for platform_ in platforms:
                interpreter = "cp{version}".format(
                    version=_version_nodot((python_version[0], minor_version))
                )
                yield Tag(interpreter, "abi3", platform_)


def _generic_abi() -> Iterator[str]:
    abi = sysconfig.get_config_var("SOABI")
    if abi:
        yield _normalize_string(abi)


def generic_tags(
    interpreter: Optional[str] = None,
    abis: Optional[Iterable[str]] = None,
    platforms: Optional[Iterable[str]] = None,
    *,
    warn: bool = False,
