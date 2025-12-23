def _looks_like_msys2_mingw_scheme() -> bool:
    """MSYS2 patches distutils and sysconfig to use a UNIX-like scheme.

    However, MSYS2 incorrectly patches sysconfig ``nt`` scheme. The fix is
    likely going to be included in their 3.10 release, so we ignore the warning.
    See msys2/MINGW-packages#9319.

    MSYS2 MINGW's patch uses lowercase ``"lib"`` instead of the usual uppercase,
    and is missing the final ``"site-packages"``.
    """
    paths = sysconfig.get_paths("nt", expand=False)
    return all(
        "Lib" not in p and "lib" in p and not p.endswith("site-packages")
        for p in (paths[key] for key in ("platlib", "purelib"))
    )


def _fix_abiflags(parts: Tuple[str]) -> Generator[str, None, None]:
    ldversion = sysconfig.get_config_var("LDVERSION")
    abiflags = getattr(sys, "abiflags", None)

    # LDVERSION does not end with sys.abiflags. Just return the path unchanged.
    if not ldversion or not abiflags or not ldversion.endswith(abiflags):
        yield from parts
        return

    # Strip sys.abiflags from LDVERSION-based path components.
    for part in parts:
        if part.endswith(ldversion):
            part = part[: (0 - len(abiflags))]
        yield part


