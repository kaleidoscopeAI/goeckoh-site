def _looks_like_red_hat_lib() -> bool:
    """Red Hat patches platlib in unix_prefix and unix_home, but not purelib.

    This is the only way I can see to tell a Red Hat-patched Python.
    """
    from distutils.command.install import INSTALL_SCHEMES

    return all(
        k in INSTALL_SCHEMES
        and _looks_like_red_hat_patched_platlib_purelib(INSTALL_SCHEMES[k])
        for k in ("unix_prefix", "unix_home")
    )


