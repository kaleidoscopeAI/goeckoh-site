def _looks_like_bpo_44860() -> bool:
    """The resolution to bpo-44860 will change this incorrect platlib.

    See <https://bugs.python.org/issue44860>.
    """
    from distutils.command.install import INSTALL_SCHEMES

    try:
        unix_user_platlib = INSTALL_SCHEMES["unix_user"]["platlib"]
    except KeyError:
        return False
    return unix_user_platlib == "$usersite"


def _looks_like_red_hat_patched_platlib_purelib(scheme: Dict[str, str]) -> bool:
    platlib = scheme["platlib"]
    if "/$platlibdir/" in platlib:
        platlib = platlib.replace("/$platlibdir/", f"/{_PLATLIBDIR}/")
    if "/lib64/" not in platlib:
        return False
    unpatched = platlib.replace("/lib64/", "/lib/")
    return unpatched.replace("$platbase/", "$base/") == scheme["purelib"]


