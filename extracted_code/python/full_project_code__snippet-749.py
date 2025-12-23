# provide a proxy_bypass version on Windows without DNS lookups

def proxy_bypass_registry(host):
    try:
        import winreg
    except ImportError:
        return False

    try:
        internetSettings = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        # ProxyEnable could be REG_SZ or REG_DWORD, normalizing it
        proxyEnable = int(winreg.QueryValueEx(internetSettings, "ProxyEnable")[0])
        # ProxyOverride is almost always a string
        proxyOverride = winreg.QueryValueEx(internetSettings, "ProxyOverride")[0]
    except (OSError, ValueError):
        return False
    if not proxyEnable or not proxyOverride:
        return False

    # make a check value list from the registry entry: replace the
    # '<local>' string by the localhost entry and the corresponding
    # canonical entry.
    proxyOverride = proxyOverride.split(";")
    # now check if we match one of the registry values.
    for test in proxyOverride:
        if test == "<local>":
            if "." not in host:
                return True
        test = test.replace(".", r"\.")  # mask dots
        test = test.replace("*", r".*")  # change glob sequence
        test = test.replace("?", r".")  # change glob char
        if re.match(test, host, re.I):
            return True
    return False

def proxy_bypass(host):  # noqa
    """Return True, if the host should be bypassed.

    Checks proxy settings gathered from the environment, if specified,
    or the registry.
    """
    if getproxies_environment():
        return proxy_bypass_environment(host)
    else:
        return proxy_bypass_registry(host)


