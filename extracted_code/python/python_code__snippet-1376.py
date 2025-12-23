def site_config_dirs(appname: str) -> List[str]:
    if sys.platform == "darwin":
        return [_appdirs.site_data_dir(appname, appauthor=False, multipath=True)]

    dirval = _appdirs.site_config_dir(appname, appauthor=False, multipath=True)
    if sys.platform == "win32":
        return [dirval]

    # Unix-y system. Look in /etc as well.
    return dirval.split(os.pathsep) + ["/etc"]


