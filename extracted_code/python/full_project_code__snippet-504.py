"""Return directory from user-dirs.dirs config file. See https://freedesktop.org/wiki/Software/xdg-user-dirs/."""
user_dirs_config_path = Path(Unix().user_config_dir) / "user-dirs.dirs"
if user_dirs_config_path.exists():
    parser = ConfigParser()

    with user_dirs_config_path.open() as stream:
        # Add fake section header, so ConfigParser doesn't complain
        parser.read_string(f"[top]\n{stream.read()}")

    if key not in parser["top"]:
        return None

    path = parser["top"][key].strip('"')
    # Handle relative home paths
    return path.replace("$HOME", os.path.expanduser("~"))  # noqa: PTH111

return None


