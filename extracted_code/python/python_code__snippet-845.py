import os
import sys
from typing import List

from pip._vendor import platformdirs as _appdirs


def user_cache_dir(appname: str) -> str:
    return _appdirs.user_cache_dir(appname, appauthor=False)


def _macos_user_config_dir(appname: str, roaming: bool = True) -> str:
    # Use ~/Application Support/pip, if the directory exists.
    path = _appdirs.user_data_dir(appname, appauthor=False, roaming=roaming)
    if os.path.isdir(path):
        return path

    # Use a Linux-like ~/.config/pip, by default.
    linux_like_path = "~/.config/"
    if appname:
        linux_like_path = os.path.join(linux_like_path, appname)

    return os.path.expanduser(linux_like_path)


def user_config_dir(appname: str, roaming: bool = True) -> str:
    if sys.platform == "darwin":
        return _macos_user_config_dir(appname, roaming)

    return _appdirs.user_config_dir(appname, appauthor=False, roaming=roaming)


