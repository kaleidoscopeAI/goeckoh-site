    from pathlib import Path

    if sys.version_info >= (3, 8):  # pragma: no cover (py38+)
        from typing import Literal
    else:  # pragma: no cover (py38+)
        from pip._vendor.typing_extensions import Literal


def _set_platform_dir_class() -> type[PlatformDirsABC]:
    if sys.platform == "win32":
        from pip._vendor.platformdirs.windows import Windows as Result
    elif sys.platform == "darwin":
        from pip._vendor.platformdirs.macos import MacOS as Result
    else:
        from pip._vendor.platformdirs.unix import Unix as Result

    if os.getenv("ANDROID_DATA") == "/data" and os.getenv("ANDROID_ROOT") == "/system":
        if os.getenv("SHELL") or os.getenv("PREFIX"):
            return Result

        from pip._vendor.platformdirs.android import _android_folder

        if _android_folder() is not None:
            from pip._vendor.platformdirs.android import Android

            return Android  # return to avoid redefinition of result

    return Result


