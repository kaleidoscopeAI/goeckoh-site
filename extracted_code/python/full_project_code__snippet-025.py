def user_documents_dir() -> str:
    """:returns: documents directory tied to the user"""
    return PlatformDirs().user_documents_dir


def user_downloads_dir() -> str:
    """:returns: downloads directory tied to the user"""
    return PlatformDirs().user_downloads_dir


def user_pictures_dir() -> str:
    """:returns: pictures directory tied to the user"""
    return PlatformDirs().user_pictures_dir


def user_videos_dir() -> str:
    """:returns: videos directory tied to the user"""
    return PlatformDirs().user_videos_dir


def user_music_dir() -> str:
    """:returns: music directory tied to the user"""
    return PlatformDirs().user_music_dir


def user_runtime_dir(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
