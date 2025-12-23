def user_documents_path() -> Path:
    """:returns: documents path tied to the user"""
    return PlatformDirs().user_documents_path


def user_downloads_path() -> Path:
    """:returns: downloads path tied to the user"""
    return PlatformDirs().user_downloads_path


def user_pictures_path() -> Path:
    """:returns: pictures path tied to the user"""
    return PlatformDirs().user_pictures_path


def user_videos_path() -> Path:
    """:returns: videos path tied to the user"""
    return PlatformDirs().user_videos_path


def user_music_path() -> Path:
    """:returns: music path tied to the user"""
    return PlatformDirs().user_music_path


def user_runtime_path(
    appname: str | None = None,
    appauthor: str | None | Literal[False] = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
