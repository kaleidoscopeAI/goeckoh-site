def _record_to_fs_path(record_path: RecordPath, lib_dir: str) -> str:
    return os.path.join(lib_dir, record_path)


def _fs_to_record_path(path: str, lib_dir: str) -> RecordPath:
    # On Windows, do not handle relative paths if they belong to different
    # logical disks
    if os.path.splitdrive(path)[0].lower() == os.path.splitdrive(lib_dir)[0].lower():
        path = os.path.relpath(path, lib_dir)

    path = path.replace(os.path.sep, "/")
    return cast("RecordPath", path)


def get_csv_rows_for_installed(
    old_csv_rows: List[List[str]],
    installed: Dict[RecordPath, RecordPath],
    changed: Set[RecordPath],
    generated: List[str],
    lib_dir: str,
