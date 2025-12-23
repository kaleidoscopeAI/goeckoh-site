def __init__(
    self, src_record_path: RecordPath, dest_path: str, zip_file: ZipFile
) -> None:
    self.src_record_path = src_record_path
    self.dest_path = dest_path
    self._zip_file = zip_file
    self.changed = False

def _getinfo(self) -> ZipInfo:
    return self._zip_file.getinfo(self.src_record_path)

def save(self) -> None:
    # directory creation is lazy and after file filtering
    # to ensure we don't install empty dirs; empty dirs can't be
    # uninstalled.
    parent_dir = os.path.dirname(self.dest_path)
    ensure_dir(parent_dir)

    # When we open the output file below, any existing file is truncated
    # before we start writing the new contents. This is fine in most
    # cases, but can cause a segfault if pip has loaded a shared
    # object (e.g. from pyopenssl through its vendored urllib3)
    # Since the shared object is mmap'd an attempt to call a
    # symbol in it will then cause a segfault. Unlinking the file
    # allows writing of new contents while allowing the process to
    # continue to use the old copy.
    if os.path.exists(self.dest_path):
        os.unlink(self.dest_path)

    zipinfo = self._getinfo()

    with self._zip_file.open(zipinfo) as f:
        with open(self.dest_path, "wb") as dest:
            shutil.copyfileobj(f, dest)

    if zip_item_is_executable(zipinfo):
        set_extracted_file_to_default_mode_plus_executable(self.dest_path)


