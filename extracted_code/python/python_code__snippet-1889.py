def __init__(self, file: "File") -> None:
    self._file = file
    self.src_record_path = self._file.src_record_path
    self.dest_path = self._file.dest_path
    self.changed = False

def save(self) -> None:
    self._file.save()
    self.changed = fix_script(self.dest_path)


