def _open(self) -> TextIOWrapper:
    ensure_dir(os.path.dirname(self.baseFilename))
    return super()._open()


