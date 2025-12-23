def __init__(self, location: str, stream: IO[bytes]) -> None:
    self.location = location
    self.stream = stream

def as_zipfile(self) -> zipfile.ZipFile:
    return zipfile.ZipFile(self.stream, allowZip64=True)


