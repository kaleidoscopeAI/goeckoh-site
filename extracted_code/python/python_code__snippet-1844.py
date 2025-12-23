def __init__(self, location: str) -> None:
    self.location = location

def as_zipfile(self) -> zipfile.ZipFile:
    return zipfile.ZipFile(self.location, allowZip64=True)


