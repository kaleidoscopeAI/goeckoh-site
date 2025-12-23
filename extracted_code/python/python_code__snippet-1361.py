from zipfile import ZipExtFile as BaseZipExtFile

class ZipExtFile(BaseZipExtFile):

    def __init__(self, base):
        self.__dict__.update(base.__dict__)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        # return None, so if an exception occurred, it will propagate

class ZipFile(BaseZipFile):

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        # return None, so if an exception occurred, it will propagate

    def open(self, *args, **kwargs):
        base = BaseZipFile.open(self, *args, **kwargs)
        return ZipExtFile(base)


