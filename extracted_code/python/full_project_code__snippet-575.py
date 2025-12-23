def makefile(self, mode, bufsize=-1):
    self._makefile_refs += 1
    return _fileobject(self, mode, bufsize, close=True)

