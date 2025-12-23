@classmethod
def find_spec(self, fullname, path=None, target=None):  # type: ignore
    if fullname != "pip":
        return None

    spec = PathFinder.find_spec(fullname, [PIP_SOURCES_ROOT], target)
    assert spec, (PIP_SOURCES_ROOT, fullname)
    return spec


