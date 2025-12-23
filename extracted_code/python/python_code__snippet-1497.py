_key: Union[CmpKey, LegacyCmpKey]

def __hash__(self) -> int:
    return hash(self._key)

# Please keep the duplicated `isinstance` check
# in the six comparisons hereunder
# unless you find a way to avoid adding overhead function calls.
def __lt__(self, other: "_BaseVersion") -> bool:
    if not isinstance(other, _BaseVersion):
        return NotImplemented

    return self._key < other._key

def __le__(self, other: "_BaseVersion") -> bool:
    if not isinstance(other, _BaseVersion):
        return NotImplemented

    return self._key <= other._key

def __eq__(self, other: object) -> bool:
    if not isinstance(other, _BaseVersion):
        return NotImplemented

    return self._key == other._key

def __ge__(self, other: "_BaseVersion") -> bool:
    if not isinstance(other, _BaseVersion):
        return NotImplemented

    return self._key >= other._key

def __gt__(self, other: "_BaseVersion") -> bool:
    if not isinstance(other, _BaseVersion):
        return NotImplemented

    return self._key > other._key

def __ne__(self, other: object) -> bool:
    if not isinstance(other, _BaseVersion):
        return NotImplemented

    return self._key != other._key


