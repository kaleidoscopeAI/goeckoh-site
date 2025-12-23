"""Flags that map to parsed keys/namespaces."""

# Marks an immutable namespace (inline array or inline table).
FROZEN = 0
# Marks a nest that has been explicitly created and can no longer
# be opened using the "[table]" syntax.
EXPLICIT_NEST = 1

def __init__(self) -> None:
    self._flags: dict[str, dict] = {}
    self._pending_flags: set[tuple[Key, int]] = set()

def add_pending(self, key: Key, flag: int) -> None:
    self._pending_flags.add((key, flag))

def finalize_pending(self) -> None:
    for key, flag in self._pending_flags:
        self.set(key, flag, recursive=False)
    self._pending_flags.clear()

def unset_all(self, key: Key) -> None:
    cont = self._flags
    for k in key[:-1]:
        if k not in cont:
            return
        cont = cont[k]["nested"]
    cont.pop(key[-1], None)

def set(self, key: Key, flag: int, *, recursive: bool) -> None:  # noqa: A003
    cont = self._flags
    key_parent, key_stem = key[:-1], key[-1]
    for k in key_parent:
        if k not in cont:
            cont[k] = {"flags": set(), "recursive_flags": set(), "nested": {}}
        cont = cont[k]["nested"]
    if key_stem not in cont:
        cont[key_stem] = {"flags": set(), "recursive_flags": set(), "nested": {}}
    cont[key_stem]["recursive_flags" if recursive else "flags"].add(flag)

def is_(self, key: Key, flag: int) -> bool:
    if not key:
        return False  # document root has no flags
    cont = self._flags
    for k in key[:-1]:
        if k not in cont:
            return False
        inner_cont = cont[k]
        if flag in inner_cont["recursive_flags"]:
            return True
        cont = inner_cont["nested"]
    key_stem = key[-1]
    if key_stem in cont:
        cont = cont[key_stem]
        return flag in cont["flags"] or flag in cont["recursive_flags"]
    return False


