def __init__(self) -> None:
    # The parsed content of the TOML document
    self.dict: dict[str, Any] = {}

def get_or_create_nest(
    self,
    key: Key,
    *,
    access_lists: bool = True,
) -> dict:
    cont: Any = self.dict
    for k in key:
        if k not in cont:
            cont[k] = {}
        cont = cont[k]
        if access_lists and isinstance(cont, list):
            cont = cont[-1]
        if not isinstance(cont, dict):
            raise KeyError("There is no nest behind this key")
    return cont

def append_nest_to_list(self, key: Key) -> None:
    cont = self.get_or_create_nest(key[:-1])
    last_key = key[-1]
    if last_key in cont:
        list_ = cont[last_key]
        if not isinstance(list_, list):
            raise KeyError("An object other than list found behind this key")
        list_.append({})
    else:
        cont[last_key] = [{}]


