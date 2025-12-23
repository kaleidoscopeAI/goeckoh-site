def where() -> str:
    return DEBIAN_CA_CERTS_PATH


def contents() -> str:
    with open(where(), "r", encoding="ascii") as data:
        return data.read()


