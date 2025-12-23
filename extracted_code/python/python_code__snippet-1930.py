def __init__(self, page: "IndexContent") -> None:
    assert page.cache_link_parsing
    self.page = page

def __eq__(self, other: object) -> bool:
    return isinstance(other, type(self)) and self.page.url == other.page.url

def __hash__(self) -> int:
    return hash(self.page.url)


