"""
HTMLParser that keeps the first base HREF and a list of all anchor
elements' attributes.
"""

def __init__(self, url: str) -> None:
    super().__init__(convert_charrefs=True)

    self.url: str = url
    self.base_url: Optional[str] = None
    self.anchors: List[Dict[str, Optional[str]]] = []

def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
    if tag == "base" and self.base_url is None:
        href = self.get_href(attrs)
        if href is not None:
            self.base_url = href
    elif tag == "a":
        self.anchors.append(dict(attrs))

def get_href(self, attrs: List[Tuple[str, Optional[str]]]) -> Optional[str]:
    for name, value in attrs:
        if name == "href":
            return value
    return None


