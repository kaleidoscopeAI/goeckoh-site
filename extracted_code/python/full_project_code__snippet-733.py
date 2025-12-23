"""Metadata about a language useful for training models

:ivar name: The human name for the language, in English.
:type name: str
:ivar iso_code: 2-letter ISO 639-1 if possible, 3-letter ISO code otherwise,
                or use another catalog as a last resort.
:type iso_code: str
:ivar use_ascii: Whether or not ASCII letters should be included in trained
                 models.
:type use_ascii: bool
:ivar charsets: The charsets we want to support and create data for.
:type charsets: list of str
:ivar alphabet: The characters in the language's alphabet. If `use_ascii` is
                `True`, you only need to add those not in the ASCII set.
:type alphabet: str
:ivar wiki_start_pages: The Wikipedia pages to start from if we're crawling
                        Wikipedia for training data.
:type wiki_start_pages: list of str
"""

def __init__(
    self,
    name: Optional[str] = None,
    iso_code: Optional[str] = None,
    use_ascii: bool = True,
    charsets: Optional[List[str]] = None,
    alphabet: Optional[str] = None,
    wiki_start_pages: Optional[List[str]] = None,
) -> None:
    super().__init__()
    self.name = name
    self.iso_code = iso_code
    self.use_ascii = use_ascii
    self.charsets = charsets
    if self.use_ascii:
        if alphabet:
            alphabet += ascii_letters
        else:
            alphabet = ascii_letters
    elif not alphabet:
        raise ValueError("Must supply alphabet if use_ascii is False")
    self.alphabet = "".join(sorted(set(alphabet))) if alphabet else None
    self.wiki_start_pages = wiki_start_pages

def __repr__(self) -> str:
    param_str = ", ".join(
        f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
    )
    return f"{self.__class__.__name__}({param_str})"


