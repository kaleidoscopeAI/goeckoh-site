"""``--[extra-]index-url=<path-to-directory>``.

This is treated like a remote URL; ``candidates_from_page`` contains logic
for this by appending ``index.html`` to the link.
"""

def __init__(
    self,
    candidates_from_page: CandidatesFromPage,
    link: Link,
) -> None:
    self._candidates_from_page = candidates_from_page
    self._link = link

@property
def link(self) -> Optional[Link]:
    return self._link

def page_candidates(self) -> FoundCandidates:
    yield from self._candidates_from_page(self._link)

def file_links(self) -> FoundLinks:
    return ()


