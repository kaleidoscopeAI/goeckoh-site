@property
def link(self) -> Optional[Link]:
    """Returns the underlying link, if there's one."""
    raise NotImplementedError()

def page_candidates(self) -> FoundCandidates:
    """Candidates found by parsing an archive listing HTML file."""
    raise NotImplementedError()

def file_links(self) -> FoundLinks:
    """Links found by specifying archives directly."""
    raise NotImplementedError()


