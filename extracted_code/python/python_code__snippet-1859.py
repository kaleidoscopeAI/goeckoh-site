"""
Requirement backed by an install requirement on a base package.
Trims extras from its install requirement if there are any.
"""

def __init__(self, ireq: InstallRequirement) -> None:
    assert ireq.link is None, "This is a link, not a specifier"
    self._ireq = install_req_drop_extras(ireq)
    self._extras = frozenset(canonicalize_name(e) for e in self._ireq.extras)


