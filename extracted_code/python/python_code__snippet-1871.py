def __init__(
    self, specifier: SpecifierSet, hashes: Hashes, links: FrozenSet[Link]
) -> None:
    self.specifier = specifier
    self.hashes = hashes
    self.links = links

@classmethod
def empty(cls) -> "Constraint":
    return Constraint(SpecifierSet(), Hashes(), frozenset())

@classmethod
def from_ireq(cls, ireq: InstallRequirement) -> "Constraint":
    links = frozenset([ireq.link]) if ireq.link else frozenset()
    return Constraint(ireq.specifier, ireq.hashes(trust_internet=False), links)

def __bool__(self) -> bool:
    return bool(self.specifier) or bool(self.hashes) or bool(self.links)

def __and__(self, other: InstallRequirement) -> "Constraint":
    if not isinstance(other, InstallRequirement):
        return NotImplemented
    specifier = self.specifier & other.specifier
    hashes = self.hashes & other.hashes(trust_internet=False)
    links = self.links
    if other.link:
        links = links.union([other.link])
    return Constraint(specifier, hashes, links)

def is_satisfied_by(self, candidate: "Candidate") -> bool:
    # Reject if there are any mismatched URL constraints on this package.
    if self.links and not all(_match_link(link, candidate) for link in self.links):
        return False
    # We can safely always allow prereleases here since PackageFinder
    # already implements the prerelease logic, and would have filtered out
    # prerelease candidates if the user does not expect them.
    return self.specifier.contains(candidate.version, prereleases=True)


