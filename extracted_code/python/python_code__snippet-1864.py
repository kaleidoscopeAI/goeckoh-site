is_editable = True

def __init__(
    self,
    link: Link,
    template: InstallRequirement,
    factory: "Factory",
    name: Optional[NormalizedName] = None,
    version: Optional[CandidateVersion] = None,
) -> None:
    super().__init__(
        link=link,
        source_link=link,
        ireq=make_install_req_from_editable(link, template),
        factory=factory,
        name=name,
        version=version,
    )

def _prepare_distribution(self) -> BaseDistribution:
    return self._factory.preparer.prepare_editable_requirement(self._ireq)


