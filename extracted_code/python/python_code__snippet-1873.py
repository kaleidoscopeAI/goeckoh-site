@property
def project_name(self) -> NormalizedName:
    """The "project name" of the candidate.

    This is different from ``name`` if this candidate contains extras,
    in which case ``name`` would contain the ``[...]`` part, while this
    refers to the name of the project.
    """
    raise NotImplementedError("Override in subclass")

@property
def name(self) -> str:
    """The name identifying this candidate in the resolver.

    This is different from ``project_name`` if this candidate contains
    extras, where ``project_name`` would not contain the ``[...]`` part.
    """
    raise NotImplementedError("Override in subclass")

@property
def version(self) -> CandidateVersion:
    raise NotImplementedError("Override in subclass")

@property
def is_installed(self) -> bool:
    raise NotImplementedError("Override in subclass")

@property
def is_editable(self) -> bool:
    raise NotImplementedError("Override in subclass")

@property
def source_link(self) -> Optional[Link]:
    raise NotImplementedError("Override in subclass")

def iter_dependencies(self, with_requires: bool) -> Iterable[Optional[Requirement]]:
    raise NotImplementedError("Override in subclass")

def get_install_requirement(self) -> Optional[InstallRequirement]:
    raise NotImplementedError("Override in subclass")

def format_for_error(self) -> str:
    raise NotImplementedError("Subclass should override")


