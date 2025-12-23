class RemoteNotFoundError(Exception):
    pass


class RemoteNotValidError(Exception):
    def __init__(self, url: str):
        super().__init__(url)
        self.url = url


class RevOptions:

    """
    Encapsulates a VCS-specific revision to install, along with any VCS
    install options.

    Instances of this class should be treated as if immutable.
    """

    def __init__(
        self,
        vc_class: Type["VersionControl"],
        rev: Optional[str] = None,
        extra_args: Optional[CommandArgs] = None,
    ) -> None:
        """
        Args:
          vc_class: a VersionControl subclass.
          rev: the name of the revision to install.
          extra_args: a list of extra options.
        """
        if extra_args is None:
            extra_args = []

        self.extra_args = extra_args
        self.rev = rev
        self.vc_class = vc_class
        self.branch_name: Optional[str] = None

    def __repr__(self) -> str:
        return f"<RevOptions {self.vc_class.name}: rev={self.rev!r}>"

    @property
    def arg_rev(self) -> Optional[str]:
        if self.rev is None:
            return self.vc_class.default_arg_rev

        return self.rev

    def to_args(self) -> CommandArgs:
        """
        Return the VCS-specific command arguments.
        """
        args: CommandArgs = []
        rev = self.arg_rev
        if rev is not None:
            args += self.vc_class.get_base_rev_args(rev)
        args += self.extra_args

        return args

    def to_display(self) -> str:
        if not self.rev:
            return ""

        return f" (to revision {self.rev})"

    def make_new(self, rev: str) -> "RevOptions":
        """
        Make a copy of the current instance, but with a new rev.

        Args:
          rev: the name of the revision for the new object.
        """
        return self.vc_class.make_rev_options(rev, extra_args=self.extra_args)


class VcsSupport:
    _registry: Dict[str, "VersionControl"] = {}
    schemes = ["ssh", "git", "hg", "bzr", "sftp", "svn"]

    def __init__(self) -> None:
        # Register more schemes with urlparse for various version control
        # systems
        urllib.parse.uses_netloc.extend(self.schemes)
        super().__init__()

    def __iter__(self) -> Iterator[str]:
        return self._registry.__iter__()

    @property
    def backends(self) -> List["VersionControl"]:
        return list(self._registry.values())

    @property
    def dirnames(self) -> List[str]:
        return [backend.dirname for backend in self.backends]

    @property
    def all_schemes(self) -> List[str]:
        schemes: List[str] = []
        for backend in self.backends:
            schemes.extend(backend.schemes)
        return schemes

    def register(self, cls: Type["VersionControl"]) -> None:
        if not hasattr(cls, "name"):
            logger.warning("Cannot register VCS %s", cls.__name__)
            return
        if cls.name not in self._registry:
            self._registry[cls.name] = cls()
            logger.debug("Registered VCS backend: %s", cls.name)

    def unregister(self, name: str) -> None:
        if name in self._registry:
            del self._registry[name]

    def get_backend_for_dir(self, location: str) -> Optional["VersionControl"]:
        """
        Return a VersionControl object if a repository of that type is found
        at the given directory.
        """
        vcs_backends = {}
        for vcs_backend in self._registry.values():
            repo_path = vcs_backend.get_repository_root(location)
            if not repo_path:
                continue
            logger.debug("Determine that %s uses VCS: %s", location, vcs_backend.name)
            vcs_backends[repo_path] = vcs_backend

        if not vcs_backends:
            return None

        # Choose the VCS in the inner-most directory. Since all repository
        # roots found here would be either `location` or one of its
        # parents, the longest path should have the most path components,
        # i.e. the backend representing the inner-most repository.
        inner_most_repo_path = max(vcs_backends, key=len)
        return vcs_backends[inner_most_repo_path]

    def get_backend_for_scheme(self, scheme: str) -> Optional["VersionControl"]:
        """
        Return a VersionControl object or None.
        """
        for vcs_backend in self._registry.values():
            if scheme in vcs_backend.schemes:
                return vcs_backend
        return None

    def get_backend(self, name: str) -> Optional["VersionControl"]:
        """
        Return a VersionControl object or None.
        """
        name = name.lower()
        return self._registry.get(name)


