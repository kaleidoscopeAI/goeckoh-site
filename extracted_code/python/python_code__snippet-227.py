class InvalidSpecifier(ValueError):
    """
    An invalid specifier was found, users should refer to PEP 440.
    """


class BaseSpecifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns the str representation of this Specifier like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash value for this Specifier like object.
        """

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are equal.
        """

    @abc.abstractproperty
    def prereleases(self) -> Optional[bool]:
        """
        Returns whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        """
        Sets whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @abc.abstractmethod
    def contains(self, item: str, prereleases: Optional[bool] = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """

    @abc.abstractmethod
    def filter(
        self, iterable: Iterable[VersionTypeVar], prereleases: Optional[bool] = None
    ) -> Iterable[VersionTypeVar]:
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class _IndividualSpecifier(BaseSpecifier):

    _operators: Dict[str, str] = {}
    _regex: Pattern[str]

    def __init__(self, spec: str = "", prereleases: Optional[bool] = None) -> None:
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier(f"Invalid specifier: '{spec}'")

        self._spec: Tuple[str, str] = (
            match.group("operator").strip(),
            match.group("version").strip(),
        )

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

    def __repr__(self) -> str:
        pre = (
            f", prereleases={self.prereleases!r}"
            if self._prereleases is not None
            else ""
        )

        return f"<{self.__class__.__name__}({str(self)!r}{pre})>"

    def __str__(self) -> str:
        return "{}{}".format(*self._spec)

    @property
    def _canonical_spec(self) -> Tuple[str, str]:
        return self._spec[0], canonicalize_version(self._spec[1])

    def __hash__(self) -> int:
        return hash(self._canonical_spec)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._canonical_spec == other._canonical_spec

    def _get_operator(self, op: str) -> CallableOperator:
        operator_callable: CallableOperator = getattr(
            self, f"_compare_{self._operators[op]}"
        )
        return operator_callable

    def _coerce_version(self, version: UnparsedVersion) -> ParsedVersion:
        if not isinstance(version, (LegacyVersion, Version)):
            version = parse(version)
        return version

    @property
    def operator(self) -> str:
        return self._spec[0]

    @property
    def version(self) -> str:
        return self._spec[1]

    @property
    def prereleases(self) -> Optional[bool]:
        return self._prereleases

    @prereleases.setter
    def prereleases(self, value: bool) -> None:
        self._prereleases = value

    def __contains__(self, item: str) -> bool:
        return self.contains(item)

    def contains(
        self, item: UnparsedVersion, prereleases: Optional[bool] = None
    ) -> bool:

        # Determine if prereleases are to be allowed or not.
        if prereleases is None:
            prereleases = self.prereleases

        # Normalize item to a Version or LegacyVersion, this allows us to have
        # a shortcut for ``"2.0" in Specifier(">=2")
        normalized_item = self._coerce_version(item)

        # Determine if we should be supporting prereleases in this specifier
        # or not, if we do not support prereleases than we can short circuit
        # logic if this version is a prereleases.
        if normalized_item.is_prerelease and not prereleases:
            return False

        # Actually do the comparison to determine if this item is contained
        # within this Specifier or not.
        operator_callable: CallableOperator = self._get_operator(self.operator)
        return operator_callable(normalized_item, self.version)

    def filter(
        self, iterable: Iterable[VersionTypeVar], prereleases: Optional[bool] = None
    ) -> Iterable[VersionTypeVar]:

        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            parsed_version = self._coerce_version(version)

            if self.contains(parsed_version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later in case nothing
                # else matches this specifier.
                if parsed_version.is_prerelease and not (
                    prereleases or self.prereleases
                ):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the beginning.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


class LegacySpecifier(_IndividualSpecifier):

    _regex_str = r"""
        (?P<operator>(==|!=|<=|>=|<|>))
        \s*
        (?P<version>
            [^,;\s)]* # Since this is a "legacy" specifier, and the version
                      # string can be just about anything, we match everything
                      # except for whitespace, a semi-colon for marker support,
                      # a closing paren since versions can be enclosed in
                      # them, and a comma since it's a version separator.
        )
        """

    _regex = re.compile(r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
    }

    def __init__(self, spec: str = "", prereleases: Optional[bool] = None) -> None:
        super().__init__(spec, prereleases)

        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release",
            DeprecationWarning,
        )

    def _coerce_version(self, version: UnparsedVersion) -> LegacyVersion:
        if not isinstance(version, LegacyVersion):
            version = LegacyVersion(str(version))
        return version

    def _compare_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective == self._coerce_version(spec)

    def _compare_not_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective != self._coerce_version(spec)

    def _compare_less_than_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective <= self._coerce_version(spec)

    def _compare_greater_than_equal(
        self, prospective: LegacyVersion, spec: str
    ) -> bool:
        return prospective >= self._coerce_version(spec)

    def _compare_less_than(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective < self._coerce_version(spec)

    def _compare_greater_than(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective > self._coerce_version(spec)


def _require_version_compare(
    fn: Callable[["Specifier", ParsedVersion, str], bool]
