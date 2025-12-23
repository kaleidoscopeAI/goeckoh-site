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


