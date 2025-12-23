def _exactly_one_of(infos: Iterable[Optional["InfoType"]]) -> "InfoType":
    infos = [info for info in infos if info is not None]
    if not infos:
        raise DirectUrlValidationError(
            "missing one of archive_info, dir_info, vcs_info"
        )
    if len(infos) > 1:
        raise DirectUrlValidationError(
            "more than one of archive_info, dir_info, vcs_info"
        )
    assert infos[0] is not None
    return infos[0]


def _filter_none(**kwargs: Any) -> Dict[str, Any]:
    """Make dict excluding None values."""
    return {k: v for k, v in kwargs.items() if v is not None}


class VcsInfo:
    name = "vcs_info"

    def __init__(
        self,
        vcs: str,
        commit_id: str,
        requested_revision: Optional[str] = None,
    ) -> None:
        self.vcs = vcs
        self.requested_revision = requested_revision
        self.commit_id = commit_id

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["VcsInfo"]:
        if d is None:
            return None
        return cls(
            vcs=_get_required(d, str, "vcs"),
            commit_id=_get_required(d, str, "commit_id"),
            requested_revision=_get(d, str, "requested_revision"),
        )

    def _to_dict(self) -> Dict[str, Any]:
        return _filter_none(
            vcs=self.vcs,
            requested_revision=self.requested_revision,
            commit_id=self.commit_id,
        )


class ArchiveInfo:
    name = "archive_info"

    def __init__(
        self,
        hash: Optional[str] = None,
        hashes: Optional[Dict[str, str]] = None,
    ) -> None:
        # set hashes before hash, since the hash setter will further populate hashes
        self.hashes = hashes
        self.hash = hash

    @property
    def hash(self) -> Optional[str]:
        return self._hash

    @hash.setter
    def hash(self, value: Optional[str]) -> None:
        if value is not None:
            # Auto-populate the hashes key to upgrade to the new format automatically.
            # We don't back-populate the legacy hash key from hashes.
            try:
                hash_name, hash_value = value.split("=", 1)
            except ValueError:
                raise DirectUrlValidationError(
                    f"invalid archive_info.hash format: {value!r}"
                )
            if self.hashes is None:
                self.hashes = {hash_name: hash_value}
            elif hash_name not in self.hashes:
                self.hashes = self.hashes.copy()
                self.hashes[hash_name] = hash_value
        self._hash = value

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["ArchiveInfo"]:
        if d is None:
            return None
        return cls(hash=_get(d, str, "hash"), hashes=_get(d, dict, "hashes"))

    def _to_dict(self) -> Dict[str, Any]:
        return _filter_none(hash=self.hash, hashes=self.hashes)


class DirInfo:
    name = "dir_info"

    def __init__(
        self,
        editable: bool = False,
    ) -> None:
        self.editable = editable

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["DirInfo"]:
        if d is None:
            return None
        return cls(editable=_get_required(d, bool, "editable", default=False))

    def _to_dict(self) -> Dict[str, Any]:
        return _filter_none(editable=self.editable or None)


