    from pip._internal.metadata.base import DistributionVersion

    class _DistWithLatestInfo(BaseDistribution):
        """Give the distribution object a couple of extra fields.

        These will be populated during ``get_outdated()``. This is dirty but
        makes the rest of the code much cleaner.
        """

        latest_version: DistributionVersion
        latest_filetype: str

    _ProcessedDists = Sequence[_DistWithLatestInfo]


from pip._vendor.packaging.version import parse

