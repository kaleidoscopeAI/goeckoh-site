"""Decorator for common logic related to managing temporary
directories.
"""

def configure_tempdir_registry(registry: TempDirectoryTypeRegistry) -> None:
    for t in KEEPABLE_TEMPDIR_TYPES:
        registry.set_delete(t, False)

def wrapper(
    self: RequirementCommand, options: Values, args: List[Any]
) -> Optional[int]:
    assert self.tempdir_registry is not None
    if options.no_clean:
        configure_tempdir_registry(self.tempdir_registry)

    try:
        return func(self, options, args)
    except PreviousBuildDirError:
        # This kind of conflict can occur when the user passes an explicit
        # build directory with a pre-existing folder. In that case we do
        # not want to accidentally remove it.
        configure_tempdir_registry(self.tempdir_registry)
        raise

return wrapper


