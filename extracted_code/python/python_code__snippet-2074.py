"""
Abstract base class for commands with the index_group options.

This also corresponds to the commands that permit the pip version check.
"""

def handle_pip_version_check(self, options: Values) -> None:
    """
    Do the pip version check if not disabled.

    This overrides the default behavior of not doing the check.
    """
    # Make sure the index_group options are present.
    assert hasattr(options, "no_index")

    if options.disable_pip_version_check or options.no_index:
        return

    # Otherwise, check if we're using the latest version of pip available.
    session = self._build_session(
        options,
        retries=0,
        timeout=min(5, options.timeout),
        # This is set to ensure the function does not fail when truststore is
        # specified in use-feature but cannot be loaded. This usually raises a
        # CommandError and shows a nice user-facing error, but this function is not
        # called in that try-except block.
        fallback_to_certifi=True,
    )
    with session:
        pip_self_version_check(session, options)


