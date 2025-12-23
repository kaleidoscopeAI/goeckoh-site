def install_warning_logger() -> None:
    # Enable our Deprecation Warnings
    warnings.simplefilter("default", PipDeprecationWarning, append=True)

    global _original_showwarning

    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning
        warnings.showwarning = _showwarning


def deprecated(
    *,
    reason: str,
    replacement: Optional[str],
    gone_in: Optional[str],
    feature_flag: Optional[str] = None,
    issue: Optional[int] = None,
