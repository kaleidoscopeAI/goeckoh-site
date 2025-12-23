def _get_prepared_distribution(
    req: InstallRequirement,
    build_tracker: BuildTracker,
    finder: PackageFinder,
    build_isolation: bool,
    check_build_deps: bool,
