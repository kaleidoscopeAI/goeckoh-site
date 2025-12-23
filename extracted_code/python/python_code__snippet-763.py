def _find_egg_info(directory: str) -> str:
    """Find an .egg-info subdirectory in `directory`."""
    filenames = [f for f in os.listdir(directory) if f.endswith(".egg-info")]

    if not filenames:
        raise InstallationError(f"No .egg-info directory found in {directory}")

    if len(filenames) > 1:
        raise InstallationError(
            "More than one .egg-info directory found in {}".format(directory)
        )

    return os.path.join(directory, filenames[0])


def generate_metadata(
    build_env: BuildEnvironment,
    setup_py_path: str,
    source_dir: str,
    isolated: bool,
    details: str,
