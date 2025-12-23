def _is_list_of_str(obj: Any) -> bool:
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)


def make_pyproject_path(unpacked_source_directory: str) -> str:
    return os.path.join(unpacked_source_directory, "pyproject.toml")


