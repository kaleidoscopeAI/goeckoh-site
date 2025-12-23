def req_error_context(req_description: str) -> Generator[None, None, None]:
    try:
        yield
    except InstallationError as e:
        message = f"For req: {req_description}. {e.args[0]}"
        raise InstallationError(message) from e


def install_wheel(
    name: str,
    wheel_path: str,
    scheme: Scheme,
    req_description: str,
    pycompile: bool = True,
    warn_script_location: bool = True,
    direct_url: Optional[DirectUrl] = None,
    requested: bool = False,
