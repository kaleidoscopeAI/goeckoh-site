            "Packages installed from PyPI cannot depend on packages "
            "which are not also hosted on PyPI.\n"
            f"{comes_from.name} depends on {req} "
        )

    return InstallRequirement(
        req,
        comes_from,
        isolated=isolated,
        use_pep517=use_pep517,
        user_supplied=user_supplied,
    )


def install_req_from_parsed_requirement(
    parsed_req: ParsedRequirement,
    isolated: bool = False,
    use_pep517: Optional[bool] = None,
    user_supplied: bool = False,
    config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
