def site_packages_writable(root: Optional[str], isolated: bool) -> bool:
    return all(
        test_writable_dir(d)
        for d in set(get_lib_location_guesses(root=root, isolated=isolated))
    )


def decide_user_install(
    use_user_site: Optional[bool],
    prefix_path: Optional[str] = None,
    target_dir: Optional[str] = None,
    root_path: Optional[str] = None,
    isolated_mode: bool = False,
