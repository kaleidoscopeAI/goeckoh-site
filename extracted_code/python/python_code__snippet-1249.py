def rehash(path: str, blocksize: int = 1 << 20) -> Tuple[str, str]:
    """Return (encoded_digest, length) for path using hashlib.sha256()"""
    h, length = hash_file(path, blocksize)
    digest = "sha256=" + urlsafe_b64encode(h.digest()).decode("latin1").rstrip("=")
    return (digest, str(length))


def csv_io_kwargs(mode: str) -> Dict[str, Any]:
    """Return keyword arguments to properly open a CSV file
    in the given mode.
    """
    return {"mode": mode, "newline": "", "encoding": "utf-8"}


def fix_script(path: str) -> bool:
    """Replace #!python with #!/path/to/python
    Return True if file was changed.
    """
    # XXX RECORD hashes will need to be updated
    assert os.path.isfile(path)

    with open(path, "rb") as script:
        firstline = script.readline()
        if not firstline.startswith(b"#!python"):
            return False
        exename = sys.executable.encode(sys.getfilesystemencoding())
        firstline = b"#!" + exename + os.linesep.encode("ascii")
        rest = script.read()
    with open(path, "wb") as script:
        script.write(firstline)
        script.write(rest)
    return True


def wheel_root_is_purelib(metadata: Message) -> bool:
    return metadata.get("Root-Is-Purelib", "").lower() == "true"


def get_entrypoints(dist: BaseDistribution) -> Tuple[Dict[str, str], Dict[str, str]]:
    console_scripts = {}
    gui_scripts = {}
    for entry_point in dist.iter_entry_points():
        if entry_point.group == "console_scripts":
            console_scripts[entry_point.name] = entry_point.value
        elif entry_point.group == "gui_scripts":
            gui_scripts[entry_point.name] = entry_point.value
    return console_scripts, gui_scripts


def message_about_scripts_not_on_PATH(scripts: Sequence[str]) -> Optional[str]:
    """Determine if any scripts are not on PATH and format a warning.
    Returns a warning message if one or more scripts are not on PATH,
    otherwise None.
    """
    if not scripts:
        return None

    # Group scripts by the path they were installed in
    grouped_by_dir: Dict[str, Set[str]] = collections.defaultdict(set)
    for destfile in scripts:
        parent_dir = os.path.dirname(destfile)
        script_name = os.path.basename(destfile)
        grouped_by_dir[parent_dir].add(script_name)

    # We don't want to warn for directories that are on PATH.
    not_warn_dirs = [
        os.path.normcase(os.path.normpath(i)).rstrip(os.sep)
        for i in os.environ.get("PATH", "").split(os.pathsep)
    ]
    # If an executable sits with sys.executable, we don't warn for it.
    #     This covers the case of venv invocations without activating the venv.
    not_warn_dirs.append(
        os.path.normcase(os.path.normpath(os.path.dirname(sys.executable)))
    )
    warn_for: Dict[str, Set[str]] = {
        parent_dir: scripts
        for parent_dir, scripts in grouped_by_dir.items()
        if os.path.normcase(os.path.normpath(parent_dir)) not in not_warn_dirs
    }
    if not warn_for:
        return None

    # Format a message
    msg_lines = []
    for parent_dir, dir_scripts in warn_for.items():
        sorted_scripts: List[str] = sorted(dir_scripts)
        if len(sorted_scripts) == 1:
            start_text = f"script {sorted_scripts[0]} is"
        else:
            start_text = "scripts {} are".format(
                ", ".join(sorted_scripts[:-1]) + " and " + sorted_scripts[-1]
            )

        msg_lines.append(
            f"The {start_text} installed in '{parent_dir}' which is not on PATH."
        )

    last_line_fmt = (
        "Consider adding {} to PATH or, if you prefer "
        "to suppress this warning, use --no-warn-script-location."
    )
    if len(msg_lines) == 1:
        msg_lines.append(last_line_fmt.format("this directory"))
    else:
        msg_lines.append(last_line_fmt.format("these directories"))

    # Add a note if any directory starts with ~
    warn_for_tilde = any(
        i[0] == "~" for i in os.environ.get("PATH", "").split(os.pathsep) if i
    )
    if warn_for_tilde:
        tilde_warning_msg = (
            "NOTE: The current PATH contains path(s) starting with `~`, "
            "which may not be expanded by all applications."
        )
        msg_lines.append(tilde_warning_msg)

    # Returns the formatted multiline message
    return "\n".join(msg_lines)


def _normalized_outrows(
    outrows: Iterable[InstalledCSVRow],
