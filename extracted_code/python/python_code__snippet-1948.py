"""Protection of pip.exe from modification on Windows

On Windows, any operation modifying pip should be run as:
    python -m pip ...
"""
pip_names = [
    "pip",
    f"pip{sys.version_info.major}",
    f"pip{sys.version_info.major}.{sys.version_info.minor}",
]

# See https://github.com/pypa/pip/issues/1299 for more discussion
should_show_use_python_msg = (
    modifying_pip and WINDOWS and os.path.basename(sys.argv[0]) in pip_names
)

if should_show_use_python_msg:
    new_command = [sys.executable, "-m", "pip"] + sys.argv[1:]
    raise CommandError(
        "To modify pip, please run the following command:\n{}".format(
            " ".join(new_command)
        )
    )


