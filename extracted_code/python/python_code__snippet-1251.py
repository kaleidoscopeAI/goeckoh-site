    :param installed: A map from archive RECORD path to installation RECORD
        path.
    """
    installed_rows: List[InstalledCSVRow] = []
    for row in old_csv_rows:
        if len(row) > 3:
            logger.warning("RECORD line has more than three elements: %s", row)
        old_record_path = cast("RecordPath", row[0])
        new_record_path = installed.pop(old_record_path, old_record_path)
        if new_record_path in changed:
            digest, length = rehash(_record_to_fs_path(new_record_path, lib_dir))
        else:
            digest = row[1] if len(row) > 1 else ""
            length = row[2] if len(row) > 2 else ""
        installed_rows.append((new_record_path, digest, length))
    for f in generated:
        path = _fs_to_record_path(f, lib_dir)
        digest, length = rehash(f)
        installed_rows.append((path, digest, length))
    return installed_rows + [
        (installed_record_path, "", "") for installed_record_path in installed.values()
    ]


def get_console_script_specs(console: Dict[str, str]) -> List[str]:
    """
    Given the mapping from entrypoint name to callable, return the relevant
    console script specs.
    """
    # Don't mutate caller's version
    console = console.copy()

    scripts_to_generate = []

    # Special case pip and setuptools to generate versioned wrappers
    #
    # The issue is that some projects (specifically, pip and setuptools) use
    # code in setup.py to create "versioned" entry points - pip2.7 on Python
    # 2.7, pip3.3 on Python 3.3, etc. But these entry points are baked into
    # the wheel metadata at build time, and so if the wheel is installed with
    # a *different* version of Python the entry points will be wrong. The
    # correct fix for this is to enhance the metadata to be able to describe
    # such versioned entry points, but that won't happen till Metadata 2.0 is
    # available.
    # In the meantime, projects using versioned entry points will either have
    # incorrect versioned entry points, or they will not be able to distribute
    # "universal" wheels (i.e., they will need a wheel per Python version).
    #
    # Because setuptools and pip are bundled with _ensurepip and virtualenv,
    # we need to use universal wheels. So, as a stopgap until Metadata 2.0, we
    # override the versioned entry points in the wheel and generate the
    # correct ones. This code is purely a short-term measure until Metadata 2.0
    # is available.
    #
    # To add the level of hack in this section of code, in order to support
    # ensurepip this code will look for an ``ENSUREPIP_OPTIONS`` environment
    # variable which will control which version scripts get installed.
    #
    # ENSUREPIP_OPTIONS=altinstall
    #   - Only pipX.Y and easy_install-X.Y will be generated and installed
    # ENSUREPIP_OPTIONS=install
    #   - pipX.Y, pipX, easy_install-X.Y will be generated and installed. Note
    #     that this option is technically if ENSUREPIP_OPTIONS is set and is
    #     not altinstall
    # DEFAULT
    #   - The default behavior is to install pip, pipX, pipX.Y, easy_install
    #     and easy_install-X.Y.
    pip_script = console.pop("pip", None)
    if pip_script:
        if "ENSUREPIP_OPTIONS" not in os.environ:
            scripts_to_generate.append("pip = " + pip_script)

        if os.environ.get("ENSUREPIP_OPTIONS", "") != "altinstall":
            scripts_to_generate.append(f"pip{sys.version_info[0]} = {pip_script}")

        scripts_to_generate.append(f"pip{get_major_minor_version()} = {pip_script}")
        # Delete any other versioned pip entry points
        pip_ep = [k for k in console if re.match(r"pip(\d+(\.\d+)?)?$", k)]
        for k in pip_ep:
            del console[k]
    easy_install_script = console.pop("easy_install", None)
    if easy_install_script:
        if "ENSUREPIP_OPTIONS" not in os.environ:
            scripts_to_generate.append("easy_install = " + easy_install_script)

        scripts_to_generate.append(
            f"easy_install-{get_major_minor_version()} = {easy_install_script}"
        )
        # Delete any other versioned easy_install entry points
        easy_install_ep = [
            k for k in console if re.match(r"easy_install(-\d+\.\d+)?$", k)
        ]
        for k in easy_install_ep:
            del console[k]

    # Generate the console entry points specified in the wheel
    scripts_to_generate.extend(starmap("{} = {}".format, console.items()))

    return scripts_to_generate


class ZipBackedFile:
    def __init__(
        self, src_record_path: RecordPath, dest_path: str, zip_file: ZipFile
    ) -> None:
        self.src_record_path = src_record_path
        self.dest_path = dest_path
        self._zip_file = zip_file
        self.changed = False

    def _getinfo(self) -> ZipInfo:
        return self._zip_file.getinfo(self.src_record_path)

    def save(self) -> None:
        # directory creation is lazy and after file filtering
        # to ensure we don't install empty dirs; empty dirs can't be
        # uninstalled.
        parent_dir = os.path.dirname(self.dest_path)
        ensure_dir(parent_dir)

        # When we open the output file below, any existing file is truncated
        # before we start writing the new contents. This is fine in most
        # cases, but can cause a segfault if pip has loaded a shared
        # object (e.g. from pyopenssl through its vendored urllib3)
        # Since the shared object is mmap'd an attempt to call a
        # symbol in it will then cause a segfault. Unlinking the file
        # allows writing of new contents while allowing the process to
        # continue to use the old copy.
        if os.path.exists(self.dest_path):
            os.unlink(self.dest_path)

        zipinfo = self._getinfo()

        with self._zip_file.open(zipinfo) as f:
            with open(self.dest_path, "wb") as dest:
                shutil.copyfileobj(f, dest)

        if zip_item_is_executable(zipinfo):
            set_extracted_file_to_default_mode_plus_executable(self.dest_path)


class ScriptFile:
    def __init__(self, file: "File") -> None:
        self._file = file
        self.src_record_path = self._file.src_record_path
        self.dest_path = self._file.dest_path
        self.changed = False

    def save(self) -> None:
        self._file.save()
        self.changed = fix_script(self.dest_path)


class MissingCallableSuffix(InstallationError):
    def __init__(self, entry_point: str) -> None:
        super().__init__(
            f"Invalid script entry point: {entry_point} - A callable "
            "suffix is required. Cf https://packaging.python.org/"
            "specifications/entry-points/#use-for-scripts for more "
            "information."
        )


def _raise_for_invalid_entrypoint(specification: str) -> None:
    entry = get_export_entry(specification)
    if entry is not None and entry.suffix is None:
        raise MissingCallableSuffix(str(entry))


class PipScriptMaker(ScriptMaker):
    def make(
        self, specification: str, options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        _raise_for_invalid_entrypoint(specification)
        return super().make(specification, options)


def _install_wheel(
    name: str,
    wheel_zip: ZipFile,
    wheel_path: str,
    scheme: Scheme,
    pycompile: bool = True,
    warn_script_location: bool = True,
    direct_url: Optional[DirectUrl] = None,
    requested: bool = False,
