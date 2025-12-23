class ConfiguredBuildBackendHookCaller(BuildBackendHookCaller):
    def __init__(
        self,
        config_holder: Any,
        source_dir: str,
        build_backend: str,
        backend_path: Optional[str] = None,
        runner: Optional[Callable[..., None]] = None,
        python_executable: Optional[str] = None,
    ):
        super().__init__(
            source_dir, build_backend, backend_path, runner, python_executable
        )
        self.config_holder = config_holder

    def build_wheel(
        self,
        wheel_directory: str,
        config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().build_wheel(
            wheel_directory, config_settings=cs, metadata_directory=metadata_directory
        )

    def build_sdist(
        self,
        sdist_directory: str,
        config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().build_sdist(sdist_directory, config_settings=cs)

    def build_editable(
        self,
        wheel_directory: str,
        config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().build_editable(
            wheel_directory, config_settings=cs, metadata_directory=metadata_directory
        )

    def get_requires_for_build_wheel(
        self, config_settings: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> List[str]:
        cs = self.config_holder.config_settings
        return super().get_requires_for_build_wheel(config_settings=cs)

    def get_requires_for_build_sdist(
        self, config_settings: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> List[str]:
        cs = self.config_holder.config_settings
        return super().get_requires_for_build_sdist(config_settings=cs)

    def get_requires_for_build_editable(
        self, config_settings: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> List[str]:
        cs = self.config_holder.config_settings
        return super().get_requires_for_build_editable(config_settings=cs)

    def prepare_metadata_for_build_wheel(
        self,
        metadata_directory: str,
        config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
        _allow_fallback: bool = True,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().prepare_metadata_for_build_wheel(
            metadata_directory=metadata_directory,
            config_settings=cs,
            _allow_fallback=_allow_fallback,
        )

    def prepare_metadata_for_build_editable(
        self,
        metadata_directory: str,
        config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
        _allow_fallback: bool = True,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().prepare_metadata_for_build_editable(
            metadata_directory=metadata_directory,
            config_settings=cs,
            _allow_fallback=_allow_fallback,
        )


