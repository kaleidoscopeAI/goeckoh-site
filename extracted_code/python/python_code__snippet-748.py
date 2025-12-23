from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory


def generate_editable_metadata(
    build_env: BuildEnvironment, backend: BuildBackendHookCaller, details: str
