def _in_proc_script_path():
    return resources.as_file(
        resources.files(__package__).joinpath('_in_process.py'))


