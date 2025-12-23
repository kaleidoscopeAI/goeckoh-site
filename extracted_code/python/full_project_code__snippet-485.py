""":return: base folder for the Android OS or None if it cannot be found"""
try:
    # First try to get path to android app via pyjnius
    from jnius import autoclass

    context = autoclass("android.content.Context")
    result: str | None = context.getFilesDir().getParentFile().getAbsolutePath()
except Exception:  # noqa: BLE001
    # if fails find an android folder looking path on the sys.path
    pattern = re.compile(r"/data/(data|user/\d+)/(.+)/files")
    for path in sys.path:
        if pattern.match(path):
            result = path.split("/files")[0]
            break
    else:
        result = None
return result


