""":return: documents folder for the Android OS"""
# Get directories with pyjnius
try:
    from jnius import autoclass

    context = autoclass("android.content.Context")
    environment = autoclass("android.os.Environment")
    documents_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DOCUMENTS).getAbsolutePath()
except Exception:  # noqa: BLE001
    documents_dir = "/storage/emulated/0/Documents"

return documents_dir


