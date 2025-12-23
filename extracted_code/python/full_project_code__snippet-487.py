""":return: downloads folder for the Android OS"""
# Get directories with pyjnius
try:
    from jnius import autoclass

    context = autoclass("android.content.Context")
    environment = autoclass("android.os.Environment")
    downloads_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
except Exception:  # noqa: BLE001
    downloads_dir = "/storage/emulated/0/Downloads"

return downloads_dir


