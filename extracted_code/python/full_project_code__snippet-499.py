"""
Get folder from the registry.

This is a fallback technique at best. I'm not sure if using the registry for these guarantees us the correct answer
for all CSIDL_* names.
"""
shell_folder_name = {
    "CSIDL_APPDATA": "AppData",
    "CSIDL_COMMON_APPDATA": "Common AppData",
    "CSIDL_LOCAL_APPDATA": "Local AppData",
    "CSIDL_PERSONAL": "Personal",
    "CSIDL_DOWNLOADS": "{374DE290-123F-4565-9164-39C4925E467B}",
    "CSIDL_MYPICTURES": "My Pictures",
    "CSIDL_MYVIDEO": "My Video",
    "CSIDL_MYMUSIC": "My Music",
}.get(csidl_name)
if shell_folder_name is None:
    msg = f"Unknown CSIDL name: {csidl_name}"
    raise ValueError(msg)
if sys.platform != "win32":  # only needed for mypy type checker to know that this code runs only on Windows
    raise NotImplementedError
import winreg

key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
directory, _ = winreg.QueryValueEx(key, shell_folder_name)
return str(directory)


