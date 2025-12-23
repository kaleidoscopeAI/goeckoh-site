"""
Yield all the uninstallation paths for dist based on RECORD-without-.py[co]

Yield paths to all the files in RECORD. For each .py file in RECORD, add
the .pyc and .pyo in the same directory.

UninstallPathSet.add() takes care of the __pycache__ .py[co].

If RECORD is not found, raises UninstallationError,
with possible information from the INSTALLER file.

https://packaging.python.org/specifications/recording-installed-packages/
"""
location = dist.location
assert location is not None, "not installed"

entries = dist.iter_declared_entries()
if entries is None:
    msg = f"Cannot uninstall {dist}, RECORD file not found."
    installer = dist.installer
    if not installer or installer == "pip":
        dep = f"{dist.raw_name}=={dist.version}"
        msg += (
            " You might be able to recover from this via: "
            f"'pip install --force-reinstall --no-deps {dep}'."
        )
    else:
        msg += f" Hint: The package was installed by {installer}."
    raise UninstallationError(msg)

for entry in entries:
    path = os.path.join(location, entry)
    yield path
    if path.endswith(".py"):
        dn, fn = os.path.split(path)
        base = fn[:-3]
        path = os.path.join(dn, base + ".pyc")
        yield path
        path = os.path.join(dn, base + ".pyo")
        yield path


