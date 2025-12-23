"""
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


