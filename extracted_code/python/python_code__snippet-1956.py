filename = os.path.realpath(filename)
if (
    content_type == "application/zip"
    or filename.lower().endswith(ZIP_EXTENSIONS)
    or zipfile.is_zipfile(filename)
):
    unzip_file(filename, location, flatten=not filename.endswith(".whl"))
elif (
    content_type == "application/x-gzip"
    or tarfile.is_tarfile(filename)
    or filename.lower().endswith(TAR_EXTENSIONS + BZ2_EXTENSIONS + XZ_EXTENSIONS)
):
    untar_file(filename, location)
else:
    # FIXME: handle?
    # FIXME: magic signatures?
    logger.critical(
        "Cannot unpack file %s (downloaded from %s, content-type: %s); "
        "cannot detect archive format",
        filename,
        location,
        content_type,
    )
    raise InstallationError(f"Cannot determine archive format of {location}")


