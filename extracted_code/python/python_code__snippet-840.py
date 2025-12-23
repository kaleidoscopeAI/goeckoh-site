            "Cannot unpack file %s (downloaded from %s, content-type: %s); "
            "cannot detect archive format",
            filename,
            location,
            content_type,
        )
        raise InstallationError(f"Cannot determine archive format of {location}")


