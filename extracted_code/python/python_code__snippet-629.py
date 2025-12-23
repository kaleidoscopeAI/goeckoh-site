def unarchive(archive_filename, dest_dir, format=None, check=True):

    def check_path(path):
        if not isinstance(path, text_type):
            path = path.decode('utf-8')
        p = os.path.abspath(os.path.join(dest_dir, path))
        if not p.startswith(dest_dir) or p[plen] != os.sep:
            raise ValueError('path outside destination: %r' % p)

    dest_dir = os.path.abspath(dest_dir)
    plen = len(dest_dir)
    archive = None
    if format is None:
        if archive_filename.endswith(('.zip', '.whl')):
            format = 'zip'
        elif archive_filename.endswith(('.tar.gz', '.tgz')):
            format = 'tgz'
            mode = 'r:gz'
        elif archive_filename.endswith(('.tar.bz2', '.tbz')):
            format = 'tbz'
            mode = 'r:bz2'
        elif archive_filename.endswith('.tar'):
            format = 'tar'
            mode = 'r'
        else:  # pragma: no cover
            raise ValueError('Unknown format for %r' % archive_filename)
    try:
        if format == 'zip':
            archive = ZipFile(archive_filename, 'r')
            if check:
                names = archive.namelist()
                for name in names:
                    check_path(name)
        else:
            archive = tarfile.open(archive_filename, mode)
            if check:
                names = archive.getnames()
                for name in names:
                    check_path(name)
        if format != 'zip' and sys.version_info[0] < 3:
            # See Python issue 17153. If the dest path contains Unicode,
            # tarfile extraction fails on Python 2.x if a member path name
            # contains non-ASCII characters - it leads to an implicit
            # bytes -> unicode conversion using ASCII to decode.
            for tarinfo in archive.getmembers():
                if not isinstance(tarinfo.name, text_type):
                    tarinfo.name = tarinfo.name.decode('utf-8')

        # Limit extraction of dangerous items, if this Python
        # allows it easily. If not, just trust the input.
        # See: https://docs.python.org/3/library/tarfile.html#extraction-filters
        def extraction_filter(member, path):
            """Run tarfile.tar_filter, but raise the expected ValueError"""
            # This is only called if the current Python has tarfile filters
            try:
                return tarfile.tar_filter(member, path)
            except tarfile.FilterError as exc:
                raise ValueError(str(exc))

        archive.extraction_filter = extraction_filter

        archive.extractall(dest_dir)

    finally:
        if archive:
            archive.close()


def zip_dir(directory):
    """zip a directory tree into a BytesIO object"""
    result = io.BytesIO()
    dlen = len(directory)
    with ZipFile(result, "w") as zf:
        for root, dirs, files in os.walk(directory):
            for name in files:
                full = os.path.join(root, name)
                rel = root[dlen:]
                dest = os.path.join(rel, name)
                zf.write(full, dest)
    return result


