if running_under_virtualenv():
    src_prefix = os.path.join(sys.prefix, "src")
else:
    # FIXME: keep src in cwd for now (it is not a temporary folder)
    try:
        src_prefix = os.path.join(os.getcwd(), "src")
    except OSError:
        # In case the current working directory has been renamed or deleted
        sys.exit("The folder you are executing pip from can no longer be found.")

# under macOS + virtualenv sys.prefix is not properly resolved
# it is something like /path/to/python/bin/..
return os.path.abspath(src_prefix)


