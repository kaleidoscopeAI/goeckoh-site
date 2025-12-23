"""Returns a set containing the paths that need to be renamed.

This set may include directories when the original sequence of paths
included every file on disk.
"""
case_map = {os.path.normcase(p): p for p in paths}
remaining = set(case_map)
unchecked = sorted({os.path.split(p)[0] for p in case_map.values()}, key=len)
wildcards: Set[str] = set()

def norm_join(*a: str) -> str:
    return os.path.normcase(os.path.join(*a))

for root in unchecked:
    if any(os.path.normcase(root).startswith(w) for w in wildcards):
        # This directory has already been handled.
        continue

    all_files: Set[str] = set()
    all_subdirs: Set[str] = set()
    for dirname, subdirs, files in os.walk(root):
        all_subdirs.update(norm_join(root, dirname, d) for d in subdirs)
        all_files.update(norm_join(root, dirname, f) for f in files)
    # If all the files we found are in our remaining set of files to
    # remove, then remove them from the latter set and add a wildcard
    # for the directory.
    if not (all_files - remaining):
        remaining.difference_update(all_files)
        wildcards.add(root + os.sep)

return set(map(case_map.__getitem__, remaining)) | wildcards


