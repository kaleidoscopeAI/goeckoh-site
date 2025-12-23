"""Find destinations for resources files"""

def get_rel_path(root, path):
    # normalizes and returns a lstripped-/-separated path
    root = root.replace(os.path.sep, '/')
    path = path.replace(os.path.sep, '/')
    assert path.startswith(root)
    return path[len(root):].lstrip('/')

destinations = {}
for base, suffix, dest in rules:
    prefix = os.path.join(resources_root, base)
    for abs_base in iglob(prefix):
        abs_glob = os.path.join(abs_base, suffix)
        for abs_path in iglob(abs_glob):
            resource_file = get_rel_path(resources_root, abs_path)
            if dest is None:  # remove the entry if it was here
                destinations.pop(resource_file, None)
            else:
                rel_path = get_rel_path(abs_base, abs_path)
                rel_dest = dest.replace(os.path.sep, '/').rstrip('/')
                destinations[resource_file] = rel_dest + '/' + rel_path
return destinations


