"""Like os.renames(), but handles renaming across devices."""
# Implementation borrowed from os.renames().
head, tail = os.path.split(new)
if head and tail and not os.path.exists(head):
    os.makedirs(head)

shutil.move(old, new)

head, tail = os.path.split(old)
if head and tail:
    try:
        os.removedirs(head)
    except OSError:
        pass


