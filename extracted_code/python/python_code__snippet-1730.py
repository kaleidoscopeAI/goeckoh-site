# We only want to write to this file, so open it in write only mode
flags = os.O_WRONLY

# os.O_CREAT | os.O_EXCL will fail if the file already exists, so we only
#  will open *new* files.
# We specify this because we want to ensure that the mode we pass is the
# mode of the file.
flags |= os.O_CREAT | os.O_EXCL

# Do not follow symlinks to prevent someone from making a symlink that
# we follow and insecurely open a cache file.
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW

# On Windows we'll mark this file as binary
if hasattr(os, "O_BINARY"):
    flags |= os.O_BINARY

# Before we open our file, we want to delete any existing file that is
# there
try:
    os.remove(filename)
except OSError:
    # The file must not exist already, so we can just skip ahead to opening
    pass

# Open our file, the use of os.O_CREAT | os.O_EXCL will ensure that if a
# race condition happens between the os.remove and this line, that an
# error will be raised. Because we utilize a lockfile this should only
# happen if someone is attempting to attack us.
fd = os.open(filename, flags, fmode)
try:
    return os.fdopen(fd, "wb")

except:
    # An error occurred wrapping our FD in a file object
    os.close(fd)
    raise


