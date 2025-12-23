def has_metadata(name):
    """Does the package's distribution contain the named metadata?"""

def get_metadata(name):
    """The named metadata resource as a string"""

def get_metadata_lines(name):
    """Yield named metadata resource as list of non-blank non-comment lines

    Leading and trailing whitespace is stripped from each line, and lines
    with ``#`` as the first non-blank character are omitted."""

def metadata_isdir(name):
    """Is the named metadata a directory?  (like ``os.path.isdir()``)"""

def metadata_listdir(name):
    """List of metadata names in the directory (like ``os.listdir()``)"""

def run_script(script_name, namespace):
    """Execute the named script in the supplied namespace dictionary"""


