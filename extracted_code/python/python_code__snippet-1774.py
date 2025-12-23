"""An object that provides access to package resources"""

def get_resource_filename(manager, resource_name):
    """Return a true filesystem path for `resource_name`

    `manager` must be an ``IResourceManager``"""

def get_resource_stream(manager, resource_name):
    """Return a readable file-like object for `resource_name`

    `manager` must be an ``IResourceManager``"""

def get_resource_string(manager, resource_name):
    """Return a string containing the contents of `resource_name`

    `manager` must be an ``IResourceManager``"""

def has_resource(resource_name):
    """Does the package contain the named resource?"""

def resource_isdir(resource_name):
    """Is the named resource a directory?  (like ``os.path.isdir()``)"""

def resource_listdir(resource_name):
    """List of resource names in the directory (like ``os.listdir()``)"""


