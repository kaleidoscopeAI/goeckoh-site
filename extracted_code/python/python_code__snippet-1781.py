"""Try to implement resources and metadata for arbitrary PEP 302 loaders"""

egg_name = None
egg_info = None
loader = None

def __init__(self, module):
    self.loader = getattr(module, '__loader__', None)
    self.module_path = os.path.dirname(getattr(module, '__file__', ''))

def get_resource_filename(self, manager, resource_name):
    return self._fn(self.module_path, resource_name)

def get_resource_stream(self, manager, resource_name):
    return io.BytesIO(self.get_resource_string(manager, resource_name))

def get_resource_string(self, manager, resource_name):
    return self._get(self._fn(self.module_path, resource_name))

def has_resource(self, resource_name):
    return self._has(self._fn(self.module_path, resource_name))

def _get_metadata_path(self, name):
    return self._fn(self.egg_info, name)

def has_metadata(self, name):
    if not self.egg_info:
        return self.egg_info

    path = self._get_metadata_path(name)
    return self._has(path)

def get_metadata(self, name):
    if not self.egg_info:
        return ""
    path = self._get_metadata_path(name)
    value = self._get(path)
    try:
        return value.decode('utf-8')
    except UnicodeDecodeError as exc:
        # Include the path in the error message to simplify
        # troubleshooting, and without changing the exception type.
        exc.reason += ' in {} file at path: {}'.format(name, path)
        raise

def get_metadata_lines(self, name):
    return yield_lines(self.get_metadata(name))

def resource_isdir(self, resource_name):
    return self._isdir(self._fn(self.module_path, resource_name))

def metadata_isdir(self, name):
    return self.egg_info and self._isdir(self._fn(self.egg_info, name))

def resource_listdir(self, resource_name):
    return self._listdir(self._fn(self.module_path, resource_name))

def metadata_listdir(self, name):
    if self.egg_info:
        return self._listdir(self._fn(self.egg_info, name))
    return []

def run_script(self, script_name, namespace):
    script = 'scripts/' + script_name
    if not self.has_metadata(script):
        raise ResolutionError(
            "Script {script!r} not found in metadata at {self.egg_info!r}".format(
                **locals()
            ),
        )
    script_text = self.get_metadata(script).replace('\r\n', '\n')
    script_text = script_text.replace('\r', '\n')
    script_filename = self._fn(self.egg_info, script)
    namespace['__file__'] = script_filename
    if os.path.exists(script_filename):
        with open(script_filename) as fid:
            source = fid.read()
        code = compile(source, script_filename, 'exec')
        exec(code, namespace, namespace)
    else:
        from linecache import cache

        cache[script_filename] = (
            len(script_text),
            0,
            script_text.split('\n'),
            script_filename,
        )
        script_code = compile(script_text, script_filename, 'exec')
        exec(script_code, namespace, namespace)

def _has(self, path):
    raise NotImplementedError(
        "Can't perform this operation for unregistered loader type"
    )

def _isdir(self, path):
    raise NotImplementedError(
        "Can't perform this operation for unregistered loader type"
    )

def _listdir(self, path):
    raise NotImplementedError(
        "Can't perform this operation for unregistered loader type"
    )

def _fn(self, base, resource_name):
    self._validate_resource_path(resource_name)
    if resource_name:
        return os.path.join(base, *resource_name.split('/'))
    return base

@staticmethod
def _validate_resource_path(path):
    """
    Validate the resource paths according to the docs.
    https://setuptools.pypa.io/en/latest/pkg_resources.html#basic-resource-access

    >>> warned = getfixture('recwarn')
    >>> warnings.simplefilter('always')
    >>> vrp = NullProvider._validate_resource_path
    >>> vrp('foo/bar.txt')
    >>> bool(warned)
    False
    >>> vrp('../foo/bar.txt')
    >>> bool(warned)
    True
    >>> warned.clear()
    >>> vrp('/foo/bar.txt')
    >>> bool(warned)
    True
    >>> vrp('foo/../../bar.txt')
    >>> bool(warned)
    True
    >>> warned.clear()
    >>> vrp('foo/f../bar.txt')
    >>> bool(warned)
    False

    Windows path separators are straight-up disallowed.
    >>> vrp(r'\\foo/bar.txt')
    Traceback (most recent call last):
    ...
    ValueError: Use of .. or absolute path in a resource path \
