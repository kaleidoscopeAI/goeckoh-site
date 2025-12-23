def get_fullname(self, filesafe=False):
    """Return the distribution name with version.

    If filesafe is true, return a filename-escaped form."""
    return _get_name_and_version(self['Name'], self['Version'], filesafe)

def is_field(self, name):
    """return True if name is a valid metadata key"""
    name = self._convert_name(name)
    return name in _ALL_FIELDS

def is_multi_field(self, name):
    name = self._convert_name(name)
    return name in _LISTFIELDS

def read(self, filepath):
    """Read the metadata values from a file path."""
    fp = codecs.open(filepath, 'r', encoding='utf-8')
    try:
        self.read_file(fp)
    finally:
        fp.close()

def read_file(self, fileob):
    """Read the metadata values from a file object."""
    msg = message_from_file(fileob)
    self._fields['Metadata-Version'] = msg['metadata-version']

    # When reading, get all the fields we can
    for field in _ALL_FIELDS:
        if field not in msg:
            continue
        if field in _LISTFIELDS:
            # we can have multiple lines
            values = msg.get_all(field)
            if field in _LISTTUPLEFIELDS and values is not None:
                values = [tuple(value.split(',')) for value in values]
            self.set(field, values)
        else:
            # single line
            value = msg[field]
            if value is not None and value != 'UNKNOWN':
                self.set(field, value)

    # PEP 566 specifies that the body be used for the description, if
    # available
    body = msg.get_payload()
    self["Description"] = body if body else self["Description"]
    # logger.debug('Attempting to set metadata for %s', self)
    # self.set_metadata_version()

def write(self, filepath, skip_unknown=False):
    """Write the metadata fields to filepath."""
    fp = codecs.open(filepath, 'w', encoding='utf-8')
    try:
        self.write_file(fp, skip_unknown)
    finally:
        fp.close()

def write_file(self, fileobject, skip_unknown=False):
    """Write the PKG-INFO format data to a file object."""
    self.set_metadata_version()

    for field in _version2fieldlist(self['Metadata-Version']):
        values = self.get(field)
        if skip_unknown and values in ('UNKNOWN', [], ['UNKNOWN']):
            continue
        if field in _ELEMENTSFIELD:
            self._write_field(fileobject, field, ','.join(values))
            continue
        if field not in _LISTFIELDS:
            if field == 'Description':
                if self.metadata_version in ('1.0', '1.1'):
                    values = values.replace('\n', '\n        ')
                else:
                    values = values.replace('\n', '\n       |')
            values = [values]

        if field in _LISTTUPLEFIELDS:
            values = [','.join(value) for value in values]

        for value in values:
            self._write_field(fileobject, field, value)

def update(self, other=None, **kwargs):
    """Set metadata values from the given iterable `other` and kwargs.

    Behavior is like `dict.update`: If `other` has a ``keys`` method,
    they are looped over and ``self[key]`` is assigned ``other[key]``.
    Else, ``other`` is an iterable of ``(key, value)`` iterables.

    Keys that don't match a metadata field or that have an empty value are
    dropped.
    """
    def _set(key, value):
        if key in _ATTR2FIELD and value:
            self.set(self._convert_name(key), value)

    if not other:
        # other is None or empty container
        pass
    elif hasattr(other, 'keys'):
        for k in other.keys():
            _set(k, other[k])
    else:
        for k, v in other:
            _set(k, v)

    if kwargs:
        for k, v in kwargs.items():
            _set(k, v)

def set(self, name, value):
    """Control then set a metadata field."""
    name = self._convert_name(name)

    if ((name in _ELEMENTSFIELD or name == 'Platform') and
        not isinstance(value, (list, tuple))):
        if isinstance(value, string_types):
            value = [v.strip() for v in value.split(',')]
        else:
            value = []
    elif (name in _LISTFIELDS and
          not isinstance(value, (list, tuple))):
        if isinstance(value, string_types):
            value = [value]
        else:
            value = []

    if logger.isEnabledFor(logging.WARNING):
        project_name = self['Name']

        scheme = get_scheme(self.scheme)
        if name in _PREDICATE_FIELDS and value is not None:
            for v in value:
                # check that the values are valid
                if not scheme.is_valid_matcher(v.split(';')[0]):
                    logger.warning(
                        "'%s': '%s' is not valid (field '%s')",
                        project_name, v, name)
        # FIXME this rejects UNKNOWN, is that right?
        elif name in _VERSIONS_FIELDS and value is not None:
            if not scheme.is_valid_constraint_list(value):
                logger.warning("'%s': '%s' is not a valid version (field '%s')",
                               project_name, value, name)
        elif name in _VERSION_FIELDS and value is not None:
            if not scheme.is_valid_version(value):
                logger.warning("'%s': '%s' is not a valid version (field '%s')",
                               project_name, value, name)

    if name in _UNICODEFIELDS:
        if name == 'Description':
            value = self._remove_line_prefix(value)

    self._fields[name] = value

def get(self, name, default=_MISSING):
    """Get a metadata field."""
    name = self._convert_name(name)
    if name not in self._fields:
        if default is _MISSING:
            default = self._default_value(name)
        return default
    if name in _UNICODEFIELDS:
        value = self._fields[name]
        return value
    elif name in _LISTFIELDS:
        value = self._fields[name]
        if value is None:
            return []
        res = []
        for val in value:
            if name not in _LISTTUPLEFIELDS:
                res.append(val)
            else:
                # That's for Project-URL
                res.append((val[0], val[1]))
        return res

    elif name in _ELEMENTSFIELD:
        value = self._fields[name]
        if isinstance(value, string_types):
            return value.split(',')
    return self._fields[name]

def check(self, strict=False):
    """Check if the metadata is compliant. If strict is True then raise if
    no Name or Version are provided"""
    self.set_metadata_version()

    # XXX should check the versions (if the file was loaded)
    missing, warnings = [], []

    for attr in ('Name', 'Version'):  # required by PEP 345
        if attr not in self:
            missing.append(attr)

    if strict and missing != []:
        msg = 'missing required metadata: %s' % ', '.join(missing)
        raise MetadataMissingError(msg)

    for attr in ('Home-page', 'Author'):
        if attr not in self:
            missing.append(attr)

    # checking metadata 1.2 (XXX needs to check 1.1, 1.0)
    if self['Metadata-Version'] != '1.2':
        return missing, warnings

    scheme = get_scheme(self.scheme)

    def are_valid_constraints(value):
        for v in value:
            if not scheme.is_valid_matcher(v.split(';')[0]):
                return False
        return True

    for fields, controller in ((_PREDICATE_FIELDS, are_valid_constraints),
                               (_VERSIONS_FIELDS,
                                scheme.is_valid_constraint_list),
                               (_VERSION_FIELDS,
                                scheme.is_valid_version)):
        for field in fields:
            value = self.get(field, None)
            if value is not None and not controller(value):
                warnings.append("Wrong value for '%s': %s" % (field, value))

    return missing, warnings

def todict(self, skip_missing=False):
    """Return fields as a dict.

    Field names will be converted to use the underscore-lowercase style
    instead of hyphen-mixed case (i.e. home_page instead of Home-page).
    This is as per https://www.python.org/dev/peps/pep-0566/#id17.
    """
    self.set_metadata_version()

    fields = _version2fieldlist(self['Metadata-Version'])

    data = {}

    for field_name in fields:
        if not skip_missing or field_name in self._fields:
            key = _FIELD2ATTR[field_name]
            if key != 'project_url':
                data[key] = self[field_name]
            else:
                data[key] = [','.join(u) for u in self[field_name]]

    return data

def add_requirements(self, requirements):
    if self['Metadata-Version'] == '1.1':
        # we can't have 1.1 metadata *and* Setuptools requires
        for field in ('Obsoletes', 'Requires', 'Provides'):
            if field in self:
                del self[field]
    self['Requires-Dist'] += requirements

# Mapping API
# TODO could add iter* variants

def keys(self):
    return list(_version2fieldlist(self['Metadata-Version']))

def __iter__(self):
    for key in self.keys():
        yield key

def values(self):
    return [self[key] for key in self.keys()]

def items(self):
    return [(key, self[key]) for key in self.keys()]

def __repr__(self):
    return '<%s %s %s>' % (self.__class__.__name__, self.name,
                           self.version)


