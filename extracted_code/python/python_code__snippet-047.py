def _version2fieldlist(version):
    if version == '1.0':
        return _241_FIELDS
    elif version == '1.1':
        return _314_FIELDS
    elif version == '1.2':
        return _345_FIELDS
    elif version in ('1.3', '2.1'):
        # avoid adding field names if already there
        return _345_FIELDS + tuple(f for f in _566_FIELDS if f not in _345_FIELDS)
    elif version == '2.0':
        raise ValueError('Metadata 2.0 is withdrawn and not supported')
        # return _426_FIELDS
    elif version == '2.2':
        return _643_FIELDS
    raise MetadataUnrecognizedVersionError(version)


def _best_version(fields):
    """Detect the best version depending on the fields used."""
    def _has_marker(keys, markers):
        return any(marker in keys for marker in markers)

    keys = [key for key, value in fields.items() if value not in ([], 'UNKNOWN', None)]
    possible_versions = ['1.0', '1.1', '1.2', '1.3', '2.1', '2.2']  # 2.0 removed

    # first let's try to see if a field is not part of one of the version
    for key in keys:
        if key not in _241_FIELDS and '1.0' in possible_versions:
            possible_versions.remove('1.0')
            logger.debug('Removed 1.0 due to %s', key)
        if key not in _314_FIELDS and '1.1' in possible_versions:
            possible_versions.remove('1.1')
            logger.debug('Removed 1.1 due to %s', key)
        if key not in _345_FIELDS and '1.2' in possible_versions:
            possible_versions.remove('1.2')
            logger.debug('Removed 1.2 due to %s', key)
        if key not in _566_FIELDS and '1.3' in possible_versions:
            possible_versions.remove('1.3')
            logger.debug('Removed 1.3 due to %s', key)
        if key not in _566_FIELDS and '2.1' in possible_versions:
            if key != 'Description':  # In 2.1, description allowed after headers
                possible_versions.remove('2.1')
                logger.debug('Removed 2.1 due to %s', key)
        if key not in _643_FIELDS and '2.2' in possible_versions:
            possible_versions.remove('2.2')
            logger.debug('Removed 2.2 due to %s', key)
        # if key not in _426_FIELDS and '2.0' in possible_versions:
            # possible_versions.remove('2.0')
            # logger.debug('Removed 2.0 due to %s', key)

    # possible_version contains qualified versions
    if len(possible_versions) == 1:
        return possible_versions[0]   # found !
    elif len(possible_versions) == 0:
        logger.debug('Out of options - unknown metadata set: %s', fields)
        raise MetadataConflictError('Unknown metadata set')

    # let's see if one unique marker is found
    is_1_1 = '1.1' in possible_versions and _has_marker(keys, _314_MARKERS)
    is_1_2 = '1.2' in possible_versions and _has_marker(keys, _345_MARKERS)
    is_2_1 = '2.1' in possible_versions and _has_marker(keys, _566_MARKERS)
    # is_2_0 = '2.0' in possible_versions and _has_marker(keys, _426_MARKERS)
    is_2_2 = '2.2' in possible_versions and _has_marker(keys, _643_MARKERS)
    if int(is_1_1) + int(is_1_2) + int(is_2_1) + int(is_2_2) > 1:
        raise MetadataConflictError('You used incompatible 1.1/1.2/2.1/2.2 fields')

    # we have the choice, 1.0, or 1.2, 2.1 or 2.2
    #   - 1.0 has a broken Summary field but works with all tools
    #   - 1.1 is to avoid
    #   - 1.2 fixes Summary but has little adoption
    #   - 2.1 adds more features
    #   - 2.2 is the latest
    if not is_1_1 and not is_1_2 and not is_2_1 and not is_2_2:
        # we couldn't find any specific marker
        if PKG_INFO_PREFERRED_VERSION in possible_versions:
            return PKG_INFO_PREFERRED_VERSION
    if is_1_1:
        return '1.1'
    if is_1_2:
        return '1.2'
    if is_2_1:
        return '2.1'
    # if is_2_2:
        # return '2.2'

    return '2.2'

