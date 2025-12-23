if not _have_compatible_abi(arch):
    return
# Oldest glibc to be supported regardless of architecture is (2, 17).
too_old_glibc2 = _GLibCVersion(2, 16)
if arch in {"x86_64", "i686"}:
    # On x86/i686 also oldest glibc to be supported is (2, 5).
    too_old_glibc2 = _GLibCVersion(2, 4)
current_glibc = _GLibCVersion(*_get_glibc_version())
glibc_max_list = [current_glibc]
# We can assume compatibility across glibc major versions.
# https://sourceware.org/bugzilla/show_bug.cgi?id=24636
#
# Build a list of maximum glibc versions so that we can
# output the canonical list of all glibc from current_glibc
# down to too_old_glibc2, including all intermediary versions.
for glibc_major in range(current_glibc.major - 1, 1, -1):
    glibc_minor = _LAST_GLIBC_MINOR[glibc_major]
    glibc_max_list.append(_GLibCVersion(glibc_major, glibc_minor))
for glibc_max in glibc_max_list:
    if glibc_max.major == too_old_glibc2.major:
        min_minor = too_old_glibc2.minor
    else:
        # For other glibc major versions oldest supported is (x, 0).
        min_minor = -1
    for glibc_minor in range(glibc_max.minor, min_minor, -1):
        glibc_version = _GLibCVersion(glibc_max.major, glibc_minor)
        tag = "manylinux_{}_{}".format(*glibc_version)
        if _is_compatible(tag, arch, glibc_version):
            yield linux.replace("linux", tag)
        # Handle the legacy manylinux1, manylinux2010, manylinux2014 tags.
        if glibc_version in _LEGACY_MANYLINUX_MAP:
            legacy_tag = _LEGACY_MANYLINUX_MAP[glibc_version]
            if _is_compatible(legacy_tag, arch, glibc_version):
                yield linux.replace("linux", legacy_tag)


