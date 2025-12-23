ldversion = sysconfig.get_config_var("LDVERSION")
abiflags = getattr(sys, "abiflags", None)

# LDVERSION does not end with sys.abiflags. Just return the path unchanged.
if not ldversion or not abiflags or not ldversion.endswith(abiflags):
    yield from parts
    return

# Strip sys.abiflags from LDVERSION-based path components.
for part in parts:
    if part.endswith(ldversion):
        part = part[: (0 - len(abiflags))]
    yield part


