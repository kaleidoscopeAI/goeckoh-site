"""Makes a dependency graph from the given distributions.

:parameter dists: a list of distributions
:type dists: list of :class:`distutils2.database.InstalledDistribution` and
             :class:`distutils2.database.EggInfoDistribution` instances
:rtype: a :class:`DependencyGraph` instance
"""
scheme = get_scheme(scheme)
graph = DependencyGraph()
provided = {}  # maps names to lists of (version, dist) tuples

# first, build the graph and find out what's provided
for dist in dists:
    graph.add_distribution(dist)

    for p in dist.provides:
        name, version = parse_name_and_version(p)
        logger.debug('Add to provided: %s, %s, %s', name, version, dist)
        provided.setdefault(name, []).append((version, dist))

# now make the edges
for dist in dists:
    requires = (dist.run_requires | dist.meta_requires
                | dist.build_requires | dist.dev_requires)
    for req in requires:
        try:
            matcher = scheme.matcher(req)
        except UnsupportedVersionError:
            # XXX compat-mode if cannot read the version
            logger.warning('could not read version %r - using name only',
                           req)
            name = req.split()[0]
            matcher = scheme.matcher(name)

        name = matcher.key  # case-insensitive

        matched = False
        if name in provided:
            for version, provider in provided[name]:
                try:
                    match = matcher.match(version)
                except UnsupportedVersionError:
                    match = False

                if match:
                    graph.add_edge(dist, provider, req)
                    matched = True
                    break
        if not matched:
            graph.add_missing(dist, req)
return graph


