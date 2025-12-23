"""
Represents a dependency graph between distributions.

The dependency relationships are stored in an ``adjacency_list`` that maps
distributions to a list of ``(other, label)`` tuples where  ``other``
is a distribution and the edge is labeled with ``label`` (i.e. the version
specifier, if such was provided). Also, for more efficient traversal, for
every distribution ``x``, a list of predecessors is kept in
``reverse_list[x]``. An edge from distribution ``a`` to
distribution ``b`` means that ``a`` depends on ``b``. If any missing
dependencies are found, they are stored in ``missing``, which is a
dictionary that maps distributions to a list of requirements that were not
provided by any other distributions.
"""

def __init__(self):
    self.adjacency_list = {}
    self.reverse_list = {}
    self.missing = {}

def add_distribution(self, distribution):
    """Add the *distribution* to the graph.

    :type distribution: :class:`distutils2.database.InstalledDistribution`
                        or :class:`distutils2.database.EggInfoDistribution`
    """
    self.adjacency_list[distribution] = []
    self.reverse_list[distribution] = []
    # self.missing[distribution] = []

def add_edge(self, x, y, label=None):
    """Add an edge from distribution *x* to distribution *y* with the given
    *label*.

    :type x: :class:`distutils2.database.InstalledDistribution` or
             :class:`distutils2.database.EggInfoDistribution`
    :type y: :class:`distutils2.database.InstalledDistribution` or
             :class:`distutils2.database.EggInfoDistribution`
    :type label: ``str`` or ``None``
    """
    self.adjacency_list[x].append((y, label))
    # multiple edges are allowed, so be careful
    if x not in self.reverse_list[y]:
        self.reverse_list[y].append(x)

def add_missing(self, distribution, requirement):
    """
    Add a missing *requirement* for the given *distribution*.

    :type distribution: :class:`distutils2.database.InstalledDistribution`
                        or :class:`distutils2.database.EggInfoDistribution`
    :type requirement: ``str``
    """
    logger.debug('%s missing %r', distribution, requirement)
    self.missing.setdefault(distribution, []).append(requirement)

def _repr_dist(self, dist):
    return '%s %s' % (dist.name, dist.version)

def repr_node(self, dist, level=1):
    """Prints only a subgraph"""
    output = [self._repr_dist(dist)]
    for other, label in self.adjacency_list[dist]:
        dist = self._repr_dist(other)
        if label is not None:
            dist = '%s [%s]' % (dist, label)
        output.append('    ' * level + str(dist))
        suboutput = self.repr_node(other, level + 1)
        subs = suboutput.split('\n')
        output.extend(subs[1:])
    return '\n'.join(output)

def to_dot(self, f, skip_disconnected=True):
    """Writes a DOT output for the graph to the provided file *f*.

    If *skip_disconnected* is set to ``True``, then all distributions
    that are not dependent on any other distribution are skipped.

    :type f: has to support ``file``-like operations
    :type skip_disconnected: ``bool``
    """
    disconnected = []

    f.write("digraph dependencies {\n")
    for dist, adjs in self.adjacency_list.items():
        if len(adjs) == 0 and not skip_disconnected:
            disconnected.append(dist)
        for other, label in adjs:
            if label is not None:
                f.write('"%s" -> "%s" [label="%s"]\n' %
                        (dist.name, other.name, label))
            else:
                f.write('"%s" -> "%s"\n' % (dist.name, other.name))
    if not skip_disconnected and len(disconnected) > 0:
        f.write('subgraph disconnected {\n')
        f.write('label = "Disconnected"\n')
        f.write('bgcolor = red\n')

        for dist in disconnected:
            f.write('"%s"' % dist.name)
            f.write('\n')
        f.write('}\n')
    f.write('}\n')

def topological_sort(self):
    """
    Perform a topological sort of the graph.
    :return: A tuple, the first element of which is a topologically sorted
             list of distributions, and the second element of which is a
             list of distributions that cannot be sorted because they have
             circular dependencies and so form a cycle.
    """
    result = []
    # Make a shallow copy of the adjacency list
    alist = {}
    for k, v in self.adjacency_list.items():
        alist[k] = v[:]
    while True:
        # See what we can remove in this run
        to_remove = []
        for k, v in list(alist.items())[:]:
            if not v:
                to_remove.append(k)
                del alist[k]
        if not to_remove:
            # What's left in alist (if anything) is a cycle.
            break
        # Remove from the adjacency list of others
        for k, v in alist.items():
            alist[k] = [(d, r) for d, r in v if d not in to_remove]
        logger.debug('Moving to result: %s',
                     ['%s (%s)' % (d.name, d.version) for d in to_remove])
        result.extend(to_remove)
    return result, list(alist.keys())

def __repr__(self):
    """Representation of the graph"""
    output = []
    for dist, adjs in self.adjacency_list.items():
        output.append(self.repr_node(dist))
    return '\n'.join(output)


