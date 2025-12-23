"""
This locator finds installed distributions in a path. It can be useful for
adding to an :class:`AggregatingLocator`.
"""
def __init__(self, distpath, **kwargs):
    """
    Initialise an instance.

    :param distpath: A :class:`DistributionPath` instance to search.
    """
    super(DistPathLocator, self).__init__(**kwargs)
    assert isinstance(distpath, DistributionPath)
    self.distpath = distpath

def _get_project(self, name):
    dist = self.distpath.get_distribution(name)
    if dist is None:
        result = {'urls': {}, 'digests': {}}
    else:
        result = {
            dist.version: dist,
            'urls': {dist.version: set([dist.source_url])},
            'digests': {dist.version: set([None])}
        }
    return result


