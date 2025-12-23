"""
This locator uses special extended metadata (not available on PyPI) and is
the basis of performant dependency resolution in distlib. Other locators
require archive downloads before dependencies can be determined! As you
might imagine, that can be slow.
"""
def get_distribution_names(self):
    """
    Return all the distribution names known to this locator.
    """
    raise NotImplementedError('Not available from this locator')

def _get_project(self, name):
    result = {'urls': {}, 'digests': {}}
    data = get_project_data(name)
    if data:
        for info in data.get('files', []):
            if info['ptype'] != 'sdist' or info['pyversion'] != 'source':
                continue
            # We don't store summary in project metadata as it makes
            # the data bigger for no benefit during dependency
            # resolution
            dist = make_dist(data['name'], info['version'],
                             summary=data.get('summary',
                                              'Placeholder for summary'),
                             scheme=self.scheme)
            md = dist.metadata
            md.source_url = info['url']
            # TODO SHA256 digest
            if 'digest' in info and info['digest']:
                dist.digest = ('md5', info['digest'])
            md.dependencies = info.get('requirements', {})
            dist.exports = info.get('exports', {})
            result[dist.version] = dist
            result['urls'].setdefault(dist.version, set()).add(info['url'])
    return result


