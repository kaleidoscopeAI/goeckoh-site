"""
This locator uses XML-RPC to locate distributions. It therefore
cannot be used with simple mirrors (that only mirror file content).
"""
def __init__(self, url, **kwargs):
    """
    Initialise an instance.

    :param url: The URL to use for XML-RPC.
    :param kwargs: Passed to the superclass constructor.
    """
    super(PyPIRPCLocator, self).__init__(**kwargs)
    self.base_url = url
    self.client = ServerProxy(url, timeout=3.0)

def get_distribution_names(self):
    """
    Return all the distribution names known to this locator.
    """
    return set(self.client.list_packages())

def _get_project(self, name):
    result = {'urls': {}, 'digests': {}}
    versions = self.client.package_releases(name, True)
    for v in versions:
        urls = self.client.release_urls(name, v)
        data = self.client.release_data(name, v)
        metadata = Metadata(scheme=self.scheme)
        metadata.name = data['name']
        metadata.version = data['version']
        metadata.license = data.get('license')
        metadata.keywords = data.get('keywords', [])
        metadata.summary = data.get('summary')
        dist = Distribution(metadata)
        if urls:
            info = urls[0]
            metadata.source_url = info['url']
            dist.digest = self._get_digest(info)
            dist.locator = self
            result[v] = dist
            for info in urls:
                url = info['url']
                digest = self._get_digest(info)
                result['urls'].setdefault(v, set()).add(url)
                result['digests'][url] = digest
    return result


