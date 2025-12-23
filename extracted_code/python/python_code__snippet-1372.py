"""
This locator uses PyPI's JSON interface. It's very limited in functionality
and probably not worth using.
"""
def __init__(self, url, **kwargs):
    super(PyPIJSONLocator, self).__init__(**kwargs)
    self.base_url = ensure_slash(url)

def get_distribution_names(self):
    """
    Return all the distribution names known to this locator.
    """
    raise NotImplementedError('Not available from this locator')

def _get_project(self, name):
    result = {'urls': {}, 'digests': {}}
    url = urljoin(self.base_url, '%s/json' % quote(name))
    try:
        resp = self.opener.open(url)
        data = resp.read().decode()  # for now
        d = json.loads(data)
        md = Metadata(scheme=self.scheme)
        data = d['info']
        md.name = data['name']
        md.version = data['version']
        md.license = data.get('license')
        md.keywords = data.get('keywords', [])
        md.summary = data.get('summary')
        dist = Distribution(md)
        dist.locator = self
        # urls = d['urls']
        result[md.version] = dist
        for info in d['urls']:
            url = info['url']
            dist.download_urls.add(url)
            dist.digests[url] = self._get_digest(info)
            result['urls'].setdefault(md.version, set()).add(url)
            result['digests'][url] = self._get_digest(info)
        # Now get other releases
        for version, infos in d['releases'].items():
            if version == md.version:
                continue    # already done
            omd = Metadata(scheme=self.scheme)
            omd.name = md.name
            omd.version = version
            odist = Distribution(omd)
            odist.locator = self
            result[version] = odist
            for info in infos:
                url = info['url']
                odist.download_urls.add(url)
                odist.digests[url] = self._get_digest(info)
                result['urls'].setdefault(version, set()).add(url)
                result['digests'][url] = self._get_digest(info)
