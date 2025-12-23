_base = re.compile(r"""<base\s+href\s*=\s*['"]?([^'">]+)""", re.I | re.S)

def __init__(self, data, url):
    """
    Initialise an instance with the Unicode page contents and the URL they
    came from.
    """
    self.data = data
    self.base_url = self.url = url
    m = self._base.search(self.data)
    if m:
        self.base_url = m.group(1)

_clean_re = re.compile(r'[^a-z0-9$&+,/:;=?@.#%_\\|-]', re.I)

@cached_property
def links(self):
    """
    Return the URLs of all the links on a page together with information
    about their "rel" attribute, for determining which ones to treat as
    downloads and which ones to queue for further scraping.
    """
    def clean(url):
        "Tidy up an URL."
        scheme, netloc, path, params, query, frag = urlparse(url)
        return urlunparse((scheme, netloc, quote(path),
                           params, query, frag))

    result = set()
    for match in self._href.finditer(self.data):
        d = match.groupdict('')
        rel = (d['rel1'] or d['rel2'] or d['rel3'] or
               d['rel4'] or d['rel5'] or d['rel6'])
        url = d['url1'] or d['url2'] or d['url3']
        url = urljoin(self.base_url, url)
        url = unescape(url)
        url = self._clean_re.sub(lambda m: '%%%2x' % ord(m.group(0)), url)
        result.add((url, rel))
    # We sort the result, hoping to bring the most recent versions
    # to the front
    result = sorted(result, key=lambda t: t[0], reverse=True)
    return result


