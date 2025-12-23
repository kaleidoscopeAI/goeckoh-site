"""
This class allows you to chain and/or merge a list of locators.
"""
def __init__(self, *locators, **kwargs):
    """
    Initialise an instance.

    :param locators: The list of locators to search.
    :param kwargs: Passed to the superclass constructor,
                   except for:
                   * merge - if False (the default), the first successful
                     search from any of the locators is returned. If True,
                     the results from all locators are merged (this can be
                     slow).
    """
    self.merge = kwargs.pop('merge', False)
    self.locators = locators
    super(AggregatingLocator, self).__init__(**kwargs)

def clear_cache(self):
    super(AggregatingLocator, self).clear_cache()
    for locator in self.locators:
        locator.clear_cache()

def _set_scheme(self, value):
    self._scheme = value
    for locator in self.locators:
        locator.scheme = value

scheme = property(Locator.scheme.fget, _set_scheme)

def _get_project(self, name):
    result = {}
    for locator in self.locators:
        d = locator.get_project(name)
        if d:
            if self.merge:
                files = result.get('urls', {})
                digests = result.get('digests', {})
                # next line could overwrite result['urls'], result['digests']
                result.update(d)
                df = result.get('urls')
                if files and df:
                    for k, v in files.items():
                        if k in df:
                            df[k] |= v
                        else:
                            df[k] = v
                dd = result.get('digests')
                if digests and dd:
                    dd.update(digests)
            else:
                # See issue #18. If any dists are found and we're looking
                # for specific constraints, we only return something if
                # a match is found. For example, if a DirectoryLocator
                # returns just foo (1.0) while we're looking for
                # foo (>= 2.0), we'll pretend there was nothing there so
                # that subsequent locators can be queried. Otherwise we
                # would just return foo (1.0) which would then lead to a
                # failure to find foo (>= 2.0), because other locators
                # weren't searched. Note that this only matters when
                # merge=False.
                if self.matcher is None:
                    found = True
                else:
                    found = False
                    for k in d:
                        if self.matcher.match(k):
                            found = True
                            break
                if found:
                    result = d
                    break
    return result

def get_distribution_names(self):
    """
    Return all the distribution names known to this locator.
    """
    result = set()
    for locator in self.locators:
        try:
            result |= locator.get_distribution_names()
        except NotImplementedError:
            pass
    return result


