"""
Given a function that parses an Iterable[Link] from an IndexContent, cache the
function's result (keyed by CacheablePageContent), unless the IndexContent
`page` has `page.cache_link_parsing == False`.
"""

@functools.lru_cache(maxsize=None)
def wrapper(cacheable_page: CacheablePageContent) -> List[Link]:
    return list(fn(cacheable_page.page))

@functools.wraps(fn)
def wrapper_wrapper(page: "IndexContent") -> List[Link]:
    if page.cache_link_parsing:
        return wrapper(CacheablePageContent(page))
    return list(fn(page))

return wrapper_wrapper


