"""Returns a key/value dictionary from a CookieJar.

:param cj: CookieJar object to extract cookies from.
:rtype: dict
"""

cookie_dict = {}

for cookie in cj:
    cookie_dict[cookie.name] = cookie.value

return cookie_dict


