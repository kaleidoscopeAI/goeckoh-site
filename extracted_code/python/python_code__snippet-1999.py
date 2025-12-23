"""
The list from pypi is really a list of versions. We want a list of
packages with the list of versions stored inline. This converts the
list from pypi into one we can use.
"""
packages: Dict[str, "TransformedHit"] = OrderedDict()
for hit in hits:
    name = hit["name"]
    summary = hit["summary"]
    version = hit["version"]

    if name not in packages.keys():
        packages[name] = {
            "name": name,
            "summary": summary,
            "versions": [version],
        }
    else:
        packages[name]["versions"].append(version)

        # if this is the highest version, replace summary and score
        if version == highest_version(packages[name]["versions"]):
            packages[name]["summary"] = summary

return list(packages.values())


