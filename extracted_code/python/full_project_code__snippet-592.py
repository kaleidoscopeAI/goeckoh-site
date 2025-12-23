def replacer(match):
    return needles_and_replacements[match.group(0)]

pattern = re.compile(
    r"|".join([re.escape(needle) for needle in needles_and_replacements.keys()])
)

result = pattern.sub(replacer, value)

return result


