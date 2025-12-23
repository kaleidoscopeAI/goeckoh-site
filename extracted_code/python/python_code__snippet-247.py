    # Now that we're finished, we can convert from intermediate structures into Railroad elements
    diags = list(lookup.diagrams.values())
    if len(diags) > 1:
        # collapse out duplicate diags with the same name
        seen = set()
        deduped_diags = []
        for d in diags:
            # don't extract SkipTo elements, they are uninformative as subdiagrams
            if d.name == "...":
                continue
            if d.name is not None and d.name not in seen:
                seen.add(d.name)
                deduped_diags.append(d)
        resolved = [resolve_partial(partial) for partial in deduped_diags]
    else:
        # special case - if just one diagram, always display it, even if
        # it has no name
        resolved = [resolve_partial(partial) for partial in diags]
    return sorted(resolved, key=lambda diag: diag.index)


def _should_vertical(
    specification: int, exprs: Iterable[pyparsing.ParserElement]
