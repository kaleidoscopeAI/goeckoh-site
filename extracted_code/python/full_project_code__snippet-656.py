"""
decorator to ensure enhancements to a diagram item (such as results name annotations)
get applied on return from _to_diagram_element (we do this since there are several
returns in _to_diagram_element)
"""

def _inner(
    element: pyparsing.ParserElement,
    parent: typing.Optional[EditablePartial],
    lookup: ConverterState = None,
    vertical: int = None,
    index: int = 0,
    name_hint: str = None,
    show_results_names: bool = False,
    show_groups: bool = False,
) -> typing.Optional[EditablePartial]:
    ret = fn(
        element,
        parent,
        lookup,
        vertical,
        index,
        name_hint,
        show_results_names,
        show_groups,
    )

    # apply annotation for results name, if present
    if show_results_names and ret is not None:
        element_results_name = element.resultsName
        if element_results_name:
            # add "*" to indicate if this is a "list all results" name
            element_results_name += "" if element.modalResults else "*"
            ret = EditablePartial.from_call(
                railroad.Group, item=ret, label=element_results_name
            )

    return ret

return _inner


