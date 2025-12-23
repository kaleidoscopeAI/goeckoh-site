def _coerce_parse_result(results: Union[ParseResults, List[Any]]) -> List[Any]:
    if isinstance(results, ParseResults):
        return [_coerce_parse_result(i) for i in results]
    else:
        return results


def _format_marker(
    marker: Union[List[str], Tuple[Node, ...], str], first: Optional[bool] = True
