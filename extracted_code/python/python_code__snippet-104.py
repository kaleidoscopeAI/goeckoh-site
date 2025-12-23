def _is_version_marker(s):
    return isinstance(s, string_types) and s in _VERSION_MARKERS


def _is_literal(o):
    if not isinstance(o, string_types) or not o:
        return False
    return o[0] in '\'"'


def _get_versions(s):
    return {LV(m.groups()[0]) for m in _VERSION_PATTERN.finditer(s)}


class Evaluator(object):
    """
    This class is used to evaluate marker expressions.
    """

    operations = {
        '==': lambda x, y: x == y,
        '===': lambda x, y: x == y,
        '~=': lambda x, y: x == y or x > y,
        '!=': lambda x, y: x != y,
        '<': lambda x, y: x < y,
        '<=': lambda x, y: x == y or x < y,
        '>': lambda x, y: x > y,
        '>=': lambda x, y: x == y or x > y,
        'and': lambda x, y: x and y,
        'or': lambda x, y: x or y,
        'in': lambda x, y: x in y,
        'not in': lambda x, y: x not in y,
    }

    def evaluate(self, expr, context):
        """
        Evaluate a marker expression returned by the :func:`parse_requirement`
        function in the specified context.
        """
        if isinstance(expr, string_types):
            if expr[0] in '\'"':
                result = expr[1:-1]
            else:
                if expr not in context:
                    raise SyntaxError('unknown variable: %s' % expr)
                result = context[expr]
        else:
            assert isinstance(expr, dict)
            op = expr['op']
            if op not in self.operations:
                raise NotImplementedError('op not implemented: %s' % op)
            elhs = expr['lhs']
            erhs = expr['rhs']
            if _is_literal(expr['lhs']) and _is_literal(expr['rhs']):
                raise SyntaxError('invalid comparison: %s %s %s' %
                                  (elhs, op, erhs))

            lhs = self.evaluate(elhs, context)
            rhs = self.evaluate(erhs, context)
            if ((_is_version_marker(elhs) or _is_version_marker(erhs))
                    and op in ('<', '<=', '>', '>=', '===', '==', '!=', '~=')):
                lhs = LV(lhs)
                rhs = LV(rhs)
            elif _is_version_marker(elhs) and op in ('in', 'not in'):
                lhs = LV(lhs)
                rhs = _get_versions(rhs)
            result = self.operations[op](lhs, rhs)
        return result


