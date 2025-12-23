"""decorator to trim function calls to match the arity of the target"""
global _trim_arity_call_line

if func in _single_arg_builtins:
    return lambda s, l, t: func(t)

limit = 0
found_arity = False

# synthesize what would be returned by traceback.extract_stack at the call to
# user's parse action 'func', so that we don't incur call penalty at parse time

# fmt: off
LINE_DIFF = 7
# IF ANY CODE CHANGES, EVEN JUST COMMENTS OR BLANK LINES, BETWEEN THE NEXT LINE AND
# THE CALL TO FUNC INSIDE WRAPPER, LINE_DIFF MUST BE MODIFIED!!!!
_trim_arity_call_line = (_trim_arity_call_line or traceback.extract_stack(limit=2)[-1])
pa_call_line_synth = (_trim_arity_call_line[0], _trim_arity_call_line[1] + LINE_DIFF)

def wrapper(*args):
    nonlocal found_arity, limit
    while 1:
        try:
            ret = func(*args[limit:])
            found_arity = True
            return ret
        except TypeError as te:
            # re-raise TypeErrors if they did not come from our arity testing
            if found_arity:
                raise
            else:
                tb = te.__traceback__
                frames = traceback.extract_tb(tb, limit=2)
                frame_summary = frames[-1]
                trim_arity_type_error = (
                    [frame_summary[:2]][-1][:2] == pa_call_line_synth
                )
                del tb

                if trim_arity_type_error:
                    if limit < max_limit:
                        limit += 1
                        continue

                raise
# fmt: on

# copy func name to wrapper for sensible debug output
# (can't use functools.wraps, since that messes with function signature)
func_name = getattr(func, "__name__", getattr(func, "__class__").__name__)
wrapper.__name__ = func_name
wrapper.__doc__ = func.__doc__

return wrapper


