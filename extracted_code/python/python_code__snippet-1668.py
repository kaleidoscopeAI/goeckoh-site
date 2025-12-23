"""Cast an object to a renderable by calling __rich__ if present.

Args:
    renderable (object): A potentially renderable object

Returns:
    object: The result of recursively calling __rich__.
"""
from pip._vendor.rich.console import RenderableType

rich_visited_set: Set[type] = set()  # Prevent potential infinite loop
while hasattr(renderable, "__rich__") and not isclass(renderable):
    # Detect object which claim to have all the attributes
    if hasattr(renderable, _GIBBERISH):
        return repr(renderable)
    cast_method = getattr(renderable, "__rich__")
    renderable = cast_method()
    renderable_type = type(renderable)
    if renderable_type in rich_visited_set:
        break
    rich_visited_set.add(renderable_type)

return cast(RenderableType, renderable)


