"""A decorator that turns an iterable of renderables in to a group.

Args:
    fit (bool, optional): Fit dimension of group to contents, or fill available space. Defaults to True.
"""

def decorator(
    method: Callable[..., Iterable[RenderableType]]
) -> Callable[..., Group]:
    """Convert a method that returns an iterable of renderables in to a Group."""

    @wraps(method)
    def _replace(*args: Any, **kwargs: Any) -> Group:
        renderables = method(*args, **kwargs)
        return Group(*renderables, fit=fit)

    return _replace

return decorator


