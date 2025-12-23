class include(str):  # pylint: disable=invalid-name
    """
    Indicates that a state should include rules from another state.
    """
    pass


class _inherit:
    """
    Indicates the a state should inherit from its superclass.
    """
    def __repr__(self):
        return 'inherit'

