class _FinalForm(_ExtensionsSpecialForm, _root=True):
    def __getitem__(self, parameters):
        item = typing._type_check(parameters,
                                  f'{self._name} accepts only a single type.')
        return typing._GenericAlias(self, (item,))

Final = _FinalForm('Final',
                   doc="""A special typing construct to indicate that a name
                   cannot be re-assigned or overridden in a subclass.
                   For example:

                       MAX_SIZE: Final = 9000
                       MAX_SIZE += 1  # Error reported by type checker

                       class Connection:
                           TIMEOUT: Final[int] = 10
                       class FastConnector(Connection):
                           TIMEOUT = 1  # Error reported by type checker

                   There is no runtime checking of these properties.""")

