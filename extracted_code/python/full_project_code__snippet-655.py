"""
State recorded for an individual pyparsing Element
"""

# Note: this should be a dataclass, but we have to support Python 3.5
def __init__(
    self,
    element: pyparsing.ParserElement,
    converted: EditablePartial,
    parent: EditablePartial,
    number: int,
    name: str = None,
    parent_index: typing.Optional[int] = None,
):
    #: The pyparsing element that this represents
    self.element: pyparsing.ParserElement = element
    #: The name of the element
    self.name: typing.Optional[str] = name
    #: The output Railroad element in an unconverted state
    self.converted: EditablePartial = converted
    #: The parent Railroad element, which we store so that we can extract this if it's duplicated
    self.parent: EditablePartial = parent
    #: The order in which we found this element, used for sorting diagrams if this is extracted into a diagram
    self.number: int = number
    #: The index of this inside its parent
    self.parent_index: typing.Optional[int] = parent_index
    #: If true, we should extract this out into a subdiagram
    self.extract: bool = False
    #: If true, all of this element's children have been filled out
    self.complete: bool = False

def mark_for_extraction(
    self, el_id: int, state: "ConverterState", name: str = None, force: bool = False
):
    """
    Called when this instance has been seen twice, and thus should eventually be extracted into a sub-diagram
    :param el_id: id of the element
    :param state: element/diagram state tracker
    :param name: name to use for this element's text
    :param force: If true, force extraction now, regardless of the state of this. Only useful for extracting the
    root element when we know we're finished
    """
    self.extract = True

    # Set the name
    if not self.name:
        if name:
            # Allow forcing a custom name
            self.name = name
        elif self.element.customName:
            self.name = self.element.customName
        else:
            self.name = ""

    # Just because this is marked for extraction doesn't mean we can do it yet. We may have to wait for children
    # to be added
    # Also, if this is just a string literal etc, don't bother extracting it
    if force or (self.complete and _worth_extracting(self.element)):
        state.extract_into_diagram(el_id)


