"""Exception to indicate a response was invalid. Raise this within process_response() to indicate an error
and provide an error message.

Args:
    message (Union[str, Text]): Error message.
"""

def __init__(self, message: TextType) -> None:
    self.message = message

def __rich__(self) -> TextType:
    return self.message


