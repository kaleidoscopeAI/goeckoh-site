"""
Simple subclass of Group that creates an annotation label
"""

def __init__(self, label: str, item):
    super().__init__(item=item, label="[{}]".format(label) if label else label)


