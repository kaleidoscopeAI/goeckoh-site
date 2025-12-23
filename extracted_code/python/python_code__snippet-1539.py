"""Get the first paragraph from a docstring."""
paragraph, _, _ = doc.partition("\n\n")
return paragraph


