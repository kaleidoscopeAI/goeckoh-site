"""
A range to highlight in a Syntax object.
`start` and `end` are 2-integers tuples, where the first integer is the line number
(starting from 1) and the second integer is the column index (starting from 0).
"""

style: StyleType
start: SyntaxPosition
end: SyntaxPosition


