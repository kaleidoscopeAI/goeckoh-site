import sys

from pip._vendor.rich.console import Console

try:
    text = sys.argv[1]
except IndexError:
    text = "Hello, World"
console = Console()
console.print(Rule(title=text))

console = Console()
console.print(Rule("foo"), width=4)


