from .console import Console

console = Console(width=10)
console.print("12345 abcdefghijklmnopqrstuvwyxzABCDEFGHIJKLMNOPQRSTUVWXYZ 12345")
print(chop_cells("abcdefghijklmnopqrstuvwxyz", 10, position=2))


