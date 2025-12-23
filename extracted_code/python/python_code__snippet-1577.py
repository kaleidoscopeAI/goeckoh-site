import os

console = Console()

files = [f"{i} {s}" for i, s in enumerate(sorted(os.listdir()))]
columns = Columns(files, padding=(0, 1), expand=False, equal=False)
console.print(columns)
console.rule()
columns.column_first = True
console.print(columns)
columns.right_to_left = True
console.rule()
console.print(columns)


