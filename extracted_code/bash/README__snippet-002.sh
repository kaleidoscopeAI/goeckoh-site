# Read all .txt/.md/.pdf files from a folder recursively
python -m cli read-docs --path ./documents --recursive

# Non-recursive (top-level only)
python -m cli read-docs --path ./documents --no-recursive
