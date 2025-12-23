# Install fdupes (if not already installed)
sudo apt install fdupes -y

# Find duplicates in the project directory
fdupes -r /path/to/project > duplicates.txt

# Remove duplicates interactively (be careful to avoid accidental deletions)
fdupes -r -d /path/to/project

