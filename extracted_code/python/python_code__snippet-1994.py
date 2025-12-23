"""Command name auto-correct."""
from difflib import get_close_matches

name = name.lower()

close_commands = get_close_matches(name, commands_dict.keys())

if close_commands:
    return close_commands[0]
else:
    return None


