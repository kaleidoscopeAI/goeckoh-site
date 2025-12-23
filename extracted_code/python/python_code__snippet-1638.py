import argparse
import sys

parser = argparse.ArgumentParser(
    description="Render syntax to the console with Rich"
)
parser.add_argument(
    "path",
    metavar="PATH",
    help="path to file, or - for stdin",
)
parser.add_argument(
    "-c",
    "--force-color",
    dest="force_color",
    action="store_true",
    default=None,
    help="force color for non-terminals",
)
parser.add_argument(
    "-i",
    "--indent-guides",
    dest="indent_guides",
    action="store_true",
    default=False,
    help="display indent guides",
)
parser.add_argument(
    "-l",
    "--line-numbers",
    dest="line_numbers",
    action="store_true",
    help="render line numbers",
)
parser.add_argument(
    "-w",
    "--width",
    type=int,
    dest="width",
    default=None,
    help="width of output (default will auto-detect)",
)
parser.add_argument(
    "-r",
    "--wrap",
    dest="word_wrap",
    action="store_true",
    default=False,
    help="word wrap long lines",
)
parser.add_argument(
    "-s",
    "--soft-wrap",
    action="store_true",
    dest="soft_wrap",
    default=False,
    help="enable soft wrapping mode",
)
parser.add_argument(
    "-t", "--theme", dest="theme", default="monokai", help="pygments theme"
)
parser.add_argument(
    "-b",
    "--background-color",
    dest="background_color",
    default=None,
    help="Override background color",
)
parser.add_argument(
    "-x",
    "--lexer",
    default=None,
    dest="lexer_name",
    help="Lexer name",
)
parser.add_argument(
    "-p", "--padding", type=int, default=0, dest="padding", help="Padding"
)
parser.add_argument(
    "--highlight-line",
    type=int,
    default=None,
    dest="highlight_line",
    help="The line number (not index!) to highlight",
)
args = parser.parse_args()

from pip._vendor.rich.console import Console

console = Console(force_terminal=args.force_color, width=args.width)

if args.path == "-":
    code = sys.stdin.read()
    syntax = Syntax(
        code=code,
        lexer=args.lexer_name,
        line_numbers=args.line_numbers,
        word_wrap=args.word_wrap,
        theme=args.theme,
        background_color=args.background_color,
        indent_guides=args.indent_guides,
        padding=args.padding,
        highlight_lines={args.highlight_line},
    )
else:
    syntax = Syntax.from_path(
        args.path,
        lexer=args.lexer_name,
        line_numbers=args.line_numbers,
        word_wrap=args.word_wrap,
        theme=args.theme,
        background_color=args.background_color,
        indent_guides=args.indent_guides,
        padding=args.padding,
        highlight_lines={args.highlight_line},
    )
console.print(syntax, soft_wrap=args.soft_wrap)


