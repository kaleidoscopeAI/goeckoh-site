    import colorsys
    from typing import Iterable
    from pip._vendor.rich.color import Color
    from pip._vendor.rich.console import Console, ConsoleOptions
    from pip._vendor.rich.segment import Segment
    from pip._vendor.rich.style import Style

    class ColorBox:
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> Iterable[Segment]:
            height = console.size.height - 3
            for y in range(0, height):
                for x in range(options.max_width):
                    h = x / options.max_width
                    l = y / (height + 1)
                    r1, g1, b1 = colorsys.hls_to_rgb(h, l, 1.0)
                    r2, g2, b2 = colorsys.hls_to_rgb(h, l + (1 / height / 2), 1.0)
                    bgcolor = Color.from_rgb(r1 * 255, g1 * 255, b1 * 255)
                    color = Color.from_rgb(r2 * 255, g2 * 255, b2 * 255)
                    yield Segment("▄", Style(color=color, bgcolor=bgcolor))
                yield Segment.line()

    console = Console()
    console.print(ColorBox())


