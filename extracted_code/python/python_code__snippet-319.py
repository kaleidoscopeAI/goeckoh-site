    from pip._vendor.rich.console import Console

    console = Console()

    term = LegacyWindowsTerm(sys.stdout)
    term.set_title("Win32 Console Examples")

    style = Style(color="black", bgcolor="red")

    heading = Style.parse("black on green")

    # Check colour output
    console.rule("Checking colour output")
    console.print("[on red]on red!")
    console.print("[blue]blue!")
    console.print("[yellow]yellow!")
    console.print("[bold yellow]bold yellow!")
    console.print("[bright_yellow]bright_yellow!")
    console.print("[dim bright_yellow]dim bright_yellow!")
    console.print("[italic cyan]italic cyan!")
    console.print("[bold white on blue]bold white on blue!")
    console.print("[reverse bold white on blue]reverse bold white on blue!")
    console.print("[bold black on cyan]bold black on cyan!")
    console.print("[black on green]black on green!")
    console.print("[blue on green]blue on green!")
    console.print("[white on black]white on black!")
    console.print("[black on white]black on white!")
    console.print("[#1BB152 on #DA812D]#1BB152 on #DA812D!")

    # Check cursor movement
    console.rule("Checking cursor movement")
    console.print()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text("went back and wrapped to prev line")
    time.sleep(1)
    term.move_cursor_up()
    term.write_text("we go up")
    time.sleep(1)
    term.move_cursor_down()
    term.write_text("and down")
    time.sleep(1)
    term.move_cursor_up()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text("we went up and back 2")
    time.sleep(1)
    term.move_cursor_down()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text("we went down and back 2")
    time.sleep(1)

    # Check erasing of lines
    term.hide_cursor()
    console.print()
    console.rule("Checking line erasing")
    console.print("\n...Deleting to the start of the line...")
    term.write_text("The red arrow shows the cursor location, and direction of erase")
    time.sleep(1)
    term.move_cursor_to_column(16)
    term.write_styled("<", Style.parse("black on red"))
    term.move_cursor_backward()
    time.sleep(1)
    term.erase_start_of_line()
    time.sleep(1)

    console.print("\n\n...And to the end of the line...")
    term.write_text("The red arrow shows the cursor location, and direction of erase")
    time.sleep(1)

    term.move_cursor_to_column(16)
    term.write_styled(">", Style.parse("black on red"))
    time.sleep(1)
    term.erase_end_of_line()
    time.sleep(1)

    console.print("\n\n...Now the whole line will be erased...")
    term.write_styled("I'm going to disappear!", style=Style.parse("black on cyan"))
    time.sleep(1)
    term.erase_line()

    term.show_cursor()
    print("\n")


