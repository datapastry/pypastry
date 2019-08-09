DISPLAY_PATH = '.pypastry/display.txt'


def print_display():
    try:
        _print_cache_file()
    except FileNotFoundError:
        from pypastry.display import cache_display
        cache_display()
        _print_cache_file()


def _print_cache_file():
    with open(DISPLAY_PATH) as display_file:
        print(display_file.read())
