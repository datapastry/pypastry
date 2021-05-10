import argparse
import sys
from os import path, mkdir, chdir


def run():
    parser = argparse.ArgumentParser(prog='pastry init')
    parser.add_argument('directory', nargs='?', type=str,
                        help='Path to directory to create (or default to current directory)')
    args = parser.parse_args(sys.argv[2:])  # type: argparse.Namespace

    directory = args.directory if args.directory is not None else '.'

    try:
        mkdir(directory)
    except FileExistsError:
        pass

    chdir(directory)
