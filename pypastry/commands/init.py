import argparse
import sys
from os import path, mkdir, chdir
from shutil import copyfile

from git import Repo


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

    repo = Repo.init('.')  # type:  git.repo.base.Repo
    for filename in ['pie.py', '.gitignore']:
        source_file_path = path.join(path.dirname(__file__), '..', '..', 'data', filename)
        copyfile(source_file_path, filename)
        repo.index.add([filename])

