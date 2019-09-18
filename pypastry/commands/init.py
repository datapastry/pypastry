import argparse
import sys
from os import path, mkdir, chdir
from shutil import copyfile

from git import Repo


def run():
    parser = argparse.ArgumentParser(prog='pastry init')
    parser.add_argument('directory', type=str, help='Path to directory to create (or default to current directory)')
    args = parser.parse_args(sys.argv[2:])  # type: argparse.Namespace

    if args.directory is not None:
        mkdir(args.directory)
        chdir(args.directory)

    repo = Repo.init('.')  #type:  git.repo.base.Repo
    for filename in ['pie.py', '.gitignore']:
        source_file_path = path.join(path.dirname(__file__), '..', '..', 'data', filename)
        copyfile(source_file_path, filename)
        repo.index.add([filename])

