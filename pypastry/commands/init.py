from os import path
from shutil import copyfile

from git import Repo


def run():
    repo = Repo.init('.')
    for filename in ['pie.py', '.gitignore']:
        source_file_path = path.join(path.dirname(__file__), '..', '..', 'data', filename)
        copyfile(source_file_path, filename)
        repo.index.add([filename])
