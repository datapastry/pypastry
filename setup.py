import sys

from setuptools import setup

if sys.version_info < (3, 5):
    sys.exit('Sorry, PyPastry requires Python version 3.5 or greater')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pypastry',
    version='0.0.1',
    description='PyPastry machine learning experimentation framework',
    author='Daoud Clarke',
    scripts=['pastry'],
    install_requires=['tomlkit', 'pandas', 'scikit-learn', 'pyarrow', 'gitpython'],
    package_dir ={'': 'pypastry'},
    long_description = long_description,
    long_description_content_type="text/markdown",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
