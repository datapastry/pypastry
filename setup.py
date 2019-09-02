import sys

from setuptools import setup

if sys.version_info < (3, 5):
    sys.exit('Sorry, PyPastry requires Python version 3.5 or greater')

setup(
    name='pypastry',
    version='0.0.1',
    description='PyPastry machine learning experimentation framework',
    author='Daoud Clarke',
    scripts=['pastry'],
    install_requires=['tomlkit', 'pandas', 'scikit-learn', 'pyarrow', 'gitpython'],
    packages=['pypastry'],
)
