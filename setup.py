import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    sys.exit('Sorry, PyPastry requires Python version 3.5 or greater')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pypastry',
    version='0.0.1',
    description='PyPastry machine learning experimentation framework',
    author='Daoud Clarke',
    url='https://github.com/datapastry/pypastry',
    scripts=['pastry'],
    install_requires=['tomlkit', 'pandas', 'scikit-learn', 'pyarrow', 'gitpython'],
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    python_requires='>=3.5',
    )
