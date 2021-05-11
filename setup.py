import sys

from setuptools import setup, find_packages

#check to make sure the python version is compatible 
if sys.version_info < (3, 6):
    sys.exit('Sorry, PyPastry requires Python version 3.6 or greater')

# Reading in the ReadMe file as the doc file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pypastry',
    version='0.2.0',
    description='PyPastry machine learning experimentation framework',
    author='Daoud Clarke',
    url='https://github.com/datapastry/pypastry',
    scripts=['pastry'],
    install_requires=['tomlkit', 'pandas', 'scikit-learn', 'pyarrow', 'gitpython'],
    #To find the packages 
    packages=find_packages(),
    #To read in data file modules 
    py_modules=['data/pie'],
    # commands that can be run in a console in the commands folder
    entry_points={
        'console_scripts': [
            'init = pypastry.commands.init:run',
            'print = pypastry.commands.print_:run',
            'run = pypastry.commands.run:run'
            ]},
    package_data={

         '' : ['data/*.gitignore'],
        # And include any *.gitignore files found in the 'data' package, too:
        'data': ['*.gitignore'],

    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    # Make the setup file aware of the Manifest file
    include_package_data=True,
    #Minimum requirement of python, licesnse, and operating system. 
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    python_requires='>=3.5',
    )
