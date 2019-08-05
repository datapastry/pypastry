from setuptools import setup

setup(
    name='pypastry',
    version='0.0.1',
    description='PyPastry machine learning experimentation framework',
    author='Daoud Clarke',
    scripts=['pastry'],
    install_requires=['tomlkit', 'pandas', 'scikit-learn'],
    packages=['pypastry'],
)
