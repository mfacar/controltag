# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Control TAC',
    version='0.1.0',
    description='Classification of anxiety level',
    long_description=readme,
    author='Mercy Falconi',
    author_email='mercy.falconi@outlook.com',
    url='https://github.com/mfacar/controltac',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

