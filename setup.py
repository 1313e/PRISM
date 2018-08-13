# -*- coding: utf-8 -*-

"""
PRISM Setup
===========
Contains the setup script required for installing the PRISM package.
This can be ran directly by using::

    python setup.py install

or anything equivalent.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, with_statement

# Built-in imports
from codecs import open

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Get the version
exec(open('prism/__version__.py', 'r').read())

# Setup function declaration
setup(name='prism-tool',
      version=prism_version,
      author="Ellert van der Velden",
      author_email="evandervelden@swin.edu.au",
      description=("PRISM: A \"Probabilistic Regression Instrument for "
                   "Simulating Models\""),
      long_description=long_description,
      url="https://github.com/1313e/PRISM",
      license='BSD-3',
      platforms=['Windows', 'Linux', 'Unix'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Utilities',
          ],
      keywords='PRISM prism model analysis emulator regression',
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
      packages=find_packages(),
      package_dir={'prism': "prism"},
      include_package_data=True,
      dependency_links=[
          'git+https://gitlab.mpcdf.mpg.de/ext-c45684b140ce/D2O.git'
          '#egg=d2o-1.1.2'],
      install_requires=requirements,
      zip_safe=False,
      )
