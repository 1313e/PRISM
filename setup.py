# -*- coding: utf-8 -*-

"""
PRISM Setup
===========
Contains the setup script required for installing the *PRISM* package.
This can be ran directly by using::

    pip install .

or anything equivalent.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, with_statement

# Built-in imports
from codecs import open
import re

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Read the __version__.py file
with open('prism/__version__.py', 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

# Setup function declaration
setup(name='prism',
      version=version,
      author="Ellert van der Velden",
      author_email="evandervelden@swin.edu.au",
      maintainer="1313e",
      description=("PRISM: An alternative to MCMC for rapid analysis of "
                   "models"),
      long_description=long_description,
      download_url=("https://github.com/1313e/PRISM/archive/v%s.zip"
                    % (version)),
      url="https://prism-tool.readthedocs.io/en/v1.0.x",
      project_urls={
          'Documentation': "https://prism-tool.readthedocs.io/en/v1.0.x",
          'Source Code': "https://github.com/1313e/PRISM?branch=v1.0.x"
          },
      license='BSD-3',
      platforms=['Windows', 'Mac OS-X', 'Linux', 'Unix'],
      classifiers=[
          'Development Status :: 7 - Inactive',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
          ],
      keywords=('PRISM prism model analysis emulator regression MCMC '
                'optimization'),
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
      packages=find_packages(),
      package_dir={'prism': "prism"},
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      )
