# -*- coding: utf-8 -*-

"""
PRISM Setup
===========
Contains the setup script required for installing the PRISM package.
This can be ran directly by using::

    python setup.py install

"""


# %% IMPORTS
# Built-in imports
from codecs import open

# Package imports
from setuptools import find_packages, setup

# PRISM imports
from prism.version import version as __version__


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Setup function declaration
setup(name="prism_tool",
      version=__version__,
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("A \"Probabilistic Regression Instrument for Simulating "
                   "Models\""),
      long_description=long_description,
      url='https://www.github.com/1313e/PRISM',
      license='BSD-3',
      platforms=["Windows", "Linux", "Unix"],
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
#      test_suite="",
      install_requires=['e13tools>=0.4.2a0',
                        'numpy>=1.8',
                        'matplotlib>=1.4.3',
                        'astropy>=1.3',
                        'scipy>=1.0.0',
                        'h5py>=2.7.1',
                        'mlxtend>=0.9.0',
                        'scikit-learn>=0.19.1'],
      zip_safe=False,
      )
