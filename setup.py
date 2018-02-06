# -*- coding: utf-8 -*-

"""
Setup file for the PRISM package.

"""

from setuptools import find_packages, setup
from codecs import open
from prism.version import version as __version__

# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

setup(name="prism_tool",
      version=__version__,
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("Probabilistic Regression Instrument for Simulating "
                   "Models"),
      long_description=long_description,
      url='https://www.github.com/1313e/PRISM',
      license='BSD-3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          ],
      keywords='PRISM prism model analysis emulator regression',
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
      packages=find_packages(),
      package_dir={'prism': "prism"},
      include_package_data=True,
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
