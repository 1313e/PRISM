|PyPI| |Python| |License| |Travis| |AppVeyor| |Coverage|

Model dispersion with PRISM, a "Probabilistic Regression Instrument for Simulating Models"
==========================================================================================
PRISM is a pure Python 2/3 package that provides an alternative method to MCMC for analyzing scientific models.

Introduction
============
Typically we probe the universe by making models that try to reconstruct reality based on our scientific knowledge.
Since our knowledge is limited, models tend to only tell part of the story.
Commonly we utilize MCMC methods in order to check how closely this resembles reality.
Although MCMC can precisely return the model realization that does this, it has a few drawbacks: It is slow, requires much additional knowledge about the model for a full Bayesian analysis, is vulnerable to irregularities and its convergence probability vs. speed depends on the initial conditions.
This makes MCMC hard to use for complex models, eliminating the possibility for developers to discover additional details about their model, be it new physics, interesting effects or errors.

*PRISM* tries to tackle this problem by providing a different way for analyzing models.
Instead of evaluating a model millions of times, often in regions of parameter space that do not contain interesting model realizations, *PRISM* constructs an approximate version of the model with polynomial functions based on a few thousand model evaluations.
By utilizing this system, *PRISM* is capable of identifying large parts of parameter space as 'implausible' with only limited model knowledge.
Additionally, *PRISM* will map out the behavior of a model, allowing developers to study its properties.
This makes *PRISM* an excellent alternative to ordinary MCMC methods for developers that seek to analyze and optimize their models.

What can *PRISM* do for your model?
-----------------------------------
- Rapid analysis of your model, being several magnitudes faster than ordinary MCMC methods;
- Provide an approximated version of the model;
- Analyze and map out its physical behavior;
- Detect inaccuracies and flaws;
- Advise on important missing constraining data;
- Massively reduce relevant parameter space by factors up to 100,000, allowing existing MCMC methods to explore and obtain the optimal model realizations much faster.

What characterizes *PRISM*?
---------------------------
- Written in pure Python 2/3, for versatility;
- Stores results in `HDF5-files`_, allowing for easy user-access;
- Can be executed in serial or MPI, on any number of processes;
- Compatible with Windows, MacOS and Unix-based machines;
- Accepts any type of model and comparison data;
- Built as a plug-and-play tool: all main classes can also be used as base classes;
- Easily linked to any model by writing a single custom ModelLink class;
- Extensively documented;
- Suited for both simple and advanced projects.

.. _HDF5-files: https://portal.hdfgroup.org/display/HDF5/HDF5

Getting started
===============
Installation
------------
*PRISM* can be easily installed by either cloning the `repository`_ and executing the following::

	$ git clone https://github.com/1313e/PRISM
	$ cd PRISM
	$ pip install .

or by installing it directly from `PyPI`_ with::

	$ pip install prism-tool

*PRISM* can now be imported as a package with ``import prism``.

.. _repository: https://github.com/1313e/PRISM
.. _PyPI: https://pypi.org/project/prism-tool

The *PRISM* package comes with several test scripts, data files and ModelLink subclasses.
These test scripts work out-of-the-box and can be used to see how *PRISM* works, and what the typical lay-outs of the required files are.

Dependencies
++++++++++++
*PRISM* requires ``python == 2.7`` or ``python >= 3.5`` and the following non-standard dependencies (installed automatically):

- ``e13tools >= 0.4.7a1``;
- ``mlxtend >= 0.9.1``;
- ``scikit-learn >= 0.19.1``;
- ``sortedcontainers >= 1.5.9``.

For running *PRISM* in MPI, the following packages are also required (not installed automatically):

- ``mpi4py >= 3.0.0``.


.. |PyPI| image:: https://img.shields.io/pypi/v/prism_tool.svg?label=PyPI
   :target: https://pypi.python.org/pypi/prism_tool
   :alt: PyPI - Release
.. |Python| image:: https://img.shields.io/pypi/pyversions/prism_tool.svg?label=Python
   :target: https://pypi.python.org/pypi/prism_tool
   :alt: PyPi - Python Versions
.. |License| image:: https://img.shields.io/pypi/l/prism_tool.svg?colorB=blue&label=License
   :target: https://github.com/1313e/PRISM/raw/master/LICENSE
   :alt: PyPI - License
.. |Travis| image:: https://img.shields.io/travis/com/1313e/PRISM/master.svg?logo=travis&label=Travis%20CI
   :target: https://travis-ci.com/1313e/PRISM
   :alt: Travis CI - Build Status
.. |AppVeyor| image:: https://img.shields.io/appveyor/ci/1313e/PRISM/master.svg?logo=appveyor&label=AppVeyor
   :target: https://ci.appveyor.com/project/1313e/PRISM
   :alt: AppVeyor - Build Status
.. |Coverage| image:: https://img.shields.io/coveralls/github/1313e/PRISM/master.svg?label=Coverage
   :target: https://coveralls.io/github/1313e/PRISM?branch=master
   :alt: Coveralls - Coverage Status
