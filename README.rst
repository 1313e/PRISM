NOTE
====
This is the v1.0.x branch of *PRISM*, which still supports Python 2.7.
Starting with v1.1.0, *PRISM* no longer supports Python 2.7, and this branch will not be maintained.

----

|PyPI| |Python| |License| |Travis| |AppVeyor| |ReadTheDocs| |Coverage|

Model dispersion with *PRISM*; an alternative to MCMC for rapid analysis of models
==================================================================================
*PRISM* is a pure Python 2/3 package that provides an alternative method to MCMC for analyzing scientific models.

Introduction
============
Rapid technological advancements allow for both computational resources and observational/experimental instruments to become better, faster and more precise with every passing year.
This leads to an ever-increasing amount of scientific data being available and more research questions being raised.
As a result, scientific models that attempt to address these questions are becoming more abundant, and are pushing the available resources to the limit as these models incorporate more complex science and more closely resemble reality.

However, as the number of available models increases, they also tend to become more distinct, making it difficult to keep track of their individual qualities.
A full analysis of every model would be required in order to recognize these qualities.
It is common to employ Markov chain Monte Carlo (MCMC) methods and Bayesian statistics for performing this task.
However, as these methods are meant to be used for making approximations of the posterior probability distribution function, there must be a more efficient way of analyzing them.

*PRISM* tries to tackle this problem by using the Bayes linear approach, the emulation technique and history matching to construct an approximation ('emulator') of any given model.
The use of these techniques can be seen as special cases of Bayesian statistics, where limited model evaluations are combined with advanced regression techniques, covariances and probability calculations.
*PRISM* is designed to easily facilitate and enhance existing MCMC methods by restricting plausible regions and exploring parameter space efficiently.
However, *PRISM* can additionally be used as a standalone alternative to MCMC for model analysis, providing insight into the bahvior of complex scientific models.
With *PRISM*, the time spent on evaluating a model is minimized, providing developers with an advanced model analysis for a fraction of the time required by more traditional methods.

Why use *PRISM*?
----------------
- Written in pure Python 2/3, for versatility;
- Stores results in `HDF5-files`_, allowing for easy user-access;
- Can be executed in serial or MPI, on any number of processes;
- Compatible with Windows, Mac OS and Unix-based machines;
- Accepts any type of model and comparison data;
- Built as a plug-and-play tool: all main classes can also be used as base classes;
- Easily linked to any model by writing a single custom ModelLink subclass;
- Capable of reducing relevant parameter space by factors over 100,000 using only a few thousand model evaluations;
- Can be used alone for analyzing models, or combined with MCMC for efficient model parameter estimations.

.. _HDF5-files: https://portal.hdfgroup.org/display/HDF5/HDF5


Getting started
===============
Installation
------------
*PRISM* can be easily installed by either cloning the `repository`_ and installing it manually::

    $ git clone https://github.com/1313e/PRISM?branch=v1.0.x
    $ cd PRISM
    $ pip install .

or by installing it directly from `PyPI`_ with::

    $ pip install prism==1.0

*PRISM* can now be imported as a package with ``import prism``.
For using *PRISM* in MPI, ``mpi4py >= 3.0.0`` is required (not installed automatically).

The *PRISM* package comes with two ModelLink subclasses.
These ModelLink subclasses can be used to experiment with *PRISM* to see how it works.
The `online docs`_ have several examples explaining the different functionalities of the package.

.. _repository: https://github.com/1313e/PRISM?branch=v1.0.x
.. _PyPI: https://pypi.org/project/prism/1.0.0
.. _online docs: https://prism-tool.readthedocs.io/en/v1.0.x


Example usage
-------------
See `online docs`_ for a documented explanation on this example.

.. code:: python

    # Imports
    from prism import Pipeline
    from prism.modellink import GaussianLink

    # Define model data and create ModelLink object
    model_data = {3: [3.0, 0.1], 5: [5.0, 0.1], 7: [3.0, 0.1]}
    modellink_obj = GaussianLink(model_data=model_data)

    # Create Pipeline object
    pipe = Pipeline(modellink_obj)

    # Construct first iteration of the emulator
    pipe.construct()

    # Create projections
    pipe.project()


.. |PyPI| image:: https://img.shields.io/badge/PyPI-v1.0.0-blue.svg
    :target: https://pypi.python.org/pypi/prism/v1.0.0
    :alt: PyPI - Latest v1.0.x Release
.. |Python| image:: https://img.shields.io/badge/Python-2.7%20%7C%203.5%20%7C%203.6%20%7C%203.7-blue.svg?logo=python&logoColor=white
    :target: https://pypi.python.org/pypi/prism/v1.0.0
    :alt: PyPI - Python Versions
.. |License| image:: https://img.shields.io/pypi/l/prism.svg?colorB=blue&label=License
    :target: https://github.com/1313e/PRISM/raw/v1.0.x/LICENSE
    :alt: PyPI - License
.. |Travis| image:: https://img.shields.io/travis/com/1313e/PRISM/v1.0.x.svg?logo=travis&label=Travis%20CI
    :target: https://travis-ci.com/1313e/PRISM?branch=v1.0.x
    :alt: Travis CI - Build Status
.. |AppVeyor| image:: https://img.shields.io/appveyor/ci/1313e/PRISM/v1.0.x.svg?logo=appveyor&label=AppVeyor
    :target: https://ci.appveyor.com/project/1313e/PRISM?branch=v1.0.x
    :alt: AppVeyor - Build Status
.. |ReadTheDocs| image:: https://img.shields.io/readthedocs/prism-tool/v1.0.x.svg?logo=read%20the%20docs&logoColor=white&label=Docs
    :target: https://prism-tool.readthedocs.io/en/v1.0.x
    :alt: ReadTheDocs - Build Status
.. |Coverage| image:: https://img.shields.io/codecov/c/github/1313e/PRISM/v1.0.x.svg?logo=codecov&logoColor=white&label=Coverage
    :target: https://codecov.io/gh/1313e/PRISM?branch=v1.0.x
    :alt: CodeCov - Coverage Status
