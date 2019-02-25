|PyPI| |Python| |License| |Travis| |AppVeyor| |ReadTheDocs| |Coverage|

Model dispersion with *PRISM*; an alternative to MCMC for rapid analysis of models
==================================================================================
*PRISM* is a pure Python 3 package that provides an alternative method to MCMC for analyzing scientific models.

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
However, *PRISM* can additionally be used as a standalone alternative to MCMC for model analysis, providing insight into the behavior of complex scientific models.
With *PRISM*, the time spent on evaluating a model is minimized, providing developers with an advanced model analysis for a fraction of the time required by more traditional methods.

Why use *PRISM*?
----------------
- Written in pure Python 3, for versatility;
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

    $ git clone https://github.com/1313e/PRISM
    $ cd PRISM
    $ pip install .

or by installing it directly from `PyPI`_ with::

    $ pip install prism

*PRISM* can now be imported as a package with ``import prism``.
For using *PRISM* in MPI, ``mpi4py >= 3.0.0`` is required (not installed automatically).

The *PRISM* package comes with two ModelLink subclasses.
These ModelLink subclasses can be used to experiment with *PRISM* to see how it works.
The `online docs`_ have several examples explaining the different functionalities of the package.

.. _repository: https://github.com/1313e/PRISM
.. _PyPI: https://pypi.org/project/prism
.. _online docs: https://prism-tool.readthedocs.io


Running tests
-------------
If one wants to run pytests on *PRISM*, all `requirements_dev`_ are required.
The easiest way to run the tests is by cloning the `repository`_, installing all requirements and then running ``pytest`` on it::

    $ git clone https://github.com/1313e/PRISM
    $ cd PRISM
    $ pip install -r requirements_dev.txt
    $ pytest

If *PRISM* and all `requirements_dev`_ are already installed, one can run the tests by running ``pytest`` in the installation directory::

    $ cd <path_to_installation_directory>/prism
    $ pytest

When using Anaconda, the installation directory path is probably of the form ``<HOME>/anaconda3/envs/<environment_name>/lib/pythonX.X/site-packages``.

.. _requirements_dev: https://github.com/1313e/PRISM/raw/master/requirements_dev.txt


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


Community guidelines
====================
*PRISM* is an open-source and free-to-use software package (and it always will be), provided under the `BSD-3 license`_.

Users are highly encouraged to make contributions to the package or request new features by opening a `GitHub issue`_.
If you would like to contribute to the package, but do not know what, then there are quite a few ToDos in the code that may give you some inspiration.
As with contributions, if you find a problem or issue with *PRISM*, please do not hesitate to open a `GitHub issue`_ about it.

And, finally, if you use *PRISM* as part of your workflow in a scientific publication, please consider including an acknowledgement like *"Parts of the results in this work were derived using the PRISM Python package."* and citing the *PRISM* pipeline paper:

::

    @ARTICLE{2019arXiv190108725V,
       author = {{van der Velden}, E. and {Duffy}, A.~R. and {Croton}, D. and 
    	{Mutch}, S.~J. and {Sinha}, M.},
        title = "{Model dispersion with PRISM; an alternative to MCMC for rapid analysis of models}",
      journal = {arXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1901.08725},
     primaryClass = "astro-ph.IM",
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Computational Physics},
         year = 2019,
        month = jan,
       adsurl = {http://adsabs.harvard.edu/abs/2019arXiv190108725V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


.. _BSD-3 license: https://github.com/1313e/PRISM/raw/master/LICENSE
.. _GitHub issue: https://github.com/1313e/PRISM/issues


.. |PyPI| image:: https://img.shields.io/pypi/v/prism.svg?label=PyPI
    :target: https://pypi.python.org/pypi/prism
    :alt: PyPI - Latest Release
.. |Python| image:: https://img.shields.io/pypi/pyversions/prism.svg?logo=python&logoColor=white&label=Python
    :target: https://pypi.python.org/pypi/prism
    :alt: PyPI - Python Versions
.. |License| image:: https://img.shields.io/pypi/l/prism.svg?colorB=blue&label=License
    :target: https://github.com/1313e/PRISM/raw/master/LICENSE
    :alt: PyPI - License
.. |Travis| image:: https://img.shields.io/travis/com/1313e/PRISM/master.svg?logo=travis&logoColor=white&label=Travis%20CI
    :target: https://travis-ci.com/1313e/PRISM
    :alt: Travis CI - Build Status
.. |AppVeyor| image:: https://img.shields.io/appveyor/ci/1313e/PRISM/master.svg?logo=appveyor&logoColor=white&label=AppVeyor
    :target: https://ci.appveyor.com/project/1313e/PRISM/branch/master
    :alt: AppVeyor - Build Status
.. |ReadTheDocs| image:: https://img.shields.io/readthedocs/prism-tool/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs
    :target: https://prism-tool.readthedocs.io/en/latest
    :alt: ReadTheDocs - Build Status
.. |Coverage| image:: https://img.shields.io/codecov/c/github/1313e/PRISM/master.svg?logo=codecov&logoColor=white&label=Coverage
    :target: https://codecov.io/gh/1313e/PRISM/branches/master
    :alt: CodeCov - Coverage Status
