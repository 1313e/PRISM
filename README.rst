.. image:: https://github.com/1313e/PRISM/raw/master/logo/png/PRISM_transparent_Logo1_crop.png
    :width: 400 px
    :align: center
    :target: https://prism-tool.readthedocs.io/en/latest
    :alt: PRISM Logo

|PyPI| |Python| |Travis| |AppVeyor| |Azure| |Coverage| |Gitter|

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
- Easily linked to any model by writing a single custom `ModelLink subclass`_;
- Capable of reducing relevant parameter space by factors over 100,000 using only a few thousand model evaluations;
- Can be used alone for analyzing models, or combined with MCMC for efficient model parameter estimations.

When (not) to use *PRISM*?
--------------------------
It may look very tempting to use *PRISM* for basically everything, but keep in mind that emulation has its limits.
Below is a general (but non-exhaustive) list of scenarios where *PRISM* can become really valuable:

- In almost any situation where one wishes to perform a parameter estimation using an MCMC Bayesian analysis (by using `hybrid sampling`_).
  This is especially true for poorly constrained models (low number of available observational constraints);
- Whenever one wishes to visualize the correlation behavior between different model parameters;
- For quickly exploring the parameter space of a model without performing a full parameter estimation.
  This can be very useful when trying out different sets of observational data to study their constraining power;
- For obtaining a reasonably accurate approximation of a model in very close proximity to the most optimal parameter set.

There are however also situations where one is better off using a different technique, with a general non-exhaustive list below:

- For obtaining a reasonably accurate approximation of a model in all of parameter space.
  Due to the way an emulator is constructed, this could easily require millions of model evaluations and a lot of time and memory;
- When dealing with a model that has a large number of parameters/degrees-of-freedom (>50).
  This however still heavily depends on the type of model that is used;
- Whenever a very large number of observational constraints are available and one wishes to use all of them (unless one also has access to a large supercomputer).
  In this case, it is a better idea to use full Bayesian instead;
- One wishes to obtain the posterior probability distribution function (PDF) of a model.

A very general and easy way to check if one should use *PRISM*, is to ask oneself the question: *"Would I use a full Bayesian analysis for this problem, given the required time and resources?"*.
If the answer is 'yes', then *PRISM* is probably a good choice, especially as it requires near-similar resources as a Bayesian analysis does (definition of parameter space; provided comparison data; and a way to evaluate the model).

.. _HDF5-files: https://portal.hdfgroup.org/display/HDF5/HDF5
.. _ModelLink subclass: https://prism-tool.readthedocs.io/en/latest/user/modellink_crash_course.html
.. _hybrid sampling: https://prism-tool.readthedocs.io/en/latest/user/using_prism.html#hybrid-sampling


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
The `online docs`_ and `the tutorials`_ have several examples explaining the different functionalities of the package.

.. _repository: https://github.com/1313e/PRISM
.. _PyPI: https://pypi.org/project/prism
.. _online docs: https://prism-tool.readthedocs.io
.. _the tutorials: https://github.com/1313e/PRISM/tree/master/tutorials


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
See `online docs`_ or `the tutorials`_ for a documented explanation on this example.

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
As with contributions, if you find a problem or issue with *PRISM*, please do not hesitate to open a `GitHub issue`_ about it or post it on `Gitter`_.

And, finally, if you use *PRISM* as part of your workflow in a scientific publication, please consider including an acknowledgement like *"Parts of the results in this work were derived using the PRISM Python package."* and citing the *PRISM* pipeline paper:

::

    @ARTICLE{2019ApJS..242...22V,
        author = {{van der Velden}, E. and {Duffy}, A.~R. and {Croton}, D. and
            {Mutch}, S.~J. and {Sinha}, M.},
        title = "{Model dispersion with PRISM; an alternative to MCMC for rapid analysis of models}",
        journal = {\apjs},
        keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Computational Physics},
        year = "2019",
        month = "Jun",
        volume = {242},
        number = {2},
        eid = {22},
        pages = {22},
        doi = {10.3847/1538-4365/ab1f7d},
        archivePrefix = {arXiv},
        eprint = {1901.08725},
        primaryClass = {astro-ph.IM},
        adsurl = {http://adsabs.harvard.edu/abs/2019ApJS..242...22V},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

.. _BSD-3 license: https://github.com/1313e/PRISM/raw/master/LICENSE
.. _GitHub issue: https://github.com/1313e/PRISM/issues
.. _Gitter: https://gitter.im/1313e/PRISM

Acknowledgements
================
Special thanks to Alan Duffy, Darren Croton, Simon Mutch and Manodeep Sinha for providing many valuable suggestions and constructive feedback points.
Huge thanks to James Josephides for making the *PRISM* logo.

.. |PyPI| image:: https://img.shields.io/pypi/v/prism.svg?logo=pypi&logoColor=white&label=PyPI
    :target: https://pypi.python.org/pypi/prism
    :alt: PyPI - Latest Release
.. |Python| image:: https://img.shields.io/pypi/pyversions/prism.svg?logo=python&logoColor=white&label=Python
    :target: https://pypi.python.org/pypi/prism
    :alt: PyPI - Python Versions
.. |Travis| image:: https://img.shields.io/travis/com/1313e/PRISM/master.svg?logo=travis%20ci&logoColor=white&label=Travis%20CI
    :target: https://travis-ci.com/1313e/PRISM
    :alt: Travis CI - Build Status
.. |AppVeyor| image:: https://img.shields.io/appveyor/ci/1313e/PRISM/master.svg?logo=appveyor&logoColor=white&label=AppVeyor
    :target: https://ci.appveyor.com/project/1313e/PRISM/branch/master
    :alt: AppVeyor - Build Status
.. |Azure| image:: https://img.shields.io/azure-devops/build/1313e/2f7c67c7-61eb-4e70-9ff3-7f54f8e39987/1?logo=azure-pipelines&logoColor=white&label=Azure
    :target: https://dev.azure.com/1313e/PRISM/_build/latest?definitionId=1
    :alt: Azure Pipelines - Build Status
.. |Coverage| image:: https://img.shields.io/codecov/c/github/1313e/PRISM/master.svg?logo=codecov&logoColor=white&label=Coverage
    :target: https://codecov.io/gh/1313e/PRISM/branches/master
    :alt: CodeCov - Coverage Status
.. |Gitter| image:: https://img.shields.io/gitter/room/1313e/PRISM.svg?logo=gitter&logoColor=white&label=Chat
    :target: https://gitter.im/1313e/PRISM
    :alt: Gitter - Chat Room
