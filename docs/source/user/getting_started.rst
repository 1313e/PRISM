.. _getting_started:

Getting started
===============
Installation
------------
*PRISM* can be easily installed by either cloning the `repository`_ and installing it manually::

    $ git clone https://github.com/1313e/PRISM --branch=v1.0.x
    $ cd PRISM
    $ pip install .

or by installing it directly from `PyPI`_ with::

    $ pip install prism==1.0

*PRISM* can now be imported as a package with :pycode:`import prism`.
For using *PRISM* in MPI, ``mpi4py >= 3.0.0`` is required (not installed automatically).

The *PRISM* package comes with two ModelLink subclasses.
These ModelLink subclasses can be used to experiment with *PRISM* to see how it works.
:ref:`using_prism` has several examples explaining the different functionalities of the package.

.. _repository: https://github.com/1313e/PRISM/tree/v1.0.x
.. _PyPI: https://pypi.org/project/prism/1.0.0


Running tests
-------------
If one wants to run pytests on *PRISM*, all `requirements_dev`_ are required.
The easiest way to run the tests is by cloning the `repository`_, installing all requirements and then running ``pytest`` on it::

    $ git clone https://github.com/1313e/PRISM --branch=v1.0.x
    $ cd PRISM
    $ pip install -r requirements_dev.txt
    $ pytest

If *PRISM* and all `requirements_dev`_ are already installed, one can run the tests by running ``pytest`` in the installation directory::

    $ cd <path_to_installation_directory>/prism
    $ pytest

When using Anaconda, the installation directory path is probably of the form ``<HOME>/anaconda3/envs/<environment_name>/lib/pythonX.X/site-packages``.

.. _requirements_dev: https://github.com/1313e/PRISM/raw/v1.0.x/requirements_dev.txt

Example usage
-------------
See :ref:`minimal_example` for a documented explanation on this example.

::

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
