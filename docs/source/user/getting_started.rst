.. _getting_started:

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

*PRISM* can now be imported as a package with :pycode:`import prism`.
For using *PRISM* in MPI, ``mpi4py >= 3.0.0`` is required (not installed automatically).

The *PRISM* package comes with two ModelLink subclasses.
These ModelLink subclasses can be used to experiment with *PRISM* to see how it works.
:ref:`using_prism` has several examples explaining the different functionalities of the package.

.. _repository: https://github.com/1313e/PRISM
.. _PyPI: https://pypi.org/project/prism


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
