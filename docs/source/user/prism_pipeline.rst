.. _prism_pipeline:

The PRISM pipeline
==================
.. figure:: images/PRISM.png
    :name: PRISM
    :alt: The *PRISM* pipeline
    :width: 100%
    :align: center

    The structure of the *PRISM* pipeline.

The overall structure of *PRISM* can be seen in :numref:`PRISM` and will be discussed below.
The :obj:`~prism.Pipeline` object plays a key-role in the *PRISM* framework as it governs all other objects and orchestrates their communications and method calls.
It also performs the process of history matching and refocusing (see the `PRISM paper`_ for the methodology used in *PRISM*).
It is linked to the model by a user-written :obj:`~prism.modellink.ModelLink` object (see :ref:`modellink_crash_course`), allowing the :obj:`~prism.Pipeline` object to extract all necessary model information and call the model.
In order to ensure flexibility and clarity, the *PRISM* framework writes all of its data to one or several `HDF5-files`_ using :mod:`~h5py`, as well as :mod:`~numpy`.

The analysis of a provided model and the construction of the emulator systems for every output value, starts and ends with the :obj:`~prism.Pipeline` object.
When a new emulator is requested, the :obj:`~prism.Pipeline` object creates a large Latin-Hypercube design (LHD) of model evaluation samples to get the construction of the first iteration of the emulator systems started.
To ensure that the maximum amount of information can be obtained from evaluating these samples, a custom Latin-Hypercube sampling code was written.
This produces LHDs that attempt to satisfy both the *maximin* criterion as well as the *correlation* criterion.
This code is customizable through *PRISM* and publicly available in the `e13Tools`_ Python package.

This Latin-Hypercube design is then given to the *Model Evaluator*, which through the provided :obj:`~prism.modellink.ModelLink` object evaluates every sample.
Using the resulting model outputs, the *Active Parameters* for every emulator system (individual data point) can now be determined.
Next, depending on the user, polynomial functions will be constructed by performing an extensive *Regression* process for every emulator system, or this can be skipped in favor of a sole Gaussian analysis (faster, but less accurate).
No matter the choice, the emulator systems now have all the required information to be constructed, which is done by calculating the *Prior Expectation* and *Prior Covariance* values for all evaluated model samples (:math:`\mathrm{E}(D_i)` and :math:`\mathrm{Var}(D_i)`).

Afterward, the emulator systems are fully constructed and are ready to be evaluated and analyzed.
Depending on whether the user wants to prepare for the next emulator iteration or create a projection (see :ref:`projections`), the *Emulator Evaluator* creates one or several LHDs of emulator evaluation samples, and evaluates them in all emulator systems, after which an *Implausibility Check* is carried out.
The samples that survive the check can then either be used to construct the new iteration of emulator systems by sending them to the *Model Evaluator*, or they can be analyzed further by performing a *Projection*.
The :obj:`~prism.Pipeline` object performs a single cycle by default (to allow for user-defined analysis algorithms), but can be easily set to continuously cycle.

In addition to the above, *PRISM* also features a high-level *Message Passing Interface* (MPI) implementation using the Python package :mod:`~mpi4py`.
All emulator systems in *PRISM* can be constructed independently from each other, in any order, and only require to communicate when performing the implausibility cut-off checks during history matching.
Additionally, since different models and/or architectures require different amounts of computational resources, *PRISM* can run on any number of MPI processes (including a single one in serial to accommodate for OpenMP codes) and the same emulator can be used on a different number of MPI processes than it was constructed on (e.g., constructing an emulator using 8 MPI processes and reloading it with 6).
More details on the MPI implementation and its scaling can be found in :ref:`MPI`.

In :ref:`using_prism` and :ref:`modellink_crash_course`, the various components of *PRISM* are described more extensively.

.. _PRISM paper: https://arxiv.org/abs/1901.08725
.. _e13Tools: https://github.com/1313e/e13Tools
.. _HDF5-files: https://portal.hdfgroup.org/display/HDF5/HDF5

.. include:: mpi.inc
