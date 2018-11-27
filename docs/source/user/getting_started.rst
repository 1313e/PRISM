.. _getting_started:

Getting started
===============
Installation
------------
*PRISM* can be easily installed by either cloning the `repository`_ and running ``setup.py`` with::

	python setup.py install

or by installing it directly from `PyPI`_ with::

	pip install prism-tool

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
