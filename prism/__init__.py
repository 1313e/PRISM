# -*- coding: utf-8 -*-

"""
PRISM
=====
A *Probabilistic Regression Instrument for Simulating Models* created by
**Ellert van der Velden** (1313e).

All docstrings in this package assume that *PRISM* is imported as::

    import prism


Short description
-----------------
This package contains the *PRISM* pipeline, an efficient and rapid alternative
to MCMC methods for optimizing and analyzing scientific models.
The *PRISM* package provides two user classes: The :class:`~Pipeline` class and
the :class:`~ModelLink` class.

The :class:`~Pipeline` class provides the user with an environment with all the
tools one needs to utilize the full capabilities of *PRISM*, while the
:class:`~ModelLink` abstract base class allows for any model to be connected to
the *PRISM* pipeline and holds all information about this model.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import os
import warnings

# Import PRISM modules
from .__version__ import prism_version as __version__
from . import _internal
from . import modellink
from .modellink import ModelLink
from .pipeline import Pipeline

# All declaration
__all__ = ['modellink', 'ModelLink', 'Pipeline']

# Author declaration
__author__ = "Ellert van der Velden"


# %% EXECUTE INITIALIZING CODE
# Import PRISM's custom cmaps
_internal.import_cmaps()

# Check if MPI is being used
try:
    from mpi4py import MPI
except ImportError:
    pass
else:
    # If so, raise warning if OMP_NUM_THREADS is not set to 1 with MPI_size > 1
    if(os.environ.get('OMP_NUM_THREADS') != '1' and
       MPI.COMM_WORLD.Get_size() > 1 and MPI.COMM_WORLD.Get_rank() == 0):
        warn_msg = ("Environment variable 'OMP_NUM_THREADS' is currently not "
                    "set to 1 (%s), with MPI enabled. Unless this was "
                    "intentional, it is advised to set 'OMP_NUM_THREADS' to 1."
                    % (os.environ.get('OMP_NUM_THREADS')))
        warnings.warn(warn_msg)
