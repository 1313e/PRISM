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
from sys import platform
import warnings

# Import PRISM modules
from .__version__ import prism_version as __version__
from . import modellink
from . import utils
from ._emulator import Emulator
from ._internal import import_cmaps
from ._pipeline import Pipeline

# All declaration
__all__ = ['modellink', 'utils', 'Emulator', 'Pipeline', 'import_cmaps']

# Author declaration
__author__ = "Ellert van der Velden (1313e)"


# %% EXECUTE INITIALIZING CODE
# Import PRISM's custom cmaps
import_cmaps(os.path.join(os.path.dirname(__file__), 'data'))

# Check if MPI is being used
try:
    from mpi4py import MPI as _MPI
except ImportError:
    pass
else:
    # If so, raise warning if OMP_NUM_THREADS is not set to 1 with MPI_size > 1
    if(os.environ.get('OMP_NUM_THREADS') != '1' and
       _MPI.COMM_WORLD.Get_size() > 1 and _MPI.COMM_WORLD.Get_rank() == 0):
        # Get platform-dependent string on how to set environment variable
        # Windows
        if platform.startswith('win'):
            set_str = "\">set OMP_NUM_THREADS=1\""
        # Linux/MacOS-X
        elif platform.startswith(('linux', 'darwin')):
            set_str = "\"$ export OMP_NUM_THREADS=1\""
        # Anything else
        else:
            set_str = "N/A"

        # Print warning message
        warn_msg = ("Environment variable 'OMP_NUM_THREADS' is currently not "
                    "set to 1 (%s), with MPI enabled. Unless this was "
                    "intentional, it is advised to set it to 1 (%s)."
                    % (os.environ.get('OMP_NUM_THREADS'), set_str))
        warnings.warn(warn_msg, stacklevel=2)
