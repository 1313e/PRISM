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
_internal.import_cmaps()
