# -*- coding: utf-8 -*-

"""
PRISM
=====
A *Probabilistic Regression Instrument for Simulating Models* created by
**1313e**.

All docstrings in this package assume that PRISM is imported as::

    import prism


Short description
-----------------
This package contains the PRISM pipeline, an efficient and rapid alternative to
MCMC methods for optimizing and analyzing scientific models.
The PRISM package provides two main classes: The :class`~Pipeline` class and
the :class:`~ModelLink` class.

The :class:`~Pipeline` class provides the user with an 'interface' with all the
tools one needs to utilize the full capabilities of PRISM, while the
:class:`~ModelLink` abstract base class allows for any model to be connected to
the PRISM pipeline and holds all information about this model.


Available modules
-----------------
:mod:`~emulator`
    Provides the definition of the class holding the emulator system of the
    PRISM package, the :class:`~Emulator` class.

:mod:`~modellink`
    Contains the definition of PRISM's :class:`~ModelLink` abstract base class
    and various default/example :class:`~ModelLink` subclasses.

:mod:`~pipeline`
    Provides the definition of the main class of the PRISM package, the
    :class:`~Pipeline` class.

:mod:`~projection`
    Provides the definition of PRISM's :class:`~Projection` class, that allows
    for projection figures detailing a model's behavior to be created.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Import PRISM modules
from .__version__ import version as __version__
from . import modellink
from .pipeline import Pipeline

# All declaration
__all__ = ['modellink', 'Pipeline']
