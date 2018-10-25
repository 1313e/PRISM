# -*- coding: utf-8 -*-

"""
Utilities
=========
Provides a collection of modules useful for using/mixing *PRISM* with other
applications.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Import utils modules
from . import mcmc
from .mcmc import *

# All declaration
__all__ = ['mcmc']
__all__.extend(mcmc.__all__)
