# -*- coding: utf-8 -*-

"""
Utilities
=========
Provides a collection of functions useful for using/mixing *PRISM* with other
applications.

"""


# %% IMPORTS
# Import utils modules
from . import mcmc
from .mcmc import *

# All declaration
__all__ = ['mcmc']
__all__.extend(mcmc.__all__)
