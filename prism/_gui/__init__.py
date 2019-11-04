# -*- coding utf-8 -*-

"""
Projection GUI
==============
Contains all definitions required for drawing *PRISM*'s Projection GUI, an
interface that allows users to interactively look at the various different
projection figures that are produced by the :class:`~prism.Pipeline`.

"""


# %% IMPORTS
# Import globals
from ._globals import *

# Import base modules
from . import core
from .core import *

# Import subpackages
from . import widgets

# All declaration
__all__ = ['core', 'widgets']
__all__.extend(core.__all__)
