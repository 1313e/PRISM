# -*- coding: utf-8 -*-

"""
ModelLink
=========
Contains the definition of *PRISM*'s :class:`~ModelLink` abstract base class
and various default/example :class:`~ModelLink` subclasses.

"""


# %% IMPORTS
# Import ModelLink abstract base class and subclasses
from ._gaussian_link import GaussianLink
from ._modellink import ModelLink
from ._poly_link import PolyLink
from ._sine_wave_link import SineWaveLink

# Import modellink modules
from . import utils
from .utils import *

# All declaration
__all__ = ['GaussianLink', 'ModelLink', 'PolyLink', 'SineWaveLink', 'utils']
__all__.extend(utils.__all__)
