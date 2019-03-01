# -*- coding: utf-8 -*-

"""
ModelLink
=========
Contains the definition of *PRISM*'s :class:`~ModelLink` abstract base class
and various default/example :class:`~ModelLink` subclasses.

"""


# %% IMPORTS
# Import modellink modules
from ._gaussian_link import GaussianLink
from ._modellink import ModelLink
from ._sine_wave_link import SineWaveLink
from . import utils
from .utils import *

# All declaration
__all__ = ['GaussianLink', 'ModelLink', 'SineWaveLink', 'utils']
__all__.extend(utils.__all__)
