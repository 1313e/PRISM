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
from ._modellink import ModelLink, test_subclass
from ._sine_wave_link import SineWaveLink

# All declaration
__all__ = ['GaussianLink', 'ModelLink', 'SineWaveLink', 'test_subclass']
