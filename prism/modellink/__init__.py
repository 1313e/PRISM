# -*- coding: utf-8 -*-

"""
ModelLink
=========
Contains the definition of *PRISM*'s :class:`~ModelLink` abstract base class
and various default/example :class:`~ModelLink` subclasses.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Import modellink modules
from .gaussian_link import GaussianLink
from .modellink import ModelLink
from .sine_wave_link import SineWaveLink
from . import tests

# All declaration
__all__ = ['GaussianLink', 'ModelLink', 'SineWaveLink', 'tests']
