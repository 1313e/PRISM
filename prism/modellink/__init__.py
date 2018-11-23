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
from ._gaussian_link import GaussianLink
from ._modellink import ModelLink
from ._sine_wave_link import SineWaveLink

# All declaration
__all__ = ['GaussianLink', 'ModelLink', 'SineWaveLink']
