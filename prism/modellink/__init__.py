# -*- coding: utf-8 -*-

"""
ModelLink
=========
Contains the definition of PRISM's :class:`~ModelLink` abstract base class and
various default/example :class:`~ModelLink` subclasses.

Available modules
-----------------
:mod:`~modellink`
    Provides the definition of the :class:`~ModelLink` abstract base class.
:mod:`~sine_wave_link`
    Provides the definition of the :class:`~SineWaveLink` class.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Import modellink modules
from .modellink import ModelLink
from .sine_wave_link import SineWaveLink
#from .constant_magnetic_field_link import ConstantMagneticFieldLink

# All declaration
__all__ = ['ModelLink', 'SineWaveLink']
