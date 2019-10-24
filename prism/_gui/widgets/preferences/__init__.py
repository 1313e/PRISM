# -*- coding utf-8 -*-

"""
GUI Preferences
===============
Contains all different :class:`~PyQt5.QtWidgets.QWidget` subclasses required
for making the 'preferences' menu in the Projection GUI.

"""


# %% IMPORTS
# Import base modules
from . import custom_boxes, kwargs_dicts, options
from .custom_boxes import *
from .kwargs_dicts import *
from .options import *

# All declaration
__all__ = ['custom_boxes', 'kwargs_dicts', 'options']
__all__.extend(custom_boxes.__all__)
__all__.extend(kwargs_dicts.__all__)
__all__.extend(options.__all__)
