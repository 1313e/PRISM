# -*- coding utf-8 -*-

"""
GUI Widgets
===========
Contains the various different :class:`~PyQt5.QtWidgets.QWidget` subclasses
created for the Projection GUI.
The widgets required for the preferences menu can be found in
:mod:`~prism._gui.widgets.preferences`.

"""


# %% IMPORTS
# Import base modules
from . import helpers
from .helpers import *
from . import main, overview, viewing_area
from .main import *
from .overview import *
from .viewing_area import *

# Import subpackages
from . import preferences

# All declaration
__all__ = ['helpers', 'main', 'overview', 'preferences', 'viewing_area']
__all__.extend(helpers.__all__)
__all__.extend(main.__all__)
__all__.extend(overview.__all__)
__all__.extend(viewing_area.__all__)
