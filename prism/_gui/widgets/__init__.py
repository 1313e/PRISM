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
from . import core
from .core import *
from . import base_layouts, base_widgets, helpers
from .base_layouts import *
from .base_widgets import *
from .helpers import *
from . import main, overview, viewing_area
from .main import *
from .overview import *
from .viewing_area import *

# Import subpackages
from . import preferences

# All declaration
__all__ = ['base_layouts', 'base_widgets', 'core', 'helpers', 'main',
           'overview', 'preferences', 'viewing_area']
__all__.extend(base_layouts.__all__)
__all__.extend(base_widgets.__all__)
__all__.extend(core.__all__)
__all__.extend(helpers.__all__)
__all__.extend(main.__all__)
__all__.extend(overview.__all__)
__all__.extend(viewing_area.__all__)
