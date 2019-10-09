# -*- coding utf-8 -*-

"""
GUI Globals
===========
Provides a collection of all global variables for the Projection GUI that must
be available.

"""


# %% IMPORTS
# Package imports
from os import path

# PRISM imports
from prism._docstrings import GUI_APP_NAME

# All declaration
__all__ = ['APP_NAME', 'DIR_PATH', 'APP_ICON_PATH']


# %% GUI GLOBALS
APP_NAME = GUI_APP_NAME                             # Name of application
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to GUI directory
APP_ICON_PATH = path.join(DIR_PATH, "data/app_icon.ico")   # App icon path
