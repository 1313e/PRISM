# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
import pytest


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% HELPER FUNCTIONS
# Make abbreviation for options
@pytest.fixture(scope='module')
def options(main_window):
    return(main_window.options)


# Make abbreviation for option_entries
@pytest.fixture(scope='module')
def option_entries(options):
    return(options.option_entries)
