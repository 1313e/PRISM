# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from PyQt5 import QtWidgets as QW
import pytest

# PRISM imports
from prism import Pipeline
from prism._gui.widgets.main import MainViewerWindow
from prism.modellink.tests.modellink import GaussianLink3D


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% HELPER FUNCTIONS
@pytest.fixture(scope='session')
def pipe(tmpdir_factory):
    # Get paths to files
    model_data = path.join(
        DIR_PATH, '../../../tests/data/data_gaussian_single.txt')
    model_parameters_3D = path.join(
        DIR_PATH, '../../../tests/data/parameters_gaussian_3D.txt')

    # Create PolyLink object
    modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                   model_data=model_data)

    # Create Pipeline object
    tmpdir = tmpdir_factory.mktemp('test_GUI')
    root_dir = path.dirname(tmpdir.strpath)
    working_dir = path.basename(tmpdir.strpath)
    prism_file = path.join(DIR_PATH, "data/prism_default.txt")
    pipe = Pipeline(modellink_obj, root_dir=root_dir, working_dir=working_dir,
                    prism_par=prism_file)

    # Construct the first and second iteration of the emulator
    pipe.construct(1)
    pipe.construct(2, analyze=False)

    # Return pipe
    return(pipe)


@pytest.fixture(scope='session')
def main_window(qapp, pipe):
    # Create the main_window
    main_window = MainViewerWindow(pipe)

    # Return main_window
    return(main_window)


@pytest.fixture(scope='session')
def menu_actions(main_window):
    # Obtain a list of all menus
    menus = [child for child in main_window.menubar.children()
             if isinstance(child, QW.QMenu)]

    # Go through all menus and obtain their actions
    menu_actions = {}
    for menu in menus:
        actions = {action.text().replace('&', ''): action
                   for action in menu.actions()}
        menu_actions[menu.title().replace('&', '')] = actions

    # Return menu_actions
    return(menu_actions)
