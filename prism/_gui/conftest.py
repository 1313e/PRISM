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
from prism._pipeline import WorkerMode
from prism.modellink.tests.modellink import GaussianLink3D


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% PYTEST CUSTOM CONFIGURATION PLUGINS
# Add an attribute to PRISM stating that pytest is being used
def pytest_configure(config):
    import prism
    prism.__PYTEST = True


# After pytest has finished, remove this attribute again
def pytest_unconfigure(config):
    import prism
    del prism.__PYTEST


# %% HELPER FUNCTIONS
@pytest.fixture(scope='session')
def pipe_GUI(tmpdir_factory):
    # Get paths to files
    model_data = path.join(
        DIR_PATH, '../tests/data/data_gaussian_single.txt')
    model_parameters_3D = path.join(
        DIR_PATH, '../tests/data/parameters_gaussian_3D.txt')

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


@pytest.fixture(scope='module')
def main_window(qapp, request, pipe_GUI):
    # Initialize worker mode
    worker_mode = WorkerMode(pipe_GUI)

    # Request for the worker_mode to be closed at the end
    request.addfinalizer(lambda: exit_worker_mode(worker_mode))

    # Enter worker mode
    worker_mode.__enter__()

    # All workers skip all tests at module level
    if pipe_GUI._is_worker:
        pytest.skip("Worker ranks are in worker mode", allow_module_level=True)

    # Controller only
    if pipe_GUI._is_controller:
        # Create the main_window
        main_window = MainViewerWindow(pipe_GUI)

        # Request for the main_window to be closed at the end
        request.addfinalizer(lambda: close_main_window(main_window))

        # Return main_window
        return(main_window)


def close_main_window(main_window_obj):
    main_window_obj.close()


def exit_worker_mode(worker_mode_obj):
    worker_mode_obj.__exit__()


@pytest.fixture(scope='module')
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
