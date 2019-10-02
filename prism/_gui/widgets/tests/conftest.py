# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from py.path import local
import pytest

# PRISM imports
from prism import Pipeline
from prism._gui.widgets.main import MainViewerWindow
from prism.modellink.tests.modellink import GaussianLink2D

# Set the current working directory to the temporary directory
local.get_temproot().chdir()


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% HELPER FUNCTIONS
@pytest.fixture(scope='session')
def pipe(tmpdir_factory):
    # Create PolyLink object
    modellink_obj = GaussianLink2D()

    # Create Pipeline object
    tmpdir = tmpdir_factory.mktemp('test2D')
    root_dir = path.dirname(tmpdir.strpath)
    working_dir = path.basename(tmpdir.strpath)
    prism_file = path.join(DIR_PATH, "../../../tests/data/prism_default.txt")
    pipe = Pipeline(modellink_obj, root_dir=root_dir, working_dir=working_dir,
                    prism_par=prism_file)

    # Construct the first and second iteration of the emulator
    pipe.construct(1)
    pipe.construct(2)

    # Return pipe
    return(pipe)


@pytest.fixture(scope='session')
def main_window(pipe):
    # Create the main_window
    main_window = MainViewerWindow(pipe)

    # Return main_window
    return(main_window)
