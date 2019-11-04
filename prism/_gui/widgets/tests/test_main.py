# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from mpi4pyd import MPI
from PyQt5 import QtGui as QG, QtWidgets as QW
import pytest

# PRISM imports
from prism._gui.widgets.main import MainViewerWindow
from prism._internal import RequestWarning


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the file menu
class TestMenus_File(object):
    # Test the save view action
    def test_save_view(self, pipe_GUI, monkeypatch, menu_actions):
        # Check if the 'save view as' action is in the proper menu
        assert 'Save view as...' in menu_actions['File']

        # Set expected filename
        filename = path.join(pipe_GUI._working_dir, "test_save_view.png")

        # Monkey patch the QFileDialog.getSaveFileName function
        monkeypatch.setattr(QW.QFileDialog, 'getSaveFileName',
                            lambda *args, **kwargs: (filename, None))

        # Save the current view by triggering its action
        menu_actions['File']['Save view as...'].trigger()

        # Check if a view was vsaved
        assert path.exists(filename)


# Pytest for the tools menu
class TestMenus_Tools(object):
    # Test the cascade option
    def test_cascade(self, menu_actions):
        # Check if the 'cascade' action is in the proper menu
        assert 'Cascade' in menu_actions['Tools']

        # Cascade the subwindows by triggering its action
        menu_actions['Tools']['Cascade'].trigger()

    # Test the tile option
    def test_tile(self, menu_actions):
        # Check if the 'tile' action is in the proper menu
        assert 'Tile' in menu_actions['Tools']

        # Tile the subwindows by triggering its action
        menu_actions['Tools']['Tile'].trigger()

    # Test the close all option
    def test_close_all(self, menu_actions):
        # Check if the 'close all' action is in the proper menu
        assert 'Close all' in menu_actions['Tools']

        # Close all the subwindows by triggering its action
        menu_actions['Tools']['Close all'].trigger()


# Pytest for the help menu
class TestMenus_Help(object):
    # Test the about window
    def test_about(self, monkeypatch, menu_actions):
        # Check if the 'about' action is in the proper menu
        assert 'About...' in menu_actions['Help']

        # Monkey patch the QMessageBox.about function
        monkeypatch.setattr(QW.QMessageBox, 'about',
                            lambda *args: QW.QMessageBox.Ok)

        # Show the about window by triggering its action
        menu_actions['Help']['About...'].trigger()

    # Test the details window
    def test_details(self, monkeypatch, menu_actions):
        # Check if the 'details' action is in the proper menu
        assert 'Details' in menu_actions['Help']

        # Monkey patch the QW.QDialog.show function
        monkeypatch.setattr(QW.QDialog, 'show', lambda *args: None)

        # Show the details window by triggering its action
        menu_actions['Help']['Details'].trigger()

    # Test the API reference action
    def test_api_reference(self, monkeypatch, menu_actions):
        # Check if the 'API reference' action is in the proper menu
        assert 'API reference' in menu_actions['Help']

        # Monkey patch the QG.QDesktopServices.openUrl function
        monkeypatch.setattr(QG.QDesktopServices, 'openUrl', lambda *args: None)

        # Show the API reference by triggering its action
        menu_actions['Help']['API reference'].trigger()


# Test if an implausible iteration is automatically skipped
@pytest.mark.last
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                    reason="Cannot be pytested in MPI")
def test_implausible_emul_i(pipe_GUI):
    # Analyze the second iteration with low implausibility cut-offs
    pipe_GUI.analyze(impl_cut=[0.001, 0.001, 0.001])

    # Initialize the main window
    with pytest.warns(RequestWarning):
        main_window = MainViewerWindow(pipe_GUI)

    # Check that the emul_i is currently set to 1
    assert (pipe_GUI._Projection__emul_i == 1)

    # Close the main_window
    main_window.close()
