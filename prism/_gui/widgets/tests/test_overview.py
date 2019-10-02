# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from e13tools.math import nCr
import matplotlib.pyplot as plt
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
from py.path import local
import pytest
from pytest_mpl.plugin import switch_backend

# PRISM imports
from prism import Pipeline
from prism._gui.widgets.overview import OverviewDockWidget
from prism._gui.widgets.helpers import ThreadedProgressDialog
from prism.modellink import PolyLink

# Set the current working directory to the temporary directory
local.get_temproot().chdir()

# Ignore any RequestWarnings
pytestmark = pytest.mark.filterwarnings(
    "ignore::prism._internal.RequestWarning")


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% HELPER FUNCTIONS


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the overview dock widget
class TestOverviewDockWidget(object):
    # Make abbreviation for overview
    @pytest.fixture(scope='class')
    def overview(self, main_window):
        return(main_window.overview_dock)

    # Test if it is bound to main_window
    def test_bound_main(self, main_window, overview):
        assert any([isinstance(child, OverviewDockWidget)
                    for child in main_window.children()])
        assert isinstance(overview, OverviewDockWidget)

    # Test if the overview contains a specific number of hcubes
    def test_hcubes(self, pipe, overview):
        # Calculate how many hcubes there should be
        n_hcubes = sum(map(len, pipe._emulator._active_par))

        # Check if this many hcubes are currently in the GUI
        assert (len(overview.hcubes) == n_hcubes)

    # Test if currently the three lists have the correct lay-out
    def test_proj_lists(self, overview):
        # Check if the number of items in each is correct
        assert (len(overview.proj_list_d) == 0)
        assert (len(overview.proj_list_a) == 0)
        assert (len(overview.proj_list_u) == len(overview.hcubes))

    # Test if the standard 'unavailable' action works
    def test_default_action_u(self, qtbot, overview, monkeypatch):
        # Monkey patch the ThreadedProgressDialog.show function
        monkeypatch.setattr(ThreadedProgressDialog, 'show', lambda *args: None)

        # Select the first item in the list
        overview.proj_list_u.setCurrentRow(0)

        # Check if the first item is indeed selected
        assert (overview.proj_list_u.selectedItems() ==
                [overview.proj_list_u.item(0)])
        item = overview.proj_list_u.item(0)

        # Press enter
        qtbot.keyPress(overview.proj_list_u, QC.Qt.Key_Enter)
        qtbot.wait(1000)

        # Check if this item is now in the available list
        assert (len(overview.proj_list_a) == 1)
        assert (len(overview.proj_list_u) == len(overview.hcubes)-1)
        assert (overview.proj_list_a.item(0) == item)

    # Test if the standard 'available' action works
    def test_default_action_a(self, qtbot, overview, monkeypatch):
        # Monkey patch the ThreadedProgressDialog.show function
        monkeypatch.setattr(ThreadedProgressDialog, 'show', lambda *args: None)

        # Monkey patch the plt.show function
        monkeypatch.setattr(plt, 'show', lambda *args: None)

        # Select the first item in the list
        overview.proj_list_a.setCurrentRow(0)

        # Check if the first item is indeed selected
        assert (overview.proj_list_a.selectedItems() ==
                [overview.proj_list_a.item(0)])
        item = overview.proj_list_a.item(0)

        # Press enter
        qtbot.keyPress(overview.proj_list_a, QC.Qt.Key_Enter)
        qtbot.wait(1000)

        # Check if this item is now in the drawn list
        assert (len(overview.proj_list_d) == 1)
        assert (len(overview.proj_list_a) == 0)
        assert (overview.proj_list_d.item(0) == item)

    # Test if the standard 'drawn' action works
    def test_default_action_d(self, qtbot, overview, monkeypatch):
        # Monkey patch the plt.show function
        monkeypatch.setattr(plt, 'show', lambda *args: None)

        # Select the first item in the list
        overview.proj_list_d.setCurrentRow(0)

        # Check if the first item is indeed selected
        assert (overview.proj_list_d.selectedItems() ==
                [overview.proj_list_d.item(0)])

        # Press enter
        qtbot.keyPress(overview.proj_list_d, QC.Qt.Key_Enter)

    # Test if the three context menus can be shown
    def test_context_menus(self, qtbot, overview, monkeypatch):
        # Monkey patch the ThreadedProgressDialog.show function
        monkeypatch.setattr(ThreadedProgressDialog, 'show', lambda *args: None)

        # First, create an other projection figure
        overview.proj_list_u.setCurrentRow(0)
        qtbot.keyPress(overview.proj_list_u, QC.Qt.Key_Enter)
        qtbot.wait(1000)

        # Make sure there is at least 1 item in every list now
        assert len(overview.proj_list_d)
        assert len(overview.proj_list_a)
        assert len(overview.proj_list_u)

        # Select the first item in every list
        overview.proj_list_d.setCurrentRow(0)
        overview.proj_list_a.setCurrentRow(0)
        overview.proj_list_u.setCurrentRow(0)

        # Open the context menu for every list
        qtbot.mouseClick(overview.proj_list_d, QC.Qt.RightButton)
        qtbot.mouseClick(overview.proj_list_a, QC.Qt.RightButton)
        qtbot.mouseClick(overview.proj_list_u, QC.Qt.RightButton)
