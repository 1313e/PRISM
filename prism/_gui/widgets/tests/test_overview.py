# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import os
from os import path
from sys import platform

# Package imports
from e13tools.math import nCr
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
import matplotlib.pyplot as plt
from mpi4pyd import MPI
from PyQt5 import QtCore as QC, QtWidgets as QW
import pytest

# PRISM imports
from prism._gui.widgets.core import set_box_value
from prism._gui.widgets.overview import OverviewDockWidget


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% HELPER FUNCTIONS
# Make abbreviation for overview
@pytest.fixture(scope='module')
def overview(main_window):
    return(main_window.overview_dock)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the overview dock widget, main properties
@pytest.mark.incremental
class TestOverviewDockWidget_Main(object):
    # Test if it is bound to main_window
    def test_bound_main(self, main_window, overview):
        assert any([isinstance(child, OverviewDockWidget)
                    for child in main_window.children()])
        assert isinstance(overview, OverviewDockWidget)

    # Test if the overview contains a specific number of hcubes
    def test_hcubes(self, pipe_GUI, overview):
        # Calculate how many hcubes there should be
        emul_i = overview.pipe._Projection__emul_i
        n_hcubes = sum(
            [nCr(n, 2)+n for n in map(
                len, pipe_GUI._emulator._active_par[:emul_i+1])])

        # Check if this many hcubes are currently in the GUI
        assert (len(overview.hcubes) == n_hcubes)

    # Test if currently the three lists have the correct lay-out
    def test_proj_lists(self, overview):
        # Check if the number of items in each is correct
        assert (len(overview.proj_list_d) == 0)
        assert (len(overview.proj_list_a) == 0)
        assert (len(overview.proj_list_u) == len(overview.hcubes))

    # Test if the standard 'unavailable' action works
    def test_default_action_u(self, qtbot, overview):
        # Select two items in the list
        overview.proj_list_u.setCurrentRow(0, QC.QItemSelectionModel.Select)
        overview.proj_list_u.setCurrentRow(len(overview.hcubes)-1,
                                           QC.QItemSelectionModel.Select)

        # Press enter
        qtbot.keyClick(overview.proj_list_u, QC.Qt.Key_Enter)

        # Check if two items are in available now
        assert (len(overview.proj_list_u) == len(overview.hcubes)-2)
        assert (len(overview.proj_list_a) == 2)

    # Test if the standard 'available' action works
    def test_default_action_a(self, qtbot, overview):
        # Check that currently no items are selected
        assert not len(overview.proj_list_a.selectedItems())

        # Select all items in the list
        qtbot.keyClick(overview.proj_list_a, QC.Qt.Key_A,
                       QC.Qt.ControlModifier)

        # Check that currently all items are selected
        assert (len(overview.proj_list_a.selectedItems()) == 2)

        # Press enter
        qtbot.keyClick(overview.proj_list_a, QC.Qt.Key_Enter)

        # Check if all items are in drawn now
        assert (len(overview.proj_list_a) == 0)
        assert (len(overview.proj_list_d) == 2)

    # Test if the standard 'drawn' action works
    def test_default_action_d(self, qtbot, overview):
        # Select all items in the list
        overview.proj_list_d.setCurrentRow(0)
        overview.proj_list_d.selectAll()

        # Press enter
        qtbot.keyClick(overview.proj_list_d, QC.Qt.Key_Enter)


# Pytest for testing the drawn list
@pytest.mark.incremental
class TestOverviewListWidget_Drawn(object):
    # Make abbreviation for proj_list
    @pytest.fixture(scope='class')
    def proj_list(self, overview):
        return(overview.proj_list_d)

    # Make abbreviation for context menu
    @pytest.fixture(scope='class')
    def menu(self, overview):
        return(overview.context_menu_d)

    # Create dict of all actions in this context menu
    @pytest.fixture(scope='class')
    def actions(self, menu):
        # Obtain a dict of all actions in this context menu
        actions = {action.text().replace('&', ''): action
                   for action in menu.actions()}

        # Return actions
        return(actions)

    # Test if the context menu can be shown
    def test_context_menu(self, qtbot, overview, proj_list, menu):
        # Make sure that there is at least 1 item in the list
        assert len(proj_list)

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Try to show the context menu
        with qtbot.waitSignal(menu.aboutToHide):
            overview.show_drawn_context_menu()
            menu.hide()

    # Test the show action
    def test_show_action(self, menu_actions, actions, proj_list, overview):
        # Make sure that this action is valid
        assert 'Show' in actions

        # Close all subwindows
        menu_actions['Tools']['Close all']

        # Trigger the show action
        actions['Show'].trigger()

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()

        # Check if its corresponding entry exists
        assert hcube_name in overview.proj_fig_registry

        # Check the entry itself
        proj_fig_entry = overview.proj_fig_registry[hcube_name]
        assert isinstance(proj_fig_entry[0], plt.Figure)
        assert isinstance(proj_fig_entry[1], QW.QMdiSubWindow)
        assert isinstance(proj_fig_entry[1].widget(), FigureCanvas)

    # Test the save action
    def test_save_action(self, actions, proj_list, overview):
        # Make sure that this action is valid
        assert 'Save' in actions

        # Trigger the action
        actions['Save'].trigger()

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()
        hcube = overview.hcubes[overview.names.index(hcube_name)]

        # Check if a new figure has been saved
        figpath = overview.call_proj_attr('get_fig_path', hcube)[0]
        assert path.exists(figpath)

    # Test the save as action
    def test_save_as_action(self, pipe_GUI, actions, monkeypatch):
        # Make sure that this action is valid
        assert 'Save as...' in actions

        # Set custom figpath
        figpath = path.join(pipe_GUI._working_dir, "test_save_as.png")

        # Monkey patch the QFileDialog.getSaveFileName function
        monkeypatch.setattr(QW.QFileDialog, 'getSaveFileName',
                            lambda *args, **kwargs: (figpath, None))

        # Trigger the action
        actions['Save as...'].trigger()

        # Check if a new figure has been saved
        assert path.exists(figpath)
        os.remove(figpath)

        # Monkey patch the QFileDialog.getSaveFileName function again
        monkeypatch.setattr(QW.QFileDialog, 'getSaveFileName',
                            lambda *args, **kwargs: ('', None))

        # Trigger the action again
        actions['Save as...'].trigger()

        # Check that no new figure has been saved
        assert not path.exists(figpath)

    # Test the redraw action
    def test_redraw_action(self, actions, proj_list, overview):
        # Make sure that this action is valid
        assert 'Redraw' in actions

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()

        # Save the current figure entry
        proj_fig_entry = overview.proj_fig_registry[hcube_name]

        # Trigger the action
        actions['Redraw'].trigger()

        # Check that the entry has been replaced
        assert overview.proj_fig_registry[hcube_name] is not proj_fig_entry

        # Check that currently no item has been selected
        assert (len(proj_list.selectedItems()) == 0)

    # Test the details action
    def test_details_action(self, qtbot, overview, menu, actions, proj_list,
                            monkeypatch):
        # Make sure that this action is valid
        assert 'Details' in actions

        # Monkey patch the QW.QDialog.show function
        monkeypatch.setattr(QW.QDialog, 'show', lambda *args: None)

        # Select the first item
        proj_list.setCurrentRow(0)

        # Trigger the action
        actions['Details'].trigger()

        # Select all items
        proj_list.selectAll()

        # Call the context menu
        with qtbot.waitSignal(menu.aboutToHide):
            overview.show_drawn_context_menu()
            menu.hide()

        # Check that currently the details action is disabled
        assert not actions['Details'].isEnabled()

    # Test the close action
    def test_close_action(self, actions, proj_list, overview):
        # Make sure that this action is valid
        assert 'Close' in actions

        # Select all items
        proj_list.setCurrentRow(0)
        proj_list.selectAll()

        # Trigger the action
        actions['Close'].trigger()

        # Check that there are no figures left drawn
        assert (len(proj_list) == 0)
        assert (len(overview.proj_fig_registry) == 0)


# Pytest for testing the available list
@pytest.mark.incremental
class TestOverviewListWidget_Available(object):
    # Make abbreviation for proj_list
    @pytest.fixture(scope='class')
    def proj_list(self, overview):
        return(overview.proj_list_a)

    # Make abbreviation for context menu
    @pytest.fixture(scope='class')
    def menu(self, overview):
        return(overview.context_menu_a)

    # Create dict of all actions in this context menu
    @pytest.fixture(scope='class')
    def actions(self, menu):
        # Obtain a dict of all actions in this context menu
        actions = {action.text().replace('&', ''): action
                   for action in menu.actions()}

        # Return actions
        return(actions)

    # Test if the context menu can be shown
    def test_context_menu(self, qtbot, overview, proj_list, menu):
        # Make sure that there is at least 1 item in the list
        assert len(proj_list)

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Try to show the context menu
        with qtbot.waitSignal(menu.aboutToHide):
            overview.show_available_context_menu()
            menu.hide()

    # Test the draw action
    def test_draw_action(self, actions, proj_list, overview):
        # Make sure that this action is valid
        assert 'Draw' in actions

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()

        # Trigger the show action
        actions['Draw'].trigger()

        # Check if its corresponding entry exists
        assert hcube_name in overview.proj_fig_registry

        # Check the entry itself
        proj_fig_entry = overview.proj_fig_registry[hcube_name]
        assert isinstance(proj_fig_entry[0], plt.Figure)
        assert isinstance(proj_fig_entry[1], QW.QMdiSubWindow)
        assert isinstance(proj_fig_entry[1].widget(), FigureCanvas)

        # Close the figure that was just drawn
        overview.close_projection_figures([item])

    # Test the draw/save action
    def test_draw_save_action(self, actions, proj_list, overview, main_window):
        # Make sure that this action is valid
        assert 'Draw  Save' in actions

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()
        hcube = overview.hcubes[overview.names.index(hcube_name)]

        # Turn the option for smoothing on
        set_box_value(main_window.options.option_entries['smooth'].box, True)
        main_window.options.save_options()

        # Trigger the action
        actions['Draw  Save'].trigger()

        # Set the options back to default
        main_window.options.reset_options()

        # Check if a new figure has been saved
        figpath = overview.call_proj_attr('get_fig_path', hcube)[1]
        assert path.exists(figpath)

        # Close the figure that was just drawn
        overview.close_projection_figures([item])

    # Test the recreate action
    def test_recreate_action(self, actions, monkeypatch, proj_list):
        # Make sure that this action is valid
        assert 'Recreate' in actions

        # Monkey patch the QMessageBox.warning function
        monkeypatch.setattr(QW.QMessageBox, 'warning',
                            lambda *args, **kwargs: QW.QMessageBox.Yes)

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Save the item that is currently selected
        item = proj_list.item(0)

        # Trigger the action
        actions['Recreate'].trigger()

        # Check if the item is still in the proj_list
        assert (proj_list.row(item) == 0)

    # Test the details action
    def test_details_action(self, qtbot, overview, menu, actions, proj_list,
                            monkeypatch):
        # Make sure that this action is valid
        assert 'Details' in actions

        # Monkey patch the QW.QDialog.show function
        monkeypatch.setattr(QW.QDialog, 'show', lambda *args: None)

        # Create a projection figure first
        item = overview.proj_list_u.item(len(overview.proj_list_u)-1)
        overview._create_projection_figure(item)

        # Select the created item
        proj_list.setCurrentItem(item)

        # Trigger the action
        actions['Details'].trigger()

        # Select all items
        proj_list.selectAll()

        # Call the context menu
        with qtbot.waitSignal(menu.aboutToHide):
            overview.show_available_context_menu()
            menu.hide()

        # Check that currently the details action is disabled
        assert not actions['Details'].isEnabled()

    # Test the delete action
    def test_delete_action(self, actions, proj_list, overview, monkeypatch):
        # Make sure that this action is valid
        assert 'Delete' in actions

        # Monkey patch the QMessageBox.warning function
        monkeypatch.setattr(QW.QMessageBox, 'warning',
                            lambda *args, **kwargs: QW.QMessageBox.Yes)

        # Select all items
        proj_list.setCurrentRow(0)
        proj_list.selectAll()

        # Trigger the action
        actions['Delete'].trigger()

        # Check that there are no figures left available
        assert (len(proj_list) == 0)


# Pytest for testing the unavailable list
class TestOverviewListWidget_Unavailable(object):
    # Make abbreviation for proj_list
    @pytest.fixture(scope='class')
    def proj_list(self, overview):
        return(overview.proj_list_u)

    # Make abbreviation for context menu
    @pytest.fixture(scope='class')
    def menu(self, overview):
        return(overview.context_menu_u)

    # Create dict of all actions in this context menu
    @pytest.fixture(scope='class')
    def actions(self, menu):
        # Obtain a dict of all actions in this context menu
        actions = {action.text().replace('&', ''): action
                   for action in menu.actions()}

        # Return actions
        return(actions)

    # Test if the context menu can be shown
    def test_context_menu(self, qtbot, overview, proj_list, menu):
        # Make sure that there is at least 1 item in the list
        assert len(proj_list)

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Try to show the context menu
        with qtbot.waitSignal(menu.aboutToHide):
            overview.show_unavailable_context_menu()
            menu.hide()

    # Test the create action
    def test_create_action(self, main_window, actions, proj_list, overview,
                           monkeypatch):
        # Make sure that this action is valid
        assert 'Create' in actions

        # Make sure that the first item is selected
        proj_list.setCurrentRow(0)

        # Get the selected item
        item = proj_list.item(0)

        # Trigger the action
        actions['Create'].trigger()

        # Check that this item has indeed moved to available
        assert (overview.proj_list_a.row(item) == 0)
        assert (len(overview.proj_list_a) == 1)

        # Monkey patch the QMessageBox.warning function
        monkeypatch.setattr(QW.QMessageBox, 'warning',
                            lambda *args, **kwargs: QW.QMessageBox.Yes)

        # Delete the projection figure again
        overview.delete_projection_figures([item])

    # Test the create/draw action
    # TODO: Figure out why this test stalls forever on Linux, Travis CI in MPI
    @pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1 and
                        'TRAVIS' in os.environ and
                        platform.startswith('linux'),
                        reason="Cannot be tested in MPI on Linux on Travis CI")
    def test_create_draw_action(self, actions, proj_list, overview,
                                main_window):
        # Make sure that this action is valid
        assert 'Create  Draw' in actions

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()

        # Use a threaded progress dialog for this
        box = main_window.options.option_entries['use_progress_dialog'].box
        set_box_value(box, True)
        main_window.options.save_options()

        # Trigger the show action
        actions['Create  Draw'].trigger()

        # Do not use a threaded progress dialog anymore
        main_window.options.reset_options()

        # Check if its corresponding entry exists
        assert hcube_name in overview.proj_fig_registry

        # Check the entry itself
        proj_fig_entry = overview.proj_fig_registry[hcube_name]
        assert isinstance(proj_fig_entry[0], plt.Figure)
        assert isinstance(proj_fig_entry[1], QW.QMdiSubWindow)
        assert isinstance(proj_fig_entry[1].widget(), FigureCanvas)

    # Test the create/draw/save action
    def test_create_draw_save_action(self, actions, proj_list, overview,
                                     main_window):
        # Make sure that this action is valid
        assert 'Create, Draw  Save' in actions

        # Make sure the first item is selected
        proj_list.setCurrentRow(0)

        # Get the name of the selected hcube item
        item = proj_list.item(0)
        hcube_name = item.text()
        hcube = overview.hcubes[overview.names.index(hcube_name)]

        # Trigger the action
        actions['Create, Draw  Save'].trigger()

        # Check if a new figure has been saved
        figpath = overview.call_proj_attr('get_fig_path', hcube)[0]
        assert path.exists(figpath)
