# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from PyQt5 import QtCore as QC
import pytest

# PRISM imports
from prism._gui.widgets.core import get_box_value
from prism._gui.widgets.preferences.options import OptionsDialog, OptionsEntry


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the options dialog widget, main properties
@pytest.mark.incremental
class TestOptionsDialog_Main(object):
    # Test if it is bound to main_window
    def test_bound_main(self, main_window, options):
        assert any([isinstance(child, OptionsDialog)
                    for child in main_window.children()])
        assert isinstance(options, OptionsDialog)

    # Test that the options menu has a dict with option entries
    def test_option_entries(self, options, option_entries):
        assert isinstance(option_entries, dict)
        for key, entry in option_entries.items():
            assert isinstance(key, str)
            assert isinstance(entry, OptionsEntry)

    # Test that the options entries all have the proper values
    def test_option_values(self, options, option_entries):
        # Test that the box value, default value and entry value are equal
        for key, entry in option_entries.items():
            assert (key == entry.name)
            assert (get_box_value(entry.box) == entry.default)
            assert (entry.value == get_box_value(entry.box))
            repr(entry)

    # Test the get_option function
    def test_get_option(self, main_window, options, option_entries):
        # Check that main_window also has this method
        assert hasattr(main_window, 'get_option')
        assert (main_window.get_option == options.get_option)

        # Request the values of all options entries and check them
        for key, entry in option_entries.items():
            assert options.get_option(key) is entry.value

    # Test that the options menu can be properly opened
    def test_open(self, menu_actions, options):
        # Check if the 'preferences' action is in the proper menu
        assert 'Preferences' in menu_actions['Help']

        # Show the options menu by triggering its action
        menu_actions['Help']['Preferences'].trigger()

        # Check if the options menu is currently open
        assert options.isVisible()

    # Test if an option can have its value changed
    def test_change_option_value_save(self, qtbot, options, option_entries):
        # Make sure that the options menu is still visible
        assert options.isVisible()

        # Make sure that the save button is currently disabled
        assert not options.save_but.isEnabled()

        # Obtain the projection resolution entry
        proj_res = option_entries['proj_res']

        # Remove the value in the proj_res box
        qtbot.keyClick(proj_res.box, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(proj_res.box, QC.Qt.Key_Delete)

        # Set the current projection resolution to 105
        qtbot.keyClicks(proj_res.box, '105')

        # Check that the value has been changed to 105
        assert (get_box_value(proj_res.box) == 105)

    # Test if changes to the options can be properly saved
    def test_save_options(self, qtbot, options, option_entries):
        # Check that the save button is currently enabled
        assert options.save_but.isEnabled()

        # Try to save the options
        qtbot.mouseClick(options.save_but, QC.Qt.LeftButton)

        # Check that the save button is currently disabled
        assert not options.save_but.isEnabled()

        # Check that the new value of proj_res is correct
        proj_res = option_entries['proj_res']
        proj_res_val = get_box_value(proj_res.box)
        assert (proj_res_val == 105)
        assert (proj_res_val != proj_res.default)
        assert (proj_res.value == proj_res_val)

    # Test if an other option can have its value changed (will be reset)
    def test_change_option_value_reset(self, option_entries):
        # Obtain the auto_tile entry
        auto_tile = option_entries['auto_tile']

        # Toggle the value of the auto_tile box
        auto_tile.box.click()

        # Check that the checkbox has been toggled
        assert get_box_value(auto_tile.box) is not auto_tile.value

    # Test if options can be reset to their default values
    def test_reset_options(self, qtbot, options, option_entries):
        # Check that the save button is currently enabled
        assert options.save_but.isEnabled()

        # Reset the options
        qtbot.mouseClick(options.reset_but, QC.Qt.LeftButton)

        # Check that the save button is currently disabled
        assert not options.save_but.isEnabled()

        # Check that the values of all entries are equal to the defaults
        for entry in option_entries.values():
            assert (get_box_value(entry.box) == entry.default)
            assert (entry.value == get_box_value(entry.box))

    # Test if an other option can have its value changed (will be discarded)
    def test_change_option_value_discard(self, qtbot, option_entries):
        # Obtain the align entries
        align_col = option_entries['align_col']
        align_row = option_entries['align_row']

        # Toggle the alignment value
        qtbot.mouseClick(align_col.box if align_row.value else align_row.box,
                         QC.Qt.LeftButton)

        # Check that the alignment was toggled
        assert get_box_value(align_col.box) is not align_col.value
        assert get_box_value(align_row.box) is not align_row.value

    # Test that the options menu can be properly closed
    def test_close(self, qtbot, options):
        # Close the options menu
        qtbot.mouseClick(options.close_but, QC.Qt.LeftButton)

        # Check if the options menu is currently closed
        assert options.isHidden()

    # Test that all box value changes were discarded when the menu closed
    def test_discarded_options(self, option_entries):
        # Test that every box value is equal to its entry value
        for entry in option_entries.values():
            assert (get_box_value(entry.box) == entry.value)
