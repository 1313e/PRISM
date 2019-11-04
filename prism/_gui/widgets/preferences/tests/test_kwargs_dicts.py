# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from PyQt5 import QtCore as QC, QtWidgets as QW
import pytest

# PRISM imports
from prism._gui.widgets.core import get_box_value, set_box_value
from prism._gui.widgets.preferences.custom_boxes import (
    ColorBox, DefaultBox, FigSizeBox)
from prism._gui.widgets.preferences.kwargs_dicts import (
    KwargsDictBoxLayout, KwargsDictDialog, KwargsDictDialogPage)


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% HELPER FUNCTIONS
# Make abbreviation for kwargs_dict dialog
@pytest.fixture(scope='module')
def kwargs_dicts(options):
    return(options.dict_dialog)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the kwargs dict box
@pytest.mark.incremental
class TestKwargsDictBoxLayout(object):
    # Test if it is bound to options
    def test_bound_options(self, options):
        assert hasattr(options, 'kwargs_dict_box')
        assert isinstance(options.kwargs_dict_box, KwargsDictBoxLayout)
        obj = options.kwargs_dict_box
        while obj is not options and obj is not None:
            obj = obj.parentWidget()
        assert obj is not None


# Pytest for the kwargs dict dialog
@pytest.mark.incremental
class TestKwargsDictDialog_Main(object):
    # Test if it is bound to options
    def test_bound_options(self, options, kwargs_dicts):
        assert any([isinstance(child, KwargsDictDialog)
                    for child in options.children()])
        assert isinstance(kwargs_dicts, KwargsDictDialog)

    # Test if the dialog contains the correct number of pages
    def test_n_pages(self, option_entries, kwargs_dicts):
        # Obtain the number of pages there should be
        n_pages = len([name for name in option_entries if 'kwargs' in name])

        # Check that there are this many pages
        assert kwargs_dicts.pages.count() == n_pages
        assert kwargs_dicts.contents.count() == kwargs_dicts.pages.count()

    # Test if the dialog contains the proper pages
    def test_pages(self, kwargs_dicts):
        # Check that all pages are instances of KwargsDictDialogPage
        n_pages = kwargs_dicts.pages.count()
        for page in map(kwargs_dicts.pages.widget, range(n_pages)):
            assert isinstance(page.widget(), KwargsDictDialogPage)

        # Check that a page has the dialog (somewhere) as its parent
        obj = page.widget()
        while obj is not kwargs_dicts and obj is not None:
            obj = obj.parentWidget()
        assert obj is not None

    # Test if all pages have the proper layout
    def test_page_layouts(self, kwargs_dicts):
        # Check that all pages have a grid layout with n_count divisible by 3
        n_pages = kwargs_dicts.pages.count()
        for page in map(kwargs_dicts.pages.widget, range(n_pages)):
            page = page.widget()
            assert hasattr(page, 'kwargs_grid')
            assert isinstance(page.kwargs_grid, QW.QGridLayout)
            assert not page.kwargs_grid.count() % 3

    # Test if the kwargs_dict dialog can be opened
    def test_open(self, qtbot, options, kwargs_dicts):
        # Try to open the kwargs_dicts window
        qtbot.mouseClick(options.kwargs_dict_box.view_but, QC.Qt.LeftButton)

        # Check that currently the kwargs_dict dialog is open
        assert kwargs_dicts.isVisible()

    # Test if a new dict entry can be added
    def test_add_editable_entry(self, qtbot, option_entries, kwargs_dicts):
        # Obtain the fig_kwargs page
        page = option_entries['fig_kwargs'].box

        # Check that this page currently contains at least 1 row
        n_rows = page.kwargs_grid.count()//3
        assert n_rows

        # Click on the 'add' button of this page
        qtbot.mouseClick(page.add_but, QC.Qt.LeftButton)

        # Check that an extra row has been added
        assert (page.kwargs_grid.count()//3 == n_rows+1)
        n_rows += 1

        # Obtain the kwargs box at the last row and validate it
        kwargs_box = page.kwargs_grid.itemAtPosition(n_rows-1, 1).widget()
        assert isinstance(kwargs_box, QW.QComboBox)

        # Check that this kwargs_box currently has nothing selected
        assert (get_box_value(kwargs_box) == '')

        # Add three more entries
        page.add_but.click()
        page.add_but.click()
        page.add_but.click()

        # Remove the second-last entry again
        row = page.kwargs_grid.count()//3-2
        del_but = page.kwargs_grid.itemAtPosition(row, 0).widget()
        del_but.click()

    # Test if this new entry can have its field set
    def test_set_entry_default(self, qtbot, option_entries, kwargs_dicts):
        # Obtain the fig_kwargs page and the index of the second-last row
        page = option_entries['fig_kwargs'].box
        row = page.kwargs_grid.count()//3-3

        # Obtain the kwargs_box
        kwargs_box = page.kwargs_grid.itemAtPosition(row, 1).widget()

        # Make sure that 'a_test' is not a default entry type
        assert 'a_test' not in page.std_entries

        # Set the value of this kwargs_box
        qtbot.keyClicks(kwargs_box, 'a_test')

        # Check that the current value of the kwargs_box is 'a_test'
        assert (get_box_value(kwargs_box) == 'a_test')

        # Check that the field_box is an instance of the DefaultBox
        field_box = page.kwargs_grid.itemAtPosition(row, 2).widget()
        assert isinstance(field_box, DefaultBox)

        # Set the field_box to bool and False
        set_box_value(field_box.type_box, 'bool')
        set_box_value(field_box.value_box, False)

    # Test if another new entry can have its field set the same way
    def test_set_entry_duplicate(self, qtbot, option_entries, kwargs_dicts):
        # Obtain the fig_kwargs page and the index of the last row
        page = option_entries['fig_kwargs'].box
        row = page.kwargs_grid.count()//3-2

        # Obtain the kwargs_box
        kwargs_box = page.kwargs_grid.itemAtPosition(row, 1).widget()

        # Set the value of this kwargs_box
        qtbot.keyClicks(kwargs_box, 'a_test')

        # Set the field box to a float of 1.5
        field_box = page.kwargs_grid.itemAtPosition(row, 2).widget()
        set_box_value(field_box.type_box, 'float')
        set_box_value(field_box.value_box, 1.5)

    # Test if saving the kwargs_dicts works
    def test_save_kwargs_dicts(self, options, option_entries):
        # Try to save the options
        options.save_but.click()

        # Check that the associated fig_kwargs entry has been updated
        entry = option_entries['fig_kwargs']
        fig_kwargs = get_box_value(entry.box)
        assert (fig_kwargs['a_test'] == 1.5)
        assert (fig_kwargs != entry.default)
        assert (entry.value == fig_kwargs)

    # Test if the kwargs_dict dialog closes when the options menu closes
    def test_close_preferences_window(self, options, kwargs_dicts):
        # Check that the window is currently open
        assert kwargs_dicts.isVisible()

        # Close the options menu
        options.close_but.click()

        # Check that both windows are now closed
        assert not options.isVisible()
        assert not kwargs_dicts.isVisible()

        # Open the kwargs_dicts window again
        options.kwargs_dict_box.view_but.click()

    # Test what happens if an empty kwargs_box is used
    def test_set_entry_empty(self, option_entries, kwargs_dicts):
        # Obtain the fig_kwargs page
        page = option_entries['fig_kwargs'].box

        # Check which row should have 'a_test'
        row = option_entries['fig_kwargs'].value.keys().index('a_test')+1

        # Check that this row indeed has 'a_test'
        kwargs_box = page.kwargs_grid.itemAtPosition(row, 1).widget()
        assert (get_box_value(kwargs_box) == 'a_test')

        # Set the kwargs_box to empty
        set_box_value(kwargs_box, '')

        # Check that the associated field_box is now an empty label
        field_box = page.kwargs_grid.itemAtPosition(row, 2).widget()
        assert isinstance(field_box, QW.QLabel)
        assert (field_box.text() == '')

    # Test what happens if a banned entry_type is used
    def test_set_entry_banned(self, option_entries, kwargs_dicts):
        # Obtain the impl_kwargs_2D page
        page = option_entries['impl_kwargs_2D'].box

        # Add a new entry to this page
        page.add_but.click()

        # Obtain a banned entry
        assert len(page.banned_entries)
        entry_type = page.banned_entries[0]

        # Set the corresponding kwargs_box to this entry_type
        row = page.kwargs_grid.count()//3-1
        kwargs_box = page.kwargs_grid.itemAtPosition(row, 1).widget()
        set_box_value(kwargs_box, entry_type)

        # Check that the field_box is now a non-empty label
        field_box = page.kwargs_grid.itemAtPosition(row, 2).widget()
        assert isinstance(field_box, QW.QLabel)
        assert (field_box.text() != '')

    # Test if saving the kwargs_dicts still works
    def test_save_kwargs_dicts2(self, options, option_entries):
        # Try to save the options
        options.save_but.click()

        # Check that the impl_kwargs_2D entry has not changed
        entry = option_entries['impl_kwargs_2D']
        impl_kwargs_2D = get_box_value(entry.box)
        assert (impl_kwargs_2D == entry.default)
        assert (entry.value == impl_kwargs_2D)


# Pytest for the kwargs dict dialog entry types
class TestKwargsDictDialog_EntryTypes(object):
    # Test if all standard entry types can be properly used
    @pytest.mark.parametrize(
        "page_name, entry_type, field_type, field_value",
        [('impl_kwargs_2D', 'alpha', QW.QDoubleSpinBox, 0.5),
         ('impl_kwargs_3D', 'cmap', QW.QComboBox, 'rainforest'),
         ('los_kwargs_2D', 'color', ColorBox, 'cyan'),
         ('fig_kwargs', 'dpi', QW.QSpinBox, 175),
         ('fig_kwargs', 'figsize', FigSizeBox, (13, 13)),
         ('line_kwargs_est', 'linestyle', QW.QComboBox, '--'),
         ('line_kwargs_cut', 'linewidth', QW.QDoubleSpinBox, 6.9),
         ('impl_kwargs_2D', 'marker', QW.QComboBox, '*'),
         ('los_kwargs_2D', 'markersize', QW.QDoubleSpinBox, 42),
         ('los_kwargs_3D', 'xscale', QW.QComboBox, 'linear'),
         ('impl_kwargs_3D', 'yscale', QW.QComboBox, 'log')])
    def test_set_standard_entry(self, page_name, entry_type, field_type,
                                field_value, option_entries, kwargs_dicts):
        # Obtain the proper page
        page = option_entries[page_name].box

        # Add a new entry to this page
        page.add_but.click()

        # Set the kwargs_box to entry_type
        row = page.kwargs_grid.count()//3-1
        kwargs_box = page.kwargs_grid.itemAtPosition(row, 1).widget()
        set_box_value(kwargs_box, entry_type)

        # Check that the field box is an instance of given field_type
        field_box = page.kwargs_grid.itemAtPosition(row, 2).widget()
        assert isinstance(field_box, field_type)

        # Set the value of this box
        set_box_value(field_box, field_value)

    # Test if an error message is given if a bad colormap is chosen
    def test_set_bad_cmap(self, monkeypatch, option_entries, kwargs_dicts):
        # Obtain the los_kwargs_3D page
        page = option_entries['los_kwargs_3D'].box

        # Add a new entry to this page
        page.add_but.click()

        # Set the kwargs_box to 'cmap'
        row = page.kwargs_grid.count()//3-1
        kwargs_box = page.kwargs_grid.itemAtPosition(row, 1).widget()
        set_box_value(kwargs_box, 'cmap')

        # Monkey patch the QMessagebox.warning function
        monkeypatch.setattr(QW.QMessageBox, 'warning',
                            lambda *args: QW.QMessageBox.Ok)

        # Set the value of the field_box
        field_box = page.kwargs_grid.itemAtPosition(row, 2).widget()
        set_box_value(field_box, 'jet')
