# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from PyQt5 import QtCore as QC
import pytest

# PRISM imports
from prism._gui.widgets.preferences.kwargs_dicts import (
    KwargsDictBoxLayout, KwargsDictDialog, KwargsDictDialogPage)
from prism._gui.widgets.preferences.helpers import get_box_value


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
class TestKwargsDictDialog(object):
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
    def test_pages(self, option_entries, kwargs_dicts):
        # Check that all pages are instances of KwargsDictDialogPage
        n_pages = kwargs_dicts.pages.count()
        for page in map(kwargs_dicts.pages.widget, range(n_pages)):
            assert isinstance(page.widget(), KwargsDictDialogPage)

        # Check that a page has the dialog (somewhere) as its parent
        obj = page.widget()
        while obj is not kwargs_dicts and obj is not None:
            obj = obj.parentWidget()
        assert obj is not None

    # Test if the kwargs_dict dialog can be opened
    def test_open(self, qtbot, options, kwargs_dicts):
        # Try to open the kwargs_dicts window
        qtbot.mouseClick(options.kwargs_dict_box.view_but, QC.Qt.LeftButton)

        # Check that currently the kwargs_dict dialog is open
        assert kwargs_dicts.isVisible()
