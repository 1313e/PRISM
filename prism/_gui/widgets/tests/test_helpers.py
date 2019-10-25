# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path
import sys

# Package imports
from PyQt5 import QtCore as QC

# PRISM imports
from prism._gui.widgets.helpers import ExceptionDialog, show_exception_details


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for the ExceptionDialog
class TestExceptionDialog(object):
    # Test if the main window is currently using a custom excepthook
    def test_custom_excepthook(self, main_window):
        assert sys.excepthook.func is show_exception_details

    # Test if error information can be passed to an ExceptionDialog
    # Due to qapp.exec_ not running, it is impossible to raise and catch an
    # error successfully from within main_window
    def test_raise_error(self, monkeypatch, main_window):
        # Monkey patch the ExceptionDialog.show function
        monkeypatch.setattr(ExceptionDialog, 'show', lambda *args: None)

        # Raise an error
        try:
            raise Exception
        except Exception:
            show_exception_details(main_window, *sys.exc_info())

    # Test if the traceback of the error can be shown
    def test_show_traceback(self, qtbot):
        # Raise an error (again)
        try:
            raise Exception
        except Exception:
            # Create the ExceptionDialog
            exception_dialog = ExceptionDialog(None, *sys.exc_info())
            qtbot.addWidget(exception_dialog)

            # Show the exception dialog
            exception_dialog.show()

            # Try to show the traceback box
            qtbot.mouseClick(exception_dialog.tb_but, QC.Qt.LeftButton)

            # Check that it is currently being shown
            assert exception_dialog.tb_box.isVisible()
