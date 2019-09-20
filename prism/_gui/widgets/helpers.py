# -*- coding: utf-8 -*-

"""
GUI Widget Helpers
==================
Provides a collection of custom :class:`~PyQt5.QtWidgets.QWidget` subclasses
that allow for certain layouts to be standardized.

"""


# %% IMPORTS
# Built-in imports
import sys
import traceback
from traceback import format_exception_only, format_tb

# Package imports
from PyQt5 import QtCore as QC, QtWidgets as QW

# PRISM imports
from prism._gui import APP_NAME

# All declaration
__all__ = ['QW_QAction', 'ThreadedProgressDialog', 'WorkerThread',
           'show_exception_details']


# %% CLASS DEFINITIONS
# Make subclass of QW.QAction that automatically sets details based on status
class QW_QAction(QW.QAction):
    # Make new method that automatically sets Shortcut, ToolTip and StatusTip
    def setDetails(self, *, shortcut=None, tooltip=None, statustip=None):
        # If shortcut is not None, set it
        if shortcut is not None:
            super().setShortcut(shortcut)
            shortcut = self.shortcut().toString()

        # If tooltip is None, its base is set to the action's name
        if tooltip is None:
            base_tooltip = self.text().replace('&', '')
            tooltip = base_tooltip
        # Else, provided tooltip is used as the base
        else:
            base_tooltip = tooltip

        # If shortcut is not None, add it to the tooltip
        if shortcut is not None:
            tooltip = "%s (%s)" % (base_tooltip, shortcut)

        # Set tooltip
        super().setToolTip(tooltip)

        # If statustip is None, it is set to base_tooltip
        if statustip is None:
            statustip = base_tooltip

        # Set statustip
        super().setStatusTip(statustip)

    # Override setShortcut to raise an error when used
    def setShortcut(self, *args, **kwargs):  # pragma: no cover
        raise AttributeError("Using this method is not allowed! Use "
                             "'setDetails()' instead!")

    # Override setToolTip to raise an error when used
    def setToolTip(self, *args, **kwargs):  # pragma: no cover
        raise AttributeError("Using this method is not allowed! Use "
                             "'setDetails()' instead!")

    # Override setStatusTip to raise an error when used
    def setStatusTip(self, *args, **kwargs):  # pragma: no cover
        raise AttributeError("Using this method is not allowed! Use "
                             "'setDetails()' instead!")


# Class that provides a special QThreaded progress dialog
# FIXME: This dialog does not interrupt properly on Ubuntu in some cases
class ThreadedProgressDialog(QW.QProgressDialog):
    def __init__(self, main_window_obj, label, cancel, func, *iterables):
        # Save provided MainWindow obj
        self.main = main_window_obj

        # Call the super constructor
        super().__init__(self.main)

        # Create the progress dialog
        self.init(label, cancel, func, *iterables)

    # Create the threaded progress dialog
    def init(self, label, cancel, func, *iterables):
        # Set the label and cancel button
        self.setLabelText(label)
        self.setCancelButtonText(cancel)

        # Determine the minimum length of iterables
        min_len = min([len(iterable) for iterable in iterables])

        # Set the range of this progress dialog
        self.setRange(0, min_len)

        # Make this progress dialog application modal
        self.setWindowModality(QC.Qt.ApplicationModal)
        self.setWindowTitle(APP_NAME)
        self.setAttribute(QC.Qt.WA_DeleteOnClose)

        # Setup the run_map that will be used
        self.run_map = map(func, *iterables)

    # This function simply calls open()
    def __call__(self):
        return(self.open())

    # This function executes the entire run_map until finished or aborted
    def open(self):
        # Initialize the worker thread
        self.worker_thread = WorkerThread(self.run_map, self)

        # Connect the proper signals with each other
        self.worker_thread.n_finished.connect(self.setValue)
        super().open(self.worker_thread.terminate)

        # Start the worker thread
        self.worker_thread.start()

        # While the worker thread is running, keep processing user input events
        while self.worker_thread.isRunning():
            self.main.qapp.processEvents()
            self.worker_thread.wait(1)

        # Return whether the dialog ended normally
        return(not self.wasCanceled())


# Basic worker thread that loops over a provided map iterator
class WorkerThread(QC.QThread):
    # Define a signal that sends out the number of finished iterations
    n_finished = QC.pyqtSignal('int')

    def __init__(self, run_map, *args, **kwargs):
        # Save provided map iterator
        self.run_map = run_map

        # Call the super constructor
        super().__init__(*args, **kwargs)

    # This function gets called when WorkerThread.start() is called
    def run(self):
        # Emit that currently the number of finished iteration is 0
        self.n_finished.emit(0)

        # Loop over the map iterator and send a signal after each iteration
        for i, _ in enumerate(self.run_map):
            self.n_finished.emit(i+1)

    # This function is called when the thread must be terminated
    def terminate(self):
        # First try to quit it the normal way
        self.quit()

        # If this does not work, wait for 2 seconds and terminate it
        self.wait(2)
        super().terminate()


# %% FUNCTION DEFINITIONS
# This function creates a message box with exception information
def show_exception_details(parent, etype, value, tb):
    # Format the exception
    exc_list = format_exception_only(etype, value)
    exc_str = ''.join(exc_list)

    # Format the traceback
    tb_list = format_tb(tb)
    tb_str = ''.join(tb_list)

    # Create an exception message box
    exception_box = QW.QMessageBox(parent)
    exception_box.setIcon(QW.QMessageBox.Critical)
    exception_box.setWindowTitle("ERROR")

    # Set the text of the exception
    exception_box.setText(exc_str)

    # Set the traceback text as detailed text
    exception_box.setDetailedText(tb_str)
    print(exception_box.buttons()[0].text())

    # Show the exception message box
    exception_box.show()
