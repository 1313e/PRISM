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
import threading
from time import sleep
from traceback import format_exception_only, format_tb

# Package imports
from PyQt5 import QtCore as QC, QtWidgets as QW

# PRISM imports
from prism._gui import APP_NAME

# All declaration
__all__ = ['ExceptionDialog', 'QW_QAction', 'QW_QComboBox',
           'QW_QDoubleSpinBox', 'QW_QEditableComboBox', 'QW_QLabel',
           'QW_QMenu', 'QW_QSpinBox', 'ThreadedProgressDialog',
           'show_exception_details']


# %% CLASS DEFINITIONS
# Make special class for showing exception details
class ExceptionDialog(QW.QDialog):
    def __init__(self, parent, etype, value, tb):
        # Save the provided values
        self.etype = etype
        self.value = value
        self.tb = tb

        # Call the super constructor
        super().__init__(parent)

        # Initialize the exception dialog
        self.init()

    # This function creates the exception dialog
    def init(self):
        # Create a window layout
        grid_layout = QW.QGridLayout(self)
        grid_layout.setColumnStretch(2, 1)

        # Set properties of message box
        self.setWindowModality(QC.Qt.ApplicationModal)
        self.setAttribute(QC.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ERROR")
        self.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        self.setWindowFlags(
            QC.Qt.MSWindowsFixedSizeDialogHint |
            QC.Qt.MSWindowsOwnDC |
            QC.Qt.Dialog |
            QC.Qt.WindowTitleHint |
            QC.Qt.WindowSystemMenuHint |
            QC.Qt.WindowCloseButtonHint)

        # Set the icon of the exception on the left
        icon_label = QW.QLabel()
        pixmap = QW.QMessageBox.standardIcon(QW.QMessageBox.Critical)
        icon_label.setPixmap(pixmap)
        grid_layout.addWidget(icon_label, 0, 0, 2, 1, QC.Qt.AlignTop)

        # Add a spacer item
        spacer_item = QW.QSpacerItem(7, 1, QW.QSizePolicy.Fixed,
                                     QW.QSizePolicy.Fixed)
        grid_layout.addItem(spacer_item, 0, 1, 2, 1)

        # Set the text of the exception
        exc_str = self.format_exception()
        exc_label = QW.QLabel(exc_str)
        grid_layout.addWidget(exc_label, 0, 2, 1, 1)

        # Create a button box for the buttons
        button_box = QW.QDialogButtonBox()
        grid_layout.addWidget(button_box, 2, 0, 1, grid_layout.columnCount())

        # Create traceback box
        self.tb_box = self.create_traceback_box()
        grid_layout.addWidget(self.tb_box, 3, 0, 1, grid_layout.columnCount())

        # Create traceback button
        self.tb_but =\
            button_box.addButton(self.tb_labels[self.tb_box.isHidden()],
                                 button_box.ActionRole)
        self.tb_but.clicked.connect(self.toggle_traceback_box)

        # Create an 'ok' button
        ok_but = button_box.addButton(button_box.Ok)
        ok_but.clicked.connect(self.close)
        ok_but.setDefault(True)

        # Update the size
        self.update_size()

    # This function formats the exception string
    def format_exception(self):
        # Format the exception
        exc_list = format_exception_only(self.etype, self.value)
        exc_str = ''.join(exc_list)

        # Return it
        return(exc_str)

    # This function formats the traceback string
    def format_traceback(self):
        # Format the traceback
        tb_list = format_tb(self.tb)
        tb_str = ''.join(tb_list)

        # Return it
        return(tb_str)

    # This function creates the traceback box
    def create_traceback_box(self):
        # Create a traceback box
        traceback_box = QW.QWidget(self)
        traceback_box.setHidden(True)

        # Create layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(QC.QMargins())
        traceback_box.setLayout(layout)

        # Add a horizontal line to the layout
        frame = QW.QFrame(traceback_box)
        frame.setFrameShape(frame.HLine)
        frame.setFrameShadow(frame.Sunken)
        layout.addWidget(frame)

        # Format the traceback
        tb_str = self.format_traceback()

        # Add a textedit to the layout
        tb_text_box = QW.QTextEdit(traceback_box)
        tb_text_box.setFixedHeight(100)
        tb_text_box.setFocusPolicy(QC.Qt.NoFocus)
        tb_text_box.setReadOnly(True)
        tb_text_box.setText(tb_str)
        layout.addWidget(tb_text_box)

        # Create a 'show traceback' button
        self.tb_labels = ['Hide Traceback...', 'Show Traceback...']

        # Return traceback box
        return(traceback_box)

    # This function shows or hides the traceback box
    def toggle_traceback_box(self):
        # Toggle the visibility of the traceback box
        self.tb_box.setHidden(not self.tb_box.isHidden())
        self.tb_but.setText(self.tb_labels[self.tb_box.isHidden()])

        # Update the size of the message box
        self.update_size()

    # This function updates the size of the dialog
    def update_size(self):
        # Determine the minimum size required for making the dialog
        self.setFixedSize(self.layout().minimumSize())


# Make subclass of QW.QAction that automatically sets details based on status
class QW_QAction(QW.QAction):
    # Override constructor
    def __init__(self, parent, text, *, shortcut=None, tooltip=None,
                 statustip=None, icon=None, triggered=None):
        # Call super constructor
        if icon is None:
            super().__init__(text, parent)
        else:
            super().__init__(icon, text, parent)

        # Set all the details
        self.setDetails(shortcut=shortcut,
                        tooltip=tooltip,
                        statustip=statustip)

        # Set the signal trigger
        if triggered is not None:
            self.triggered.connect(triggered)

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


# Create custom combobox class with more signals
class QW_QComboBox(QW.QComboBox):
    popup_shown = QC.pyqtSignal([int], [str])
    popup_hidden = QC.pyqtSignal([int], [str])

    # Override the showPopup to emit a signal whenever it is triggered
    def showPopup(self, *args, **kwargs):
        self.popup_shown[int].emit(self.currentIndex())
        self.popup_shown[str].emit(self.currentText())
        return(super().showPopup(*args, **kwargs))

    # Override the hidePopup to emit a signal whenever it is triggered.
    def hidePopup(self, *args, **kwargs):
        self.popup_hidden[int].emit(self.currentIndex())
        self.popup_hidden[str].emit(self.currentText())
        return(super().hidePopup(*args, **kwargs))


# Create custom QComboBox class that is editable
class QW_QEditableComboBox(QW_QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEditable(True)
        self.setInsertPolicy(self.NoInsert)
        self.completer().setCompletionMode(QW.QCompleter.PopupCompletion)
        self.completer().setFilterMode(QC.Qt.MatchContains)


# Create custom QAbstractSpinBox that automatically sets some properties
class QW_QAbstractSpinBox(QW.QAbstractSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStepType(self.AdaptiveDecimalStepType)
        self.setAccelerated(True)
        self.setGroupSeparatorShown(True)


# Create custom QDoubleSpinBox
class QW_QDoubleSpinBox(QW.QDoubleSpinBox, QW_QAbstractSpinBox):
    pass


# Create custom QSpinBox
class QW_QSpinBox(QW.QSpinBox, QW_QAbstractSpinBox):
    pass


# Create custom label class with more signals
class QW_QLabel(QW.QLabel):
    mousePressed = QC.pyqtSignal()

    # Override the mousePressEvent to emit a signal whenever it is triggered
    def mousePressEvent(self, event):
        self.mousePressed.emit()
        event.accept()


# Create custom QMenu class that swaps the order of inputs
class QW_QMenu(QW.QMenu):
    def __init__(self, parent, text):
        super().__init__(text, parent)


# Class that provides a special threaded progress dialog
class ThreadedProgressDialog(QW.QProgressDialog):
    def __init__(self, main_window_obj, label, cancel, func, *iterables):
        # Save provided MainWindow obj
        self.main = main_window_obj
        self.pipe = self.main.pipe

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
        # Initialize the traced thread
        self.thread = TracedControllerThread(self.run_map, self)

        # Connect the proper signals with each other
        self.thread.n_finished.connect(self.setValue)
        super().open(self.kill_threads)

        # Start the threads for all other MPI ranks
        self.pipe._make_call_workers(_run_traced_worker_threads, 'pipe')

        # Start the thread
        self.thread.start()

        # While the thread is running, keep processing user input events
        while self.thread.isAlive():
            self.main.qapp.processEvents()
            self.thread.join(0.1)

        # If the dialog ended successfully, end all the threads
        if not self.wasCanceled():
            self.end_threads()
            return(True)
        else:
            return(False)

    # This function finalizes all worker threads and then the controller thread
    def end_threads(self):
        # Let the secondary worker threads wait for a second
        self.pipe._make_call_workers(sleep, 1)

        # Kill all threads in the mean time
        self.kill_threads()

    # This function kills all worker threads and then the controller thread
    def kill_threads(self):
        for rank in range(1, self.pipe._comm.size):
            self.pipe._comm.send(True, rank, 999+rank)
        self.thread.killed = True
        self.thread.join()


# Special system traced thread that stops whenever killed is set to True
class TracedThread(threading.Thread):
    def __init__(self):
        # Set killed to False
        self.killed = False

        # Call super constructor
        super().__init__(None)

    # Make a custom system tracer
    def global_trace(self, frame, event, arg):
        # Implement default global system tracer behavior
        if(event == 'call'):
            return(self.local_trace)

    # This function implements the local part of the system tracer
    def local_trace(self, frame, event, arg):
        # If this thread must be killed, raise an error at the next line
        if self.killed and (event == 'line'):
            raise SystemExit

        # Return self
        return(self.local_trace)


# https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
# System traced controller thread that loops over a provided map iterator
class TracedControllerThread(QC.QObject, TracedThread):
    # Define a signal that sends out the number of finished iterations
    n_finished = QC.pyqtSignal('int')

    # Define a signal that is emitted whenever an exception occurs
    exception = QC.pyqtSignal()

    def __init__(self, run_map, parent):
        # Save provided map iterator
        self.run_map = run_map

        # Call the super constructors
        super().__init__(parent)
        TracedThread.__init__(self)

    # This function gets called when TracedThread.start() is called
    def run(self):
        # Set the system tracer
        sys.settrace(self.global_trace)

        # Emit that currently the number of finished iteration is 0
        self.n_finished.emit(0)

        # Loop over the map iterator and send a signal after each iteration
        for i, _ in enumerate(self.run_map):
            self.n_finished.emit(i+1)


# https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
# Special system traced worker thread that connects to the controller thread
class TracedWorkerThread(TracedThread):
    def __init__(self, pipeline_obj):
        # Save provided pipeline_obj
        self.pipe = pipeline_obj

        # Call the super constructor
        super().__init__()

    # This function gets called when TracedThread.start() is called
    def run(self):
        # Set the system tracer
        sys.settrace(self.global_trace)

        # Start listening for calls on this thread as well
        self.pipe._listen_for_calls()


# %% FUNCTION DEFINITIONS
# This function starts up the threads for all workers
def _run_traced_worker_threads(pipeline_obj):
    # Abbreviate pipeline_obj
    pipe = pipeline_obj

    # Initialize a worker thread
    thread = TracedWorkerThread(pipe)

    # Start executing on this thread
    thread.start()

    # Keep listening for the controller telling to stop the worker thread
    thread.killed = pipe._comm.recv(None, 0, 999+pipe._comm.rank)

    # Connect to the thread to make sure it ended properly
    thread.join()


# This function creates a message box with exception information
def show_exception_details(parent, etype, value, tb):
    # Emit the exception signal
    parent.exception.emit()

    # Create exception message box
    exception_box = ExceptionDialog(parent, etype, value, tb)

    # Show the exception message box
    exception_box.show()
