# -*- coding: utf-8 -*-

"""
GUI Widget Helpers
==================
Provides a collection of custom :class:`~PyQt5.QtWidgets.QWidget` subclasses
that provide specific functionalities.

"""


# %% IMPORTS
# Built-in imports
import sys
import threading
from time import sleep
from traceback import format_exception_only, format_tb

# Package imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore as QC, QtWidgets as QW

# PRISM imports
from prism._gui import APP_NAME

# All declaration
__all__ = ['ExceptionDialog', 'FigureCanvas', 'OverviewListWidget',
           'ThreadedProgressDialog', 'show_exception_details']


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
        grid_layout.setRowStretch(3, 1)

        # Set properties of message box
        self.setWindowModality(QC.Qt.ApplicationModal)
        self.setAttribute(QC.Qt.WA_DeleteOnClose)
        self.setWindowTitle("ERROR")
        self.setWindowFlags(
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
        tb_text_box.setMinimumHeight(100)
        tb_text_box.setFocusPolicy(QC.Qt.NoFocus)
        tb_text_box.setReadOnly(True)
        tb_text_box.setText(tb_str)
        layout.addWidget(tb_text_box)

        # Create a 'show traceback' button
        self.tb_labels = ['Hide Traceback...', 'Show Traceback...']

        # Return traceback box
        return(traceback_box)

    # This function shows or hides the traceback box
    @QC.pyqtSlot()
    def toggle_traceback_box(self):
        # Toggle the visibility of the traceback box
        self.tb_box.setHidden(not self.tb_box.isHidden())
        self.tb_but.setText(self.tb_labels[self.tb_box.isHidden()])

        # Update the size of the message box
        self.update_size()

    # This function updates the size of the dialog
    def update_size(self):
        # Determine the minimum/maximum size required for making the dialog
        min_size = self.layout().minimumSize()
        max_size = self.layout().maximumSize()

        # Set the fixed width
        self.setFixedWidth(min_size.width())

        # If the traceback box is shown, set minimum/maximum height
        if self.tb_box.isVisible():
            self.setMinimumHeight(min_size.height())
            self.setMaximumHeight(max_size.height())
        # Else, set fixed height
        else:
            self.setFixedHeight(min_size.height())


# Class used for holding the projection figures in the projection viewing area
class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, figure, *args, **kwargs):
        # Call the super constructor
        super().__init__(figure)

        # Create the figure canvas
        self.init()

    # Create the figure canvas
    def init(self):
        pass

    # Override the resizeEvent to automatically properly resize figure
    def resizeEvent(self, *args, **kwargs):
        # Call super event
        super().resizeEvent(*args, **kwargs)

        # Recalculate the constrained layout of the figure
#        self.figure.execute_constrained_layout()


# Class used for making the overview lists in the GUI
class OverviewListWidget(QW.QListWidget):
    def __init__(self, *, hcubes_list, status_tip, context_menu, activated):
        # Call the super constructor
        super().__init__()

        # Create the overview list
        self.init(hcubes_list, status_tip, context_menu, activated)

    # Create the overview list widget
    def init(self, hcubes_list, status_tip, context_menu, activated):
        # Add the items to the list
        self.addItems(hcubes_list)
        self.setStatusTip(status_tip)

        # Set some properties
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(self.ExtendedSelection)
        self.setContextMenuPolicy(QC.Qt.CustomContextMenu)

        # Set signal handling
        self.customContextMenuRequested.connect(context_menu)
        self.itemActivated.connect(activated)

        # Make sure the items in the list are sorted
        self.sortItems()

    # Override keyPressEvent
    def keyPressEvent(self, event):
        # Check if the event involved pressing Enter or Return
        if event.key() in (QC.Qt.Key_Enter, QC.Qt.Key_Return):
            # If so, emit the itemActivated signal
            self.itemActivated.emit(self.currentItem())
        # Else, check if the event involved pressing CTRL + A
        elif(event.key() == QC.Qt.CTRL + QC.Qt.Key_A):
            # If so, select all items in the list
            self.selectAll()
        # Else, handle as normal
        else:
            super().keyPressEvent(event)


# Class that provides a special threaded progress dialog
class ThreadedProgressDialog(QW.QProgressDialog):
    # Make a signal that is emitted whenever the progress dialog finishes
    finished = QC.pyqtSignal()

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
        self.setAutoReset(False)

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
        self.thread.finished.connect(self.set_successful_finish)
        self.thread.exception.connect(self.raise_exception)
        super().open(self.kill_threads)

        # Save that progress dialog has currently not finished successfully
        self.successful = False

        # Start the threads for all other MPI ranks
        self.pipe._make_call_workers(_run_traced_worker_threads, 'pipe')

        # Determine what the current QApplication instance is
        qapp = QW.QApplication.instance()

        # Start the thread
        self.thread.start()

        # While the thread is running, keep processing user input events
        while self.thread.isAlive():
            qapp.processEvents()
            self.thread.join(0.1)

        # Process user input events one last time
        qapp.processEvents()

        # If the dialog was not canceled, kill all the threads
        if not self.wasCanceled():
            self.kill_threads()
            self.reset()

        # Emit that the progress dialog has finished
        self.finished.emit()

        # Return if dialog finished successfully or not
        return(self.successful)

    # This function sets an attribute and serves as a slot
    @QC.pyqtSlot()
    def set_successful_finish(self):
        self.successful = True

    # This function raises an exception caught in the controller thread
    @QC.pyqtSlot(Exception)
    def raise_exception(self, exception):
        raise exception

    # This function kills all worker threads and then the controller thread
    @QC.pyqtSlot()
    def kill_threads(self):
        # Let the secondary worker threads wait for a second
        self.pipe._make_call_workers(sleep, 1)

        # Set all worker threads to 'killed'
        for rank in range(1, self.pipe._comm.size):
            self.pipe._comm.send(True, rank, 671589+rank)
        self.thread.killed = True
        self.thread.join()

        # Use an MPI Barrier to make sure that all threads were killed
        # This means that the controller also has to wait for a second
        self.pipe._make_call('_comm.Barrier')


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

    # Define a signal that is emitted when the thread finishes executing
    finished = QC.pyqtSignal()

    # Define a signal that is emitted whenever an exception occurs
    exception = QC.pyqtSignal(Exception)

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
        try:
            for i, _ in enumerate(self.run_map):
                self.n_finished.emit(i+1)
        except Exception as error:
            self.exception.emit(error)
        # Emit signal that execution has finished
        else:
            self.finished.emit()


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
        worker_mode = self.pipe.worker_mode
        worker_mode._WorkerMode__key = -1
        worker_mode.listen_for_calls()


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
    thread.killed = pipe._comm.recv(None, 0, 671589+pipe._comm.rank)

    # Connect to the thread to make sure it ended properly
    thread.join()


# This function creates a message box with exception information
def show_exception_details(parent, etype, value, tb):
    # Create exception message box
    exception_box = ExceptionDialog(parent, etype, value, tb)

    # Show the exception message box
    exception_box.show()
