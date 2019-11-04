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
from e13tools.utils import docstring_substitute
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW

# PRISM imports
from prism._docstrings import qt_slot_doc
from prism._gui import APP_NAME

# All declaration
__all__ = ['ExceptionDialog', 'FigureCanvas', 'OverviewListWidget',
           'ThreadedProgressDialog', 'show_exception_details']


# %% CLASS DEFINITIONS
# Make special class for showing exception details
class ExceptionDialog(QW.QDialog):
    """
    Defines the :class:`~ExceptionDialog` class for the Projection GUI.

    This class takes a set of exception details and converts it into a format
    that can be shown using a dialog.

    """

    def __init__(self, parent, etype, value, tb):
        """
        Initialize an instance of the :class:`~ExceptionDialog` class.

        Parameters
        ----------
        parent : :obj:`~PyQt5.QtWidgets.QWidget` object or None
            The parent widget for this dialog or *None* for no parent.
        etype : :class:`~Exception` class
            The :class:`~Exception` class that is associated with this error.
        value : :obj:`~Exception` object
            The :class:`~Exception` instance that is associated with this
            error.
        tb : traceback object
            The corresponding traceback object.

        """

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
        """
        Sets up the exception dialog after it has been initialized.

        This function is mainly responsible for gathering all required
        information; formatting it; and drawing the dialog.

        """

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
        """
        Formats the exception provided during initialization and returns it.

        """

        # Format the exception
        exc_list = format_exception_only(self.etype, self.value)
        exc_str = ''.join(exc_list)

        # Return it
        return(exc_str)

    # This function formats the traceback string
    def format_traceback(self):
        """
        Formats the traceback provided during initialization and returns it.

        """

        # Format the traceback
        tb_list = format_tb(self.tb)
        tb_str = ''.join(tb_list)

        # Return it
        return(tb_str)

    # This function creates the traceback box
    def create_traceback_box(self):
        """
        Creates a special box for the exception dialog that contains the
        traceback information and returns it.

        """

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
    @docstring_substitute(qt_slot=qt_slot_doc)
    def toggle_traceback_box(self):
        """
        Toggles the visibility of the traceback box and updates the dimensions
        of the exception dialog accordingly.

        %(qt_slot)s

        """

        # Toggle the visibility of the traceback box
        self.tb_box.setHidden(not self.tb_box.isHidden())
        self.tb_but.setText(self.tb_labels[self.tb_box.isHidden()])

        # Update the size of the message box
        self.update_size()

    # This function updates the size of the dialog
    def update_size(self):
        """
        Updates the dimensions of the exception dialog depending on its current
        state (traceback box visibility).

        """

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
    pass


# Class used for making the overview lists in the GUI
class OverviewListWidget(QW.QListWidget):
    """
    Defines the :class:`~OverviewListWidget` class.

    This class defines the overview lists that are used by the
    :class:`~prism._gui.widgets.OverviewDockWidget` class.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an instance of the :class:`~OverviewListWidget` class.

        Parameters
        ----------
        args : positional arguments
            The positional arguments that need to be passed to :meth:`~init`.
        kwargs : keyword arguments
            The keyword arguments that need to be passed to :meth:`~init`.

        """

        # Call the super constructor
        super().__init__()

        # Create the overview list
        self.init(*args, **kwargs)

    # Create the overview list widget
    def init(self, *, hcubes_list, status_tip, context_menu, activated):
        """
        Sets up the overview list after it has been initialized.

        This function is mainly responsible for creating the list; adding the
        items to it; and setting some properties.

        Parameters
        ----------
        hcubes_list : list of str
            List of projection hypercube names that must be used to initialize
            this overview list with.
        statustip : str
            The statustip that will be displayed in the statusbar whenever this
            overview list is hovered.
        context_menu : function
            The function that must be called whenever the context menu is
            requested.
        activated : function
            The function that must be called whenever an item in this overview
            list is activated. This corresponds to the default action.

        """

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
        elif event.matches(QG.QKeySequence.SelectAll):
            # If so, select all items in the list
            self.selectAll()
        # Else, handle as normal
        else:
            super().keyPressEvent(event)


# Class that provides a special threaded progress dialog
# FIXME: Figure out why this dialog can stall forever when used in MPI on Linux
# This always happens on Travis CI, very rarely on Azure Pipelines and never on
# any Linux machine (supercomputer or personal) I have access to.
class ThreadedProgressDialog(QW.QProgressDialog):
    """
    Defines the :class:`~ThreadedProgressDialog` class for the Projection GUI.

    This class provides a :class:`~PyQt5.QtWidgets.QProgressDialog` class that
    automatically executes a provided operation on a separate thread, allowing
    for the user to interrupt it.

    """

    # Make a signal that is emitted whenever the progress dialog finishes
    finished = QC.pyqtSignal()

    def __init__(self, main_window_obj, *args, **kwargs):
        """
        Initialize an instance of the :class:`~ThreadedProgressDialog` class.

        Parameters
        ----------
        main_window_obj : :obj:`~prism._gui.widgets.MainViewerWindow` object
            Instance of the :class:`~prism._gui.widgets.MainViewerWindow` class
            that acts as the parent of progress dialog.
        args : positional arguments
            The positional arguments that need to be passed to :meth:`~init`.
        kwargs : keyword arguments
            The keyword arguments that need to be passed to :meth:`~init`.

        """

        # Save provided MainWindow obj
        self.main = main_window_obj
        self.pipe = self.main.pipe

        # Call the super constructor
        super().__init__(self.main)

        # Create the progress dialog
        self.init(*args, **kwargs)

    # Create the threaded progress dialog
    def init(self, label, func, *iterables):
        """
        Sets up the progress dialog after it has been initialized.

        This function is mainly responsible for preparing the dialog to be
        opened and the `func` function to be executed.

        Parameters
        ----------
        label : str
            The label that is used as the description of what operation is
            currently being executed.
        func : function
            The function that must be called iteratively using the arguments
            provided in `iterables`.
        iterables : positional arguments
            All iterables that must be used to call `func` with.

        """

        # Set the label and cancel button
        self.setLabelText(label)
        self.setCancelButtonText("Abort")

        # Determine the minimum length of iterables
        min_len = min([len(iterable) for iterable in iterables])

        # Set the range of this progress dialog
        self.setRange(0, min_len)

        # Make this progress dialog application modal
        self.setWindowModality(QC.Qt.ApplicationModal)
        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(
            QC.Qt.WindowTitleHint |
            QC.Qt.Dialog |
            QC.Qt.CustomizeWindowHint)
        self.setAttribute(QC.Qt.WA_DeleteOnClose)
        self.setAutoReset(False)

        # Setup the run_map that will be used
        self.run_map = map(func, *iterables)

    # This function simply calls open()
    def __call__(self):
        """
        Calls and returns the result of :meth:`~open`.

        """

        return(self.open())

    # This function executes the entire run_map until finished or aborted
    def open(self):
        """
        Opens the progress dialog and starts the execution of the requested
        operation.

        Returns
        -------
        result : bool
            Whether or not the operations ended successfully, which can be used
            by other functions to determine if it should continue.

        """

        # Initialize the traced thread
        self.thread = TracedControllerThread(self, self.run_map)

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

        # Emit that the progress dialog has finished
        self.finished.emit()

        # Return if dialog finished successfully or not
        return(self.successful)

    # This function sets an attribute and serves as a slot
    @QC.pyqtSlot()
    def set_successful_finish(self):
        """
        Qt slot that marks the operation as 'successful'.

        """

        self.successful = True

    # This function raises an exception caught in the controller thread
    @QC.pyqtSlot(Exception)
    def raise_exception(self, exception):
        """
        Qt slot that raises a provided exception.

        """

        raise exception

    # This function kills all worker threads and then the controller thread
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def kill_threads(self):
        """
        Terminates all currently running threads besides the main thread (on
        all MPI ranks) and returns control to the main thread.

        This function is the sole way to abort the operation.

        %(qt_slot)s

        """

        # Set all worker threads to 'killed'
        for rank in range(1, self.pipe._comm.size):
            self.pipe._comm.send(True, rank, 671589+rank)

        # Set this thread to killed
        self.thread.killed = True

        # Make all workers wait for 1 second to force their system trace
        self.pipe._make_call_workers(sleep, 1)

        # Wait for this thread to be killed
        self.thread.join()

        # Use an MPI Barrier to make sure that all threads were killed
        # This means that the controller also has to wait for a second
        self.pipe._make_call('_comm.Barrier')


# Special system traced thread that stops whenever killed is set to True
class TracedThread(threading.Thread):
    """
    Defines the :class:`~TracedThread` base class.

    This class is used to create traceable threads, which allows for those
    threads to be terminated from the outside.

    """

    def __init__(self):
        """
        Initialize an instance of the :class:`~TracedThread` class.

        """

        # Set killed to False
        self.killed = False

        # Call super constructor
        super().__init__(None)

    # Make a custom system tracer
    def global_trace(self, frame, event, arg):  # pragma: no cover
        """
        Provides the global system tracer function that automatically
        terminates this thread if its :attr:`~killed` attribute is set to
        *True*.

        """

        # If killed is True, kill thread at the next function call
        if self.killed and (event == 'call'):
            raise SystemExit


# https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
# System traced controller thread that loops over a provided map iterator
class TracedControllerThread(QC.QObject, TracedThread):
    """
    Defines the :class:`~TracedControllerThread` class.

    This class creates a traceable thread that simultaneously can also be used
    by Qt.

    """

    # Define a signal that sends out the number of finished iterations
    n_finished = QC.pyqtSignal(int)

    # Define a signal that is emitted when the thread finishes executing
    finished = QC.pyqtSignal()

    # Define a signal that is emitted whenever an exception occurs
    exception = QC.pyqtSignal(Exception)

    def __init__(self, parent, run_map):
        """
        Initialize an instance of the :class:`~TracedControllerThread` class.

        Parameters
        ----------
        parent : :obj:`~PyQt5.QtWidgets.QWidget` object or None
            The parent widget for this dialog or *None* for no parent.
        run_map : iterator
            The iterator that must be iterated over on the separate thread.

        """

        # Save provided map iterator
        self.run_map = run_map

        # Call the super constructors
        super().__init__(parent)
        TracedThread.__init__(self)

    # This function gets called when TracedThread.start() is called
    def run(self):
        """
        Executes the operations whenever this thread is started.

        """

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
    """
    Defines the :class:`~TracedWorkerThread` class.

    This class creates a traceable thread that simultaneously can also be used
    by *PRISM* in worker mode.

    """

    def __init__(self, pipeline_obj):
        """
        Initialize an instance of the :class:`~TracedWorkerThread` class.

        Parameters
        ----------
        pipeline_obj : :obj:`~prism.Pipeline` object
            The :class:`~prism.Pipeline` instance this worker thread must use.
            This is required for entering the proper worker mode.

        """

        # Save provided pipeline_obj
        self.pipe = pipeline_obj

        # Call the super constructor
        super().__init__()

    # This function gets called when TracedThread.start() is called
    def run(self):
        """
        Executes the operations whenever this thread is started.

        """

        # Set the system tracer
        sys.settrace(self.global_trace)

        # Start listening for calls on this thread as well
        worker_mode = self.pipe.worker_mode
        worker_mode._WorkerMode__key = -1
        worker_mode.listen_for_calls()


# %% FUNCTION DEFINITIONS
# This function starts up the threads for all workers
def _run_traced_worker_threads(pipeline_obj):
    """
    All workers defined in the provided `pipeline_obj` create a
    :obj:`~TracedWorkerThread` object and use it to listen for calls from the
    controller rank.

    Parameters
    ----------
    pipeline_obj : :obj:`~prism.Pipeline` object
        The :class:`~prism.Pipeline` object all created worker threads must
        use. This is required for entering the proper worker mode.

    """

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
def show_exception_details(parent, *args, **kwargs):
    """
    Creates an instance of the :class:`~ExceptionDialog` class and shows it.

    Parameters
    ----------
    parent : :obj:`~PyQt5.QtWidgets.QWidget` object or None
        The parent widget for this dialog or *None* for no parent.

    Optional
    --------
    args : positional arguments
        The positional arguments that must be passed to the constructor of
        the :class:`~prism._gui.widgets.helpers.ExceptionDialog` class.
    kwargs : keyword arguments
        The keyword arguments that must be passed to the constructor of the
        :class:`~prism._gui.widgets.helpers.ExceptionDialog` class.

    """

    # Create exception message box
    exception_box = ExceptionDialog(parent, *args, **kwargs)

    # Emit the exception signal of the parent if it has it
    if hasattr(parent, 'exception'):
        parent.exception.emit()

    # Show the exception message box
    exception_box.show()
