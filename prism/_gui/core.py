# -*- coding: utf-8 -*-

"""
GUI Core
========
Contains all core definitions required to make the Projection GUI work.

"""


# %% IMPORTS
# Built-in imports
import signal

# Package imports
from e13tools.utils import docstring_append, raise_error
import matplotlib as mpl
from matplotlib.pyplot import switch_backend
from PyQt5 import QtCore as QC, QtWidgets as QW

# PRISM imports
from prism._docstrings import start_gui_doc, start_gui_doc_pars
from prism._gui import APP_NAME
from prism._gui.widgets import MainViewerWindow
from prism._internal import getCLogger

# All declaration
__all__ = ['start_gui']


# %% FUNCTION DEFINITIONS
# This function starts up the Projection GUI 'Crystal'
@docstring_append(start_gui_doc_pars, '\n\t\n\t')
@docstring_append(start_gui_doc)
def start_gui(pipeline_obj):  # pragma: no cover
    # Create a logger
    logger = getCLogger('GUI')
    logger.info("Starting %s." % (APP_NAME))

    # Import Pipeline class here to avoid an ImportError
    from prism import Pipeline

    # Check if provided pipeline_obj is an instance of the Pipeline class
    if not isinstance(pipeline_obj, Pipeline):
        err_msg = ("Input argument 'pipeline_obj' must be an instance of the "
                   "Pipeline class!")
        raise_error(err_msg, TypeError, logger)

    # Activate worker mode
    with pipeline_obj.worker_mode:
        if pipeline_obj._is_controller:
            # Temporarily switch the backend of MPL
            cur_backend = mpl.rcParams['backend']
            switch_backend('Agg')

            # Obtain application instance
            qapp = QW.QApplication.instance()

            # If qapp is None, create a new one
            if qapp is None:
                QW.QApplication.setAttribute(QC.Qt.AA_EnableHighDpiScaling)
                qapp = QW.QApplication([APP_NAME])

            # Set name of application
            qapp.setApplicationName(APP_NAME)

            # Hide the 'whats this' tooltip on Windows
            QW.QApplication.setAttribute(
                QC.Qt.AA_DisableWindowContextHelpButton)

            # Make sure that the application quits when last window closes
            qapp.lastWindowClosed.connect(qapp.quit,
                                          QC.Qt.QueuedConnection)

            # Initialize main window and draw (show) it
            main_window = MainViewerWindow(pipeline_obj)
            main_window.show()
            main_window.raise_()
            main_window.activateWindow()

            # Replace KeyboardInterrupt error by system's default handler
            signal.signal(signal.SIGINT, signal.SIG_DFL)

            # Start application
            qapp.exec_()

            # Delete this application to stop event processing
            del qapp

            # Switch back to previous backend
            switch_backend(cur_backend)

            # Return the main window instance
            return(main_window)
