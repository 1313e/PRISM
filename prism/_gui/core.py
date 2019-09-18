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
from PyQt5 import QtCore as QC, QtWidgets as QW
from pytest_mpl.plugin import switch_backend

# PRISM imports
from prism._gui import APP_NAME
from prism._gui.widgets import MainViewerWindow

# All declaration
__all__ = ['open_gui']


# %% FUNCTIONS DEFINITIONS
def open_gui(pipeline_obj):
    # Activate worker mode
    with pipeline_obj.worker_mode:
        if pipeline_obj._is_controller:
            # Wrap entire execution in switch_backend of MPL
            # TODO: Currently, this does not properly switch the backend back
            with switch_backend('Agg'):
                # Set some application attributes
                QW.QApplication.setAttribute(QC.Qt.AA_DontShowIconsInMenus,
                                             False)
                QW.QApplication.setAttribute(QC.Qt.AA_EnableHighDpiScaling,
                                             True)
                QW.QApplication.setAttribute(QC.Qt.AA_UseHighDpiPixmaps, True)

                # Obtain application instance
                qapp = QW.QApplication.instance()

                # If qapp is None, create a new one
                if qapp is None:
                    qapp = QW.QApplication([APP_NAME])
                    qapp.setApplicationName(APP_NAME)

                # Make sure that the application quits when last window closes
                qapp.lastWindowClosed.connect(
                    qapp.quit, QC.Qt.QueuedConnection)

                # Initialize main window and draw (show) it
                main_window = MainViewerWindow(qapp, pipeline_obj)
                main_window.show()
                main_window.setVisible(True)

                # Replace KeyboardInterrupt error by system's default handler
                signal.signal(signal.SIGINT, signal.SIG_DFL)

                # Start application
                qapp.exec_()

    # Return the main window instance
    return(main_window)
