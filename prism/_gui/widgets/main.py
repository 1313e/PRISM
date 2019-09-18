# -*- coding: utf-8 -*-

"""
GUI Main Window
===============
Provides the definition of the main window of the Projection GUI.

"""


# %% IMPORTS
# Built-in imports
from contextlib import redirect_stdout
from io import StringIO
from os import path
from textwrap import dedent
import warnings

# Package imports
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW

# PRISM imports
from prism.__version__ import __version__
from prism._internal import RequestError, RequestWarning
from prism._gui import APP_NAME, DIR_PATH
from prism._gui.widgets.helpers import QW_QAction
from prism._gui.widgets.overview import OverviewDockWidget
from prism._gui.widgets.preferences import OptionsDialog
from prism._gui.widgets.viewing_area import ViewingAreaDockWidget

# All declaration
__all__ = ['MainViewerWindow']


# %% CLASS DEFINITIONS
# Define class for main viewer window
# TODO: Write documentation (docs and docstrings) for the GUI
class MainViewerWindow(QW.QMainWindow):
    # Initialize ViewerWindow class
    def __init__(self, qapplication_obj, pipeline_obj, *args, **kwargs):
        # Save qapplication_obj as qapp
        self.qapp = qapplication_obj

        # Save pipeline_obj as pipe
        self.pipe = pipeline_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Set up the main window
        self.init()

    # This function sets up the main window
    def init(self):
        # Turn logging off in the pipeline
        self.was_logging = bool(self.pipe.do_logging)
        self.pipe._make_call('__setattr__', 'do_logging', False)

        # Tell the Projection class that the GUI is being used
        self.all_set_proj_attr('use_GUI', 1)

        # Determine the last emulator iteration
        emul_i = self.pipe._make_call('_emulator._get_emul_i', None)

        # Prepare projections to be made for all iterations
        for i in range(1, emul_i+1):
            # Try to prepare this iteration
            try:
                self.all_call_proj_attr('prepare_projections',
                                        i, None, force=False, figure=True)

            # If this iteration raises a RequestError, it cannot be prepared
            except RequestError as error:
                # If that happens, emit a warning about it
                warnings.warn("%s. Falling back to previous iteration."
                              % (error), RequestWarning, stacklevel=2)

                # Reprepare the previous iteration and break
                self.all_call_proj_attr('prepare_projections',
                                        i-1, None, force=False, figure=True)
                break

        # Save some statistics about pipeline and modellink
        self.n_par = self.pipe._modellink._n_par

        # Make sure that the viewer is deleted when window is closed
        self.setAttribute(QC.Qt.WA_DeleteOnClose)

        # Disable the default context menu (right-click menu)
        self.setContextMenuPolicy(QC.Qt.NoContextMenu)

        # Set window icon and title
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QG.QIcon(path.join(DIR_PATH, 'data/app_icon.ico')))

        # Create statusbar
        self.create_statusbar()

        # Prepare the windows and toolbars menus
        self.windows_menu = QW.QMenu('&Windows', self)
        self.toolbars_menu = QW.QMenu('&Toolbars', self)

        # Get default positions of all dock widgets
        self.default_pos = self.get_default_dock_positions()

        # OVERVIEW DOCK WIDGET
        # Create the projection overview dock widget
        self.overview_dock = OverviewDockWidget(self)

        # Create an action for enabling/disabling the overview
        proj_overview_act = self.overview_dock.toggleViewAction()
        proj_overview_act.setShortcut(QC.Qt.ALT + QC.Qt.SHIFT + QC.Qt.Key_O)
        proj_overview_act.setStatusTip("Enable/disable the 'Overview' window")
        self.windows_menu.addAction(proj_overview_act)

        # VIEWING AREA DOCK WIDGET
        # Create the projection viewing area dock widget
        self.area_dock = ViewingAreaDockWidget(self)

        # Create an action for enabling/disabling the viewing area
        proj_area_act = self.area_dock.toggleViewAction()
        proj_area_act.setShortcut(QC.Qt.ALT + QC.Qt.SHIFT + QC.Qt.Key_V)
        proj_area_act.setStatusTip("Enable/disable the 'Viewing area' window")
        self.windows_menu.addAction(proj_area_act)

        # Create menubar
        self.create_menubar()

        # Set resolution of window
        self.resize(800, 600)

        # Set all dock widgets to their default positions
        self.set_default_dock_positions()

    # This function creates the menubar in the viewer
    def create_menubar(self):
        # Obtain menubar
        self.menubar = self.menuBar()

        # FILE
        # Create file menu
        file_menu = self.menubar.addMenu('&File')

        # Add save action to file menu
        save_act = QW_QAction('&Save view as...', self)
        save_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.Key_S,
            statustip="Save current projection viewing area as an image")
        save_act.triggered.connect(self.area_dock.save_view)
        file_menu.addAction(save_act)

        # Add quit action to file menu
        quit_act = QW_QAction('&Quit', self)
        quit_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.Key_Q,
            statustip="Quit viewer")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # TOOLS
        # Create tools menu, which includes all actions in the proj_toolbar
        tools_menu = self.menubar.addMenu('&Tools')
        tools_menu.addActions(self.area_dock.proj_toolbar.actions())

        # VIEW
        # Create view menu
        view_menu = self.menubar.addMenu('&View')

        # Add the windows submenu to view menu
        view_menu.addMenu(self.windows_menu)

        # Add default layout action to view menu
        default_layout_act = QW_QAction('&Default layout', self)
        default_layout_act.setDetails(
            statustip=("Reset all windows and toolbars back to their default "
                       "layout"))
        default_layout_act.triggered.connect(self.set_default_dock_positions)
        view_menu.addAction(default_layout_act)

        # Add a separator
        view_menu.addSeparator()

        # Add the toolbars submenu to view menu
        view_menu.addMenu(self.toolbars_menu)

        # HELP
        # Create help menu
        help_menu = self.menubar.addMenu('&Help')

        # Add options action to help menu
        self.options = OptionsDialog(self)
        options_act = QW_QAction('&Preferences', self)
        options_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.Key_P,
            statustip="Adjust viewer preferences")
        options_act.triggered.connect(self.options)
        help_menu.addAction(options_act)

        # Add a separator
        help_menu.addSeparator()

        # Add about action to help menu
        about_act = QW_QAction('&About', self)
        about_act.setDetails(
            statustip="About %s" % (APP_NAME))
        about_act.triggered.connect(self.about)
        help_menu.addAction(about_act)

        # Add details action to help menu
        details_act = QW_QAction('&Details', self)
        details_act.setDetails(
            statustip="Show the details overview of the specified iteration")
        details_act.triggered.connect(
            lambda: self.show_pipeline_details_overview(1))
        help_menu.addAction(details_act)

    # This function creates the statusbar in the viewer
    def create_statusbar(self):
        # Obtain statusbar
        self.statusbar = self.statusBar()

    # This function creates a message box with the 'about' information
    def about(self):
        QW.QMessageBox.about(
            self, "About %s" % (APP_NAME), dedent(r"""
                <b>PRISM v%s | %s</b><br>
                Copyright (C) 2019 Ellert van der Velden
                """ % (__version__, APP_NAME)))

    # This function is called when the viewer is closed
    def closeEvent(self, *args, **kwargs):
        # Call the closeEvent of the dock widgets
        self.overview_dock.close()
        self.area_dock.close()

        # Save that Projection GUI is no longer being used
        self.all_set_proj_attr('use_GUI', 0)

        # Set data parameters in Projection class back to defaults
        self.options.reset_options()

        # Turn logging back on in pipeline if it used to be on
        self.pipe._make_call('__setattr__', 'do_logging', self.was_logging)

        # Close the main window
        super().closeEvent(*args, **kwargs)

    # This function allows for projection attributes to be set more easily
    def set_proj_attr(self, name, value):
        setattr(self.pipe, '_Projection__%s' % (name), value)

    # This function is an MPI-version of set_proj_attr
    def all_set_proj_attr(self, name, value):
        self.pipe._make_call('__setattr__', '_Projection__%s' % (name), value)

    # This function allows for projection attributes to be read more easily
    def get_proj_attr(self, name):
        return(getattr(self.pipe, '_Projection__%s' % (name)))

    # This function allows for projection attributes to be called more easily
    def call_proj_attr(self, name, *args, **kwargs):
        return(getattr(self.pipe, '_Projection__%s' % (name))(*args, **kwargs))

    # This function is an MPI-version of call_proj_attr
    def all_call_proj_attr(self, name, *args, **kwargs):
        return(self.pipe._make_call('_Projection__%s' % (name),
                                    *args, **kwargs))

    # This function returns the default positions of dock widgets and toolbars
    def get_default_dock_positions(self):
        # Make dict including the default docking positions
        default_pos = {
            'Viewing area': QC.Qt.RightDockWidgetArea,
            'Overview': QC.Qt.LeftDockWidgetArea}

        # Return it
        return(default_pos)

    # This function sets dock widgets and toolbars to their default position
    def set_default_dock_positions(self):
        # Set the dock widgets and toolbars to their default positions
        # OVERVIEW
        self.overview_dock.setVisible(True)
        self.overview_dock.setFloating(False)
        self.addDockWidget(self.default_pos['Overview'], self.overview_dock)

        # VIEWING AREA
        self.area_dock.setVisible(True)
        self.area_dock.setFloating(False)
        self.addDockWidget(self.default_pos['Viewing area'], self.area_dock)
        self.area_dock.set_default_dock_positions()

    # This function shows the details() overview of a given emulator iteration
    # TODO: Improve the formatting
    def show_pipeline_details_overview(self, emul_i):
        # Initialize a StringIO stream to capture the output with
        with StringIO() as string_stream:
            # Use this stream to capture the overview of details()
            with redirect_stdout(string_stream):
                # Call and obtain the details at specified emulator iteration
                self.pipe._make_call('details', emul_i)

            # Save the entire string stream as a separate object
            details = string_stream.getvalue()

        # Create a details message box for this emulator iteration
        details_box = QW.QMessageBox(self)
        details_box.setWindowModality(QC.Qt.NonModal)
        details_box.setWindowFlags(
            QC.Qt.WindowSystemMenuHint |
            QC.Qt.Window |
            QC.Qt.WindowCloseButtonHint |
            QC.Qt.MSWindowsOwnDC |
            QC.Qt.MSWindowsFixedSizeDialogHint)
        details_box.layout().setSizeConstraint(QW.QLayout.SetFixedSize)
        details_box.setWindowTitle("%s: Pipeline details" % (APP_NAME))
        details_box.setText(details)

        # Show the details message box
        details_box.show()
