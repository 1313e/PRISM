# -*- coding: utf-8 -*-

"""
GUI Main Window
===============
Provides the definition of the main window of the Projection GUI.

"""


# %% IMPORTS
# Built-in imports
from collections import OrderedDict as odict
from contextlib import redirect_stdout
from functools import partial
from io import StringIO
from os import path
import sys
from textwrap import dedent
import warnings

# Package imports
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW

# PRISM imports
from prism.__version__ import __version__
from prism._internal import RequestError, RequestWarning
from prism._gui import APP_NAME, DIR_PATH
from prism._gui.widgets.helpers import (
    QW_QAction, QW_QComboBox, QW_QMenu, show_exception_details)
from prism._gui.widgets.overview import OverviewDockWidget
from prism._gui.widgets.preferences import OptionsDialog
from prism._gui.widgets.viewing_area import ViewingAreaDockWidget

# All declaration
__all__ = ['MainViewerWindow']


# %% CLASS DEFINITIONS
# Define class for main viewer window
# TODO: Write documentation (docs and docstrings) for the GUI
class MainViewerWindow(QW.QMainWindow):
    # Define a signal that is emitted whenever an exception is raised
    exception = QC.pyqtSignal()

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
        self.windows_menu = QW_QMenu(self, '&Windows')
        self.toolbars_menu = QW_QMenu(self, '&Toolbars')

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

        # Set the exception handler to an internal message window
        gui_excepthook = partial(show_exception_details, self)
        sys.excepthook = gui_excepthook

        # Turn off all regular logging in Pipeline
        self.was_logging = bool(self.pipe.do_logging)
        self.pipe._make_call('__setattr__', 'do_logging', False)

    # This function creates the menubar in the viewer
    def create_menubar(self):
        # Obtain menubar
        self.menubar = self.menuBar()

        # FILE
        # Create file menu
        file_menu = self.menubar.addMenu('&File')

        # Add save action to file menu
        save_act = QW_QAction(
            self, '&Save view as...',
            shortcut=QC.Qt.CTRL + QC.Qt.Key_S,
            statustip="Save current projection viewing area as an image",
            triggered=self.area_dock.save_view)
        file_menu.addAction(save_act)

        # Add quit action to file menu
        quit_act = QW_QAction(
            self, '&Quit',
            shortcut=QC.Qt.CTRL + QC.Qt.Key_Q,
            statustip="Quit viewer",
            triggered=self.close)
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
        default_layout_act = QW_QAction(
            self, '&Default layout',
            statustip=("Reset all windows and toolbars back to their default "
                       "layout"),
            triggered=self.set_default_dock_positions)
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
        options_act = QW_QAction(
            self, '&Preferences',
            shortcut=QC.Qt.CTRL + QC.Qt.Key_P,
            statustip="Adjust viewer preferences",
            triggered=self.options)
        help_menu.addAction(options_act)

        # Add a separator
        help_menu.addSeparator()

        # Add about action to help menu
        about_act = QW_QAction(
            self, '&About',
            statustip="About %s" % (APP_NAME),
            triggered=self.about)
        help_menu.addAction(about_act)

        # Add details action to help menu
        details_act = QW_QAction(
            self, '&Details',
            statustip="Show the details overview of a specified iteration",
            triggered=self.show_pipeline_details_overview)
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

        # Set logging in Pipeline back to what it was before
        self.pipe._make_call('__setattr__', 'do_logging', self.was_logging)

        # Set the excepthook back to its default value
        sys.excepthook = sys.__excepthook__

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
    def show_pipeline_details_overview(self):
        # Make a details dialog
        details_box = QW.QDialog(self)
        details_box.setWindowModality(QC.Qt.NonModal)
        details_box.setAttribute(QC.Qt.WA_DeleteOnClose)
        details_box.setWindowFlags(
            QC.Qt.WindowSystemMenuHint |
            QC.Qt.Window |
            QC.Qt.WindowCloseButtonHint |
            QC.Qt.MSWindowsOwnDC |
            QC.Qt.MSWindowsFixedSizeDialogHint)
        details_box.setWindowTitle("%s: Pipeline details" % (APP_NAME))

        # Create a grid layout for this dialog
        grid = QW.QGridLayout(details_box)
        grid.setColumnStretch(2, 1)

        # Obtain the latest emul_i
        emul_i = self.pipe._emulator._emul_i

        # Create a combobox for selecting the desired emulator iteration
        emul_i_box = QW_QComboBox()
        grid.addWidget(QW.QLabel("Emulator iteration:"), 0, 0)
        grid.addWidget(emul_i_box, 0, 1)

        # Create a details pages widget
        pages_widget = QW.QStackedWidget()
        grid.addWidget(pages_widget, grid.rowCount(), 0, 1, grid.columnCount())

        # Set signal handling for swapping between details pages
        emul_i_box.currentIndexChanged.connect(pages_widget.setCurrentIndex)

        # Loop over all emulator iterations and add their pages
        for i in range(1, emul_i+1):
            # Initialize a StringIO stream to capture the output with
            with StringIO() as string_stream:
                # Use this stream to capture the overview of details()
                with redirect_stdout(string_stream):
                    # Obtain the details at specified emulator iteration
                    self.pipe._make_call('details', emul_i)

                # Save the entire string stream as a separate object
                details = string_stream.getvalue()

            # Strip gathered details of all whitespaces
            details = details.strip()

            # Cut everything off before "GENERAL"
            index = details.find("GENERAL")
            details = details[index:]

            # Now split details up line-by-line
            details = details.splitlines()

            # Remove all empty lines in details
            while True:
                try:
                    details.remove('')
                except ValueError:
                    break

            # Search for lines that are in all caps, which are group titles
            group_idx = [j for j, line in enumerate(details)
                         if line.isupper() and line[0].isalpha()]
            group_idx.append(-1)

            # Create an empty ordered dict with groups
            groups = odict()

            # Split details at these indices
            for j, k in zip(group_idx[:-1], group_idx[1:]):
                # Extract the part of the list for this group
                group = details[j:k]

                # Extract the name and entry for this group
                name = group[0].capitalize()
                entry = group[1:]

                # If first or last lines contain dashes, remove
                if(entry[-1].count('-') == len(entry[-1])):
                    entry.pop(-1)
                if(entry[0].count('-') == len(entry[0])):
                    entry.pop(0)

                # Loop over all remaining lines in entry and split at \t
                entry = [line.split('\t') for line in entry]

                # Add this group entry to the dict
                groups[name] = entry

            # Make a details layout
            details_layout = QW.QVBoxLayout()
            details_layout.setSizeConstraint(QW.QLayout.SetFixedSize)

            # Make QGroupBoxes for all groups
            for name, entry in groups.items():
                # Make a QGroupBox for this group
                group = QW.QGroupBox(name)
                details_layout.addWidget(group)
                group_layout = QW.QFormLayout()
                group.setLayout(group_layout)

                # Loop over all lines in this group's entry
                for line in entry:
                    # If line is a list with one element, it spans both columns
                    if(len(line) == 1):
                        # Extract this one element
                        line = line[0]

                        # If line is solely dashes, add a separator
                        if(line.count('-') == len(line)):
                            sep_line = QW.QFrame()
                            sep_line.setFrameShape(sep_line.HLine)
                            sep_line.setFrameShadow(sep_line.Sunken)
                            group_layout.addRow(sep_line)
                        # If not, add line as a QLabel
                        else:
                            group_layout.addRow(QW.QLabel(line))
                    # Else, it contains two elements
                    else:
                        group_layout.addRow(*map(QW.QLabel, line))

            # Add a stretch to the layout
            details_layout.addStretch()

            # Add this details_layout to a new widget
            details_page = QW.QWidget(details_box)
            details_page.setLayout(details_layout)
            details_page.setFixedWidth(details_layout.sizeHint().width())

            # Add the page to a scrollarea
            scrollarea = QW.QScrollArea(details_box)
            scrollarea.setWidgetResizable(True)
            scrollarea.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
            scrollarea.horizontalScrollBar().setEnabled(False)
            scrollarea.setWidget(details_page)

            # Set size constraints on the scrollarea
            scrollarea.setFixedWidth(
                details_page.width() +
                scrollarea.verticalScrollBar().sizeHint().width())
            scrollarea.setMaximumHeight(details_page.height() + 2)

            # Add it to the pages_widget and emul_i_box
            pages_widget.addWidget(scrollarea)
            emul_i_box.addItem(str(i))

        # Set size constraints on the details box
        details_box.setFixedWidth(1.1*pages_widget.sizeHint().width())
        details_box.setMaximumHeight(scrollarea.maximumHeight() +
                                     emul_i_box.sizeHint().height() +
                                     grid.spacing()*(grid.rowCount()+2) + 4)

        # Set the emul_i_box to the latest emul_i
        emul_i_box.setCurrentIndex(emul_i-1)

        # Show the details message box
        details_box.show()