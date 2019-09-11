# -*- coding: utf-8 -*-

"""
Projection GUI
==============
Provides all definitions required for the Projection GUI, which can be opened
using the :func:`~open_gui` function, or the :meth:`~prism.Pipeline.open_gui`
method.

"""


# %% IMPORTS
# Built-in imports
from ast import literal_eval
from collections import namedtuple
from contextlib import redirect_stdout
from functools import partial
from io import StringIO
from os import path
import signal
import sys
from textwrap import dedent
import warnings

# Package imports
from matplotlib.backend_bases import _default_filetypes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib import cm
from matplotlib.colors import BASE_COLORS, CSS4_COLORS
from matplotlib.lines import lineMarkers, lineStyles
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
from pytest_mpl.plugin import switch_backend
from sortedcontainers import SortedDict as sdict, SortedSet as sset

# PRISM imports
from prism.__version__ import __version__
from prism._docstrings import proj_depth_doc, proj_res_doc
from prism._internal import RequestError, RequestWarning

# All declaration
__all__ = ['open_gui']


# %% GLOBALS
APP_NAME = "PRISM Projection Viewer"        # Name of application
DIR_PATH = path.dirname(__file__)           # Path to directory of this file


# %% CLASS DEFINITIONS GUI
# Define class for main viewer window
# TODO: This GUI must be able to work in MPI
# TODO: Write documentation (docs and docstrings) for the GUI
class MainViewerWindow(QW.QMainWindow):
    # Initialize ViewerWindow class
    def __init__(self, qapplication_obj, pipeline_obj, *args, **kwargs):
        # Call super constructor
        super().__init__(*args, **kwargs)

        # Save qapplication_obj as qapp
        self.qapp = qapplication_obj

        # Save pipeline_obj as pipe
        self.pipe = pipeline_obj

        # Turn logging off in the pipeline
        self.was_logging = bool(self.pipe.do_logging)
        self.pipe.do_logging = False

        # Tell the Projection class that the GUI is being used
        self.set_proj_attr('use_GUI', 1)

        # Determine the last emulator iteration
        emul_i = self.pipe._emulator._get_emul_i(None)

        # Prepare projections to be made for all iterations
        for i in range(1, emul_i+1):
            # Try to prepare this iteration
            try:
                self.get_proj_attr('prepare_projections')(
                    i, None, force=False, figure=True)

            # If this iteration raises a RequestError, it cannot be prepared
            except RequestError as error:
                # If that happens, emit a warning about it
                warnings.warn("%s Falling back to previous iteration."
                              % (error), RequestWarning, stacklevel=2)

                # Reprepare the previous iteration and break
                self.get_proj_attr('prepare_projections')(
                    i-1, None, force=False, figure=True)
                break

        # Save some statistics about pipeline and modellink
        self.n_par = self.pipe._modellink._n_par

        # Make sure that the viewer is deleted when window is closed
        self.setAttribute(QC.Qt.WA_DeleteOnClose)

        # Disable the default context menu (right-click menu)
        self.setContextMenuPolicy(QC.Qt.NoContextMenu)

        # Create statusbar
        self.create_statusbar()

        # Prepare the windows and toolbars menus
        self.windows_menu = QW.QMenu('&Windows')
        self.toolbars_menu = QW.QMenu('&Toolbars')

        # Get default positions of all dock widgets
        self.default_pos = self.get_default_dock_positions()

        # OVERVIEW DOCK WIDGET
        # Create the projection overview dock widget
        self.overview_dock = OverviewDockWidget(self)
        self.addDockWidget(self.default_pos['Overview'], self.overview_dock)

        # Create an action for enabling/disabling the overview
        proj_overview_act = self.overview_dock.toggleViewAction()
        proj_overview_act.setShortcut(QC.Qt.ALT + QC.Qt.SHIFT + QC.Qt.Key_O)
        proj_overview_act.setStatusTip("Enable/disable the 'Overview' window")
        self.windows_menu.addAction(proj_overview_act)

        # VIEWING AREA DOCK WIDGET
        # Create the projection viewing area dock widget
        self.area_dock = ViewingAreaDockWidget(self)
        self.addDockWidget(self.default_pos['Viewing area'], self.area_dock)

        # Create an action for enabling/disabling the viewing area
        proj_area_act = self.area_dock.toggleViewAction()
        proj_area_act.setShortcut(QC.Qt.ALT + QC.Qt.SHIFT + QC.Qt.Key_V)
        proj_area_act.setStatusTip("Enable/disable the 'Viewing area' window")
        self.windows_menu.addAction(proj_area_act)

        # Create menubar
        self.create_menubar()

        # Set resolution of window
        self.resize(800, 600)

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
        quit_act.triggered.connect(self.qapp.closeAllWindows)
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
        self.overview_dock.closeEvent(*args, **kwargs)
        self.area_dock.closeEvent(*args, **kwargs)

        # Save that Projection GUI is no longer being used
        self.set_proj_attr('use_GUI', 0)

        # Set data parameters in Projection class back to defaults
        self.options.reset_options()

        # Turn logging back on in pipeline if it used to be on
        self.pipe.do_logging = self.was_logging

        # Close the main window
        super().closeEvent(*args, **kwargs)

        # Quit the application
        self.qapp.quit()

    # This function allows for projection attributes to be set more easily
    def set_proj_attr(self, name, value):
        setattr(self.pipe, "_Projection__%s" % (name), value)

    # This function allows for projection attributes to be read more easily
    def get_proj_attr(self, name):
        return(getattr(self.pipe, "_Projection__%s" % (name)))

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
                self.pipe.details(emul_i)

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


# Define class for the projection viewing area dock widget
# TODO: Allow for multiple viewing areas to co-exist?
class ViewingAreaDockWidget(QW.QDockWidget):
    def __init__(self, main_window_obj, *args, **kwargs):
        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.set_proj_attr = self.main.set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr

        # Call super constructor
        super().__init__("Viewing area", self.main, *args, **kwargs)

        # Create the projection viewing area
        self.init()

    # This function creates the main projection viewing area
    def init(self):
        # Create an MdiArea for the viewing area
        self.area_window = QW.QMainWindow()
        self.proj_area = QW.QMdiArea(self)
        self.area_window.setCentralWidget(self.proj_area)
        self.proj_area.setFocus()
        self.setWidget(self.area_window)

        # options for proj_area
        self.proj_area.setViewMode(QW.QMdiArea.SubWindowView)
        self.proj_area.setOption(QW.QMdiArea.DontMaximizeSubWindowOnActivation)
        self.proj_area.setActivationOrder(QW.QMdiArea.StackingOrder)
        self.proj_area.setStatusTip("Main projection viewing area")

        # options for area_window
        self.area_window.setAttribute(QC.Qt.WA_DeleteOnClose)
        self.area_window.setContextMenuPolicy(QC.Qt.NoContextMenu)

        # Obtain dict of default docking positions
        self.default_pos = self.get_default_dock_positions()

        # Add toolbar to the projection viewer
        self.create_projection_toolbar()

    # This function saves the current state of the viewer to file
    # TODO: See if the window frames can be removed from the saved image
    def save_view(self):
        # Get dict of all file extensions allowed
        exts = sdict({
            'Portable Network Graphics': "*.png",
            'Joint Photographic Experts Group': "*.jpg *.jpeg",
            'Windows Bitmap': "*.bmp",
            'Portable Pixmap': "*.ppm",
            'X11 Bitmap': "*.xbm",
            'X11 Pixmap': "*.xpm"})

        # Set default extension
        default_ext = '*.png'

        # Initialize empty list of filters and default filter
        file_filters = []
        default_filter = None

        # Obtain list with the different file filters
        for name, ext in exts.items():
            # Create proper string layout for this filter
            file_filter = "%s (%s)" % (name, ext)
            file_filters.append(file_filter)

            # If this extension is the default one, save it as such
            if default_ext in file_filter:
                default_filter = file_filter

        # Add 'All (Image) Files' filter to the list of filters for convenience
        file_filters.append("All Image Files (%s)" % (' '.join(exts.values())))
        file_filters.append("All Files (*)")

        # Combine list into a single string
        file_filters = ';;'.join(file_filters)

        # Create an OS-dependent options dict
        options = {}

        # Do not use Linux' native dialog as it is bad on some dists
        if sys.platform.startswith('linux'):
            options = {'options': QW.QFileDialog.DontUseNativeDialog}

        # Open the file saving system
        filename, _ = QW.QFileDialog.getSaveFileName(
            parent=self.main,
            caption="Save view as...",
            directory=path.join(self.pipe._working_dir, "proj_area.png"),
            filter=file_filters,
            initialFilter=default_filter,
            **options)

        # If filename was provided, save image
        if(filename != ''):
            # Grab the current state of the projection area as a Pixmap
            pixmap = self.proj_area.grab()

            # Save pixmap with chosen filename
            pixmap.save(filename)

    # This function is called when the main window is closed
    def closeEvent(self, *args, **kwargs):
        # Close the main window in this widget
        self.area_window.closeEvent(*args, **kwargs)

        # Close the projection viewer
        super().closeEvent(*args, **kwargs)

    # This function returns the default positions of dock widgets and toolbars
    def get_default_dock_positions(self):
        # Make dict including the default docking positions
        default_pos = {
            'Tools': QC.Qt.TopToolBarArea}

        # Return it
        return(default_pos)

    # This function sets dock widgets and toolbars to their default position
    def set_default_dock_positions(self):
        # Set the dock widgets and toolbars to their default positions
        # TOOLS TOOLBAR
        self.proj_toolbar.setVisible(True)
        self.area_window.addToolBar(self.default_pos['Tools'],
                                    self.proj_toolbar)

    # This function creates the toolbar of the projection viewing area
    # TODO: Allow for the formatting inside a figure to be modified
    def create_projection_toolbar(self):
        # Create toolbar for projection viewer
        self.proj_toolbar = QW.QToolBar("Tools", self)
        self.area_window.addToolBar(self.default_pos['Tools'],
                                    self.proj_toolbar)

        # Create an action for enabling/disabling the toolbar
        proj_toolbar_act = self.proj_toolbar.toggleViewAction()
        proj_toolbar_act.setText("Tools toolbar")
        proj_toolbar_act.setStatusTip("Enable/disable the 'Tools' toolbar")
        self.main.toolbars_menu.addAction(proj_toolbar_act)

        # Add tools for manipulating projection figures

        # Add a separator
        self.proj_toolbar.addSeparator()

        # Add action for reoption the view
        reset_act = QW_QAction("&Reset", self)
        reset_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_R,
            statustip="Reset projection viewing area to its default state")
        self.proj_toolbar.addAction(reset_act)

        # Add a separator
        self.proj_toolbar.addSeparator()

        # Add action for cascading all subwindows
        cascade_act = QW_QAction("&Cascade", self)
        cascade_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_C,
            statustip="Cascade all subwindows")
        cascade_act.triggered.connect(self.proj_area.cascadeSubWindows)
        self.proj_toolbar.addAction(cascade_act)

        # Add action for tiling all subwindows
        tile_act = QW_QAction("&Tile", self)
        tile_act.setDetails(
                shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_T,
                statustip="Tile all subwindows")
        tile_act.triggered.connect(self.proj_area.tileSubWindows)
        self.proj_toolbar.addAction(tile_act)

        # Add action for closing all subwindows
        close_act = QW_QAction("Close all", self)
        close_act.setDetails(
                shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_X,
                statustip="Close all subwindows")
        close_act.triggered.connect(self.proj_area.closeAllSubWindows)
        self.proj_toolbar.addAction(close_act)


# Create class for the projection overview dock widget
class OverviewDockWidget(QW.QDockWidget):
    # TODO: Allow for the lists to be sorted differently?
    def __init__(self, main_window_obj, *args, **kwargs):
        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.set_proj_attr = self.main.set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr

        # Call the super constructor
        super().__init__("Overview", self.main, *args, **kwargs)

        # Create the overview widget
        self.init()

    # This function is called when the main window is closed
    def closeEvent(self, *args, **kwargs):
        # Close all currently opened figures and subwindows
        for fig, subwindow in self.proj_fig_registry.values():
            plt.close(fig)
            subwindow.close()

        # Close the projection overview
        super().closeEvent(*args, **kwargs)

    # This function creates the projection overview
    def init(self):
        # Create an overview
        overview_widget = QW.QWidget()
        self.proj_overview = QW.QVBoxLayout()
        overview_widget.setLayout(self.proj_overview)
        self.setWidget(overview_widget)

        # Create empty dict containing all projection figure instances
        self.proj_fig_registry = {}

        # Make lists of all hcubes and their names
        self.hcubes = list(self.get_proj_attr('hcubes'))
        self.names = [self.get_proj_attr('get_hcube_name')(hcube)
                      for hcube in self.hcubes]

        # Divide all hcubes up into three different lists
        # Drawn; available; unavailable
        unavail_hcubes = [self.get_proj_attr('get_hcube_name')(hcube)
                          for hcube in self.get_proj_attr('create_hcubes')]
        avail_hcubes = [name for name in self.names
                        if name not in unavail_hcubes]
        drawn_hcubes = []

        # DRAWN PROJECTIONS
        # Add list for drawn projections
        self.proj_overview.addWidget(QW.QLabel("Drawn:"))
        self.proj_list_d = QW.QListWidget()
        self.proj_list_d.addItems(drawn_hcubes)
        self.proj_list_d.setStatusTip("Lists all projections that have been "
                                      "drawn")

        # Set a variety of properties
        self.proj_list_d.setAlternatingRowColors(True)
        self.proj_list_d.setSortingEnabled(True)
        self.proj_list_d.setSelectionMode(
            QW.QAbstractItemView.ExtendedSelection)
        self.proj_list_d.setContextMenuPolicy(QC.Qt.CustomContextMenu)

        # Add signal handling
        self.create_drawn_context_menu()
        self.proj_list_d.customContextMenuRequested.connect(
            self.show_drawn_context_menu)
        self.proj_list_d.itemActivated.connect(
            lambda: self.show_projection_figures(
                self.proj_list_d.selectedItems()))

        # Add list to overview
        self.proj_overview.addWidget(self.proj_list_d)

        # AVAILABLE PROJECTIONS
        # Add list for available projections
        self.proj_overview.addWidget(QW.QLabel("Available:"))
        self.proj_list_a = QW.QListWidget()
        self.proj_list_a.addItems(avail_hcubes)
        self.proj_list_a.setStatusTip("Lists all projections that have been "
                                      "calculated but not drawn")

        # Set a variety of properties
        self.proj_list_a.setAlternatingRowColors(True)
        self.proj_list_a.setSortingEnabled(True)
        self.proj_list_a.setSelectionMode(
            QW.QAbstractItemView.ExtendedSelection)
        self.proj_list_a.setContextMenuPolicy(QC.Qt.CustomContextMenu)

        # Add signal handling
        self.create_available_context_menu()
        self.proj_list_a.customContextMenuRequested.connect(
            self.show_available_context_menu)
        self.proj_list_a.itemActivated.connect(
            lambda: self.draw_projection_figures(
                self.proj_list_a.selectedItems()))

        # Add list to overview
        self.proj_overview.addWidget(self.proj_list_a)

        # UNAVAILABLE PROJECTIONS
        # Add list for projections that can be created
        self.proj_overview.addWidget(QW.QLabel("Unavailable:"))
        self.proj_list_u = QW.QListWidget()
        self.proj_list_u.addItems(unavail_hcubes)
        self.proj_list_u.setStatusTip("Lists all projections that have not "
                                      "been calculated")

        # Set a variety of properties
        self.proj_list_u.setAlternatingRowColors(True)
        self.proj_list_u.setSortingEnabled(True)
        self.proj_list_u.setSelectionMode(
            QW.QAbstractItemView.ExtendedSelection)
        self.proj_list_u.setContextMenuPolicy(QC.Qt.CustomContextMenu)

        # Add signal handling
        self.create_unavailable_context_menu()
        self.proj_list_u.customContextMenuRequested.connect(
            self.show_unavailable_context_menu)
        self.proj_list_u.itemActivated.connect(
            lambda: self.create_projection_figures(
                self.proj_list_u.selectedItems()))

        # Add list to overview
        self.proj_overview.addWidget(self.proj_list_u)

    # This function creates the context menu for drawn projections
    def create_drawn_context_menu(self):
        # Create context menu
        menu = QW.QMenu('Drawn')

        # Make shortcut for obtaining selected items
        list_items = self.proj_list_d.selectedItems

        # Add show action to menu
        show_act = QW_QAction('S&how', self)
        show_act.setDetails(
            statustip="Show selected projection figure(s)")
        show_act.triggered.connect(
            lambda: self.show_projection_figures(list_items()))
        menu.addAction(show_act)

        # Add save action to menu
        save_act = QW_QAction('&Save', self)
        save_act.setDetails(
            statustip="Save selected projection figure(s) to file")
        save_act.triggered.connect(
            lambda: self.save_projection_figures(list_items()))
        menu.addAction(save_act)

        # Add save as action to menu
        save_as_act = QW_QAction('Save &as...', self)
        save_as_act.setDetails(
            statustip="Save selected projection figure(s) to chosen file")
        save_as_act.triggered.connect(
            lambda: self.save_projection_figures(list_items(), choose=True))
        menu.addAction(save_as_act)

        # Add redraw action to menu
        redraw_act = QW_QAction('&Redraw', self)
        redraw_act.setDetails(
            statustip="Redraw selected projection figure(s)")
        redraw_act.triggered.connect(
            lambda: self.redraw_projection_figures(list_items()))
        menu.addAction(redraw_act)

        # Add close action to menu
        close_act = QW_QAction('&Close', self)
        close_act.setDetails(
            statustip="Close selected projection figure(s)")
        close_act.triggered.connect(
            lambda: self.close_projection_figures(list_items()))
        menu.addAction(close_act)

        # Add details action to menu (single item only)
        self.details_u_act = QW_QAction('De&tails', self)
        self.details_u_act.setDetails(
            statustip="Show details about selected projection figure")
        self.details_u_act.triggered.connect(
            lambda: self.details_projection_figure(list_items()[0]))
        menu.addAction(self.details_u_act)

        # Save made menu as an attribute
        self.context_menu_d = menu

    # This function shows the context menu for drawn projections
    def show_drawn_context_menu(self):
        # Calculate number of selected items
        n_items = len(self.proj_list_d.selectedItems())

        # If there is currently at least one item selected, show context menu
        if n_items:
            # If there is exactly one item selected, enable details
            self.details_u_act.setEnabled(n_items == 1)

            # Show context menu
            self.context_menu_d.popup(QG.QCursor.pos())

    # This function creates the context menu for available projections
    def create_available_context_menu(self):
        # Create context menu
        menu = QW.QMenu('Available')

        # Make shortcut for obtaining selected items
        list_items = self.proj_list_a.selectedItems

        # Add draw action to menu
        draw_act = QW_QAction('&Draw', self)
        draw_act.setDetails(
            statustip="Draw selected projection figure(s)")
        draw_act.triggered.connect(
            lambda: self.draw_projection_figures(list_items()))
        menu.addAction(draw_act)

        # Add draw&save action to menu
        draw_save_act = QW_QAction('Draw && &Save', self)
        draw_save_act.setDetails(
            statustip="Draw & save selected projection figure(s)")
        draw_save_act.triggered.connect(
            lambda: self.draw_save_projection_figures(list_items()))
        menu.addAction(draw_save_act)

        # Add recreate action to menu
        recreate_act = QW_QAction('&Recreate', self)
        recreate_act.setDetails(
            statustip="Recreate selected projection figure(s)")
        recreate_act.triggered.connect(
            lambda: self.recreate_projection_figures(list_items()))
        menu.addAction(recreate_act)

        # Add delete action to menu
        delete_act = QW_QAction('D&elete', self)
        delete_act.setDetails(
            statustip="Delete selected projection figure(s)")
        delete_act.triggered.connect(
            lambda: self.delete_projection_figures(list_items()))
        menu.addAction(delete_act)

        # Add details action to menu (single item only)
        self.details_a_act = QW_QAction('De&tails', self)
        self.details_a_act.setDetails(
            statustip="Show details about selected projection figure")
        self.details_a_act.triggered.connect(
            lambda: self.details_projection_figure(list_items()[0]))
        menu.addAction(self.details_a_act)

        # Save made menu as an attribute
        self.context_menu_a = menu

    # This function shows the context menu for available projections
    def show_available_context_menu(self):
        # Calculate number of selected items
        n_items = len(self.proj_list_a.selectedItems())

        # If there is currently at least one item selected, show context menu
        if n_items:
            # If there is exactly one item selected, enable details
            self.details_a_act.setEnabled(n_items == 1)

            # Show context menu
            self.context_menu_a.popup(QG.QCursor.pos())

    # This function creates the context menu for unavailable projections
    def create_unavailable_context_menu(self):
        # Create context menu
        menu = QW.QMenu('Unavailable')

        # Make shortcut for obtaining selected items
        list_items = self.proj_list_u.selectedItems

        # Add create action to menu
        create_act = QW_QAction('&Create', self)
        create_act.setDetails(
            statustip="Create selected projection figure(s)")
        create_act.triggered.connect(
            lambda: self.create_projection_figures(list_items()))
        menu.addAction(create_act)

        # Add create&draw action to menu
        create_draw_act = QW_QAction('Create && &Draw', self)
        create_draw_act.setDetails(
            statustip="Create & draw selected projection figure(s)")
        create_draw_act.triggered.connect(
            lambda: self.create_draw_projection_figures(list_items()))
        menu.addAction(create_draw_act)

        # Add create, draw & save action to menu
        create_draw_save_act = QW_QAction('Create, Draw && &Save', self)
        create_draw_save_act.setDetails(
            statustip="Create, draw & save selected projection figure(s)")
        create_draw_save_act.triggered.connect(
            lambda: self.create_draw_save_projection_figures(list_items()))
        menu.addAction(create_draw_save_act)

        # Save made menu as an attribute
        self.context_menu_u = menu

    # This function shows the context menu for unavailable projections
    def show_unavailable_context_menu(self):
        # If there is currently at least one item selected, show context menu
        if len(self.proj_list_u.selectedItems()):
            self.context_menu_u.popup(QG.QCursor.pos())

    # This function shows a projection figure in the viewing area
    def show_projection_figures(self, list_items):
        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()

            # Obtain the corresponding figure and subwindow
            fig, subwindow = self.proj_fig_registry[hcube_name]

            # If subwindow is None, create a new one
            if subwindow is None:
                # Create a new subwindow
                subwindow = QW.QMdiSubWindow()
                subwindow.setWindowTitle(hcube_name)

                # Set a few properties of the subwindow
                # TODO: Make subwindow frameless when not being hovered
                subwindow.setOption(QW.QMdiSubWindow.RubberBandResize)
#                subwindow.setWindowFlag(QC.Qt.FramelessWindowHint)

                # Add subwindow to registry
                self.proj_fig_registry[hcube_name][1] = subwindow

            # If subwindow is currently not visible, create a canvas for it
            if not subwindow.isVisible():
                # Create a FigureCanvas instance
                canvas = FigureCanvas(fig)

                # Add canvas to subwindow
                subwindow.setWidget(canvas)

            # Add new subwindow to viewing area if not shown before
            if subwindow not in self.main.area_dock.proj_area.subWindowList():
                self.main.area_dock.proj_area.addSubWindow(subwindow)

            # Show subwindow
            subwindow.showNormal()
            subwindow.setFocus()

        # If auto_tile is set to True, tile all the windows
        if self.main.get_option('auto_tile'):
            self.main.area_dock.proj_area.tileSubWindows()

    # This function removes a projection figure permanently from the register
    def close_projection_figures(self, list_items):
        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()

            # Pop the figure from the registry
            fig, subwindow = self.proj_fig_registry.pop(hcube_name)

            # Close the figure, canvas and subwindow
            plt.close(fig)
            subwindow.close()

            # Move figure from drawn to available
            item = self.proj_list_d.takeItem(
                self.proj_list_d.row(list_item))
            self.proj_list_a.addItem(item)

    # This function draws a projection figure
    # OPTIMIZE: (Re)Drawing a 3D projection figure takes up to 15 seconds
    # TODO: Add threaded progress dialog while drawing the figures
    def draw_projection_figures(self, list_items):
        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()
            hcube = self.hcubes[self.names.index(hcube_name)]

            # Load in the data corresponding to the requested figure
            impl_min, impl_los, proj_res, _ =\
                self.get_proj_attr('get_proj_data')(hcube)

            # Call the proper function for drawing the projection figure
            if(len(hcube) == 2):
                fig = self.get_proj_attr('draw_2D_proj_fig')(
                    hcube, impl_min, impl_los, proj_res)
            else:
                fig = self.get_proj_attr('draw_3D_proj_fig')(
                    hcube, impl_min, impl_los, proj_res)

            # Register figure in the registry
            self.proj_fig_registry[hcube_name] = [fig, None]

            # Move figure from available to drawn
            item = self.proj_list_a.takeItem(
                self.proj_list_a.row(list_item))
            self.proj_list_d.addItem(item)

        # Show all drawn projection figures
        self.show_projection_figures(list_items)

    # This function deletes a projection figure
    # TODO: Avoid reimplementing the __get_req_hcubes() logic here
    def delete_projection_figures(self, list_items, *, skip_warning=False):
        # If skip_warning is False, ask the user if they really want this
        if not skip_warning:
            button_clicked = QW.QMessageBox.warning(
                self, "WARNING: Delete projection(s)",
                ("Are you sure you want to delete the selected projection "
                 "figure(s)? (<i>Note: This action is irreversible!</i>)"),
                QW.QMessageBox.Yes | QW.QMessageBox.No, QW.QMessageBox.No)
        # Else, this answer is always yes
        else:
            button_clicked = QW.QMessageBox.Yes

        # If the answer is yes, loop over all items in list_items
        if(button_clicked == QW.QMessageBox.Yes):
            for list_item in list_items:
                # Retrieve text of list_item
                hcube_name = list_item.text()
                hcube = self.hcubes[self.names.index(hcube_name)]

                # Retrieve the emul_i of this hcube
                emul_i = hcube[0]

                # Open hdf5-file
                with self.pipe._File('r+', None) as file:
                    # Remove the data belonging to this hcube
                    del file['%i/proj_hcube/%s' % (emul_i, hcube_name)]

                # Try to remove figures as well
                fig_path, fig_path_s =\
                    self.get_proj_attr('get_fig_path')(hcube)
                if path.exists(fig_path):
                    os.remove(fig_path)
                if path.exists(fig_path_s):
                    os.remove(fig_path_s)

                # Move figure from available to unavailable
                item = self.proj_list_a.takeItem(
                    self.proj_list_a.row(list_item))
                self.proj_list_u.addItem(item)

    # This function creates a projection figure
    # TODO: Add threaded progress dialog while creating the figures
    def create_projection_figures(self, list_items):
        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()
            hcube = self.hcubes[self.names.index(hcube_name)]

            # Calculate projection data
            _, _ = self.get_proj_attr('analyze_proj_hcube')(hcube)

            # Move figure from unavailable to available
            item = self.proj_list_u.takeItem(self.proj_list_u.row(list_item))
            self.proj_list_a.addItem(item)

    # This function saves a projection figure to file in the normal way
    def save_projection_figures(self, list_items, *, choose=False):
        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()
            hcube = self.hcubes[self.names.index(hcube_name)]

            # Obtain the corresponding figure
            fig, _ = self.proj_fig_registry[hcube_name]

            # Obtain the default figure path
            fig_paths = self.get_proj_attr('get_fig_path')(hcube)
            fig_path = fig_paths[self.get_proj_attr('smooth')]

            # If choose, save using non-default figure path
            if choose:
                # Get dict of all supported file extensions in MPL
                exts = sdict()
                for ext, name in _default_filetypes.items():
                    exts.setdefault(name, []).append("*.%s" % (ext))
                    exts[name].sort()

                # Transform all elements into the proper strings
                for name, ext_list in exts.items():
                    exts[name] = ' '.join(ext_list)

                # Set default extension
                default_ext = '*.png'

                # Initialize empty list of filters and default filter
                file_filters = []
                default_filter = None

                # Obtain list with the different file filters
                for name, ext in exts.items():
                    # Create proper string layout for this filter
                    file_filter = "%s (%s)" % (name, ext)
                    file_filters.append(file_filter)

                    # If this extension is the default one, save it as such
                    if default_ext in file_filter:
                        default_filter = file_filter

                # Add 'All (Image) Files' filter to the list of filters
                file_filters.append("All Image Files (%s)"
                                    % (' '.join(exts.values())))
                file_filters.append("All Files (*)")

                # Combine list into a single string
                file_filters = ';;'.join(file_filters)

                # Create an OS-dependent options dict
                options = {}

                # Do not use Linux' native dialog as it is bad on some dists
                if sys.platform.startswith('linux'):
                    options = {'options': QW.QFileDialog.DontUseNativeDialog}

                # Open the file saving system
                # Don't use native dialog as it is terrible on some Linux dists
                filename, _ = QW.QFileDialog.getSaveFileName(
                    parent=self.main,
                    caption="Save %s as..." % (hcube_name),
                    directory=fig_path,
                    filter=file_filters,
                    initialFilter=default_filter,
                    **options)

                # If filename was provided, save image
                if(filename != ''):
                    fig.savefig(filename)
                # Else, break the loop
                else:
                    break

            # Else, use default figure path
            else:
                fig.savefig(fig_path)

    # This function redraws a projection figure
    def redraw_projection_figures(self, list_items):
        # Close and redraw all projection figures in list_items
        self.close_projection_figures(list_items)
        self.draw_projection_figures(list_items)

    # This function draws and saves a projection figure
    def draw_save_projection_figures(self, list_items):
        # Draw and save all projection figures in list_items
        self.draw_projection_figures(list_items)
        self.save_projection_figures(list_items)

    # This function recreates a projection figure
    def recreate_projection_figures(self, list_items):
        # Ask the user if they really want to recreate the figures
        button_clicked = QW.QMessageBox.warning(
            self, "WARNING: Recreate projection(s)",
            ("Are you sure you want to recreate the selected projection "
             "figure(s)? (<i>Note: This action is irreversible!</i>)"),
            QW.QMessageBox.Yes | QW.QMessageBox.No, QW.QMessageBox.No)

        # Delete and recreate all projection figures in list_items if yes
        if(button_clicked == QW.QMessageBox.Yes):
            self.delete_projection_figures(list_items, skip_warning=True)
            self.create_projection_figures(list_items)

    # This function creates and draws a projection figure
    def create_draw_projection_figures(self, list_items):
        # Create and draw all projection figures in list_items
        self.create_projection_figures(list_items)
        self.draw_projection_figures(list_items)

    # This function creates, draws and saves a projection figure
    def create_draw_save_projection_figures(self, list_items):
        # Create, draw and save all projection figures in list_items
        self.create_projection_figures(list_items)
        self.draw_projection_figures(list_items)
        self.save_projection_figures(list_items)

    # This function shows a details overview of a projection figure
    # TODO: Add section on how the figure was drawn for drawn projections?
    def details_projection_figure(self, list_item):
        # Retrieve text of list_item
        hcube_name = list_item.text()
        hcube = self.hcubes[self.names.index(hcube_name)]

        # Is this a 3D projection?
        is_3D = (len(hcube) == 3)

        # Gather some details about this projection figure
        emul_i = hcube[0]                           # Emulator iteration
        pars = hcube[1:]                            # Plotted parameters
        proj_type = '%iD' % (len(hcube))            # Projection type

        # Open hdf5-file
        with self.pipe._File('r', None) as file:
            # Get the group that contains the data for this projection figure
            group = file["%i/proj_hcube/%s" % (emul_i, hcube_name)]

            # Gather more details about this projection figure
            impl_cut = group.attrs['impl_cut']      # Implausibility cut-offs
            cut_idx = group.attrs['cut_idx']        # Number of wildcards
            res = group.attrs['proj_res']           # Projection resolution
            depth = group.attrs['proj_depth']       # Projection depth

        # Get the percentage of plausible space remaining
        if self.pipe._n_eval_sam[emul_i]:
            pl_space_rem = "{0:#.3%}".format(
                (self.pipe._n_impl_sam[emul_i] /
                 self.pipe._n_eval_sam[emul_i]))
        else:
            pl_space_rem = "N/A"
        pl_space_rem = QW.QLabel(pl_space_rem)

        # Obtain QLabel instances of all details
        emul_i = QW.QLabel(str(emul_i))
        pars = ', '.join([self.pipe._modellink._par_name[par] for par in pars])
        pars = QW.QLabel(pars)
        proj_type = QW.QLabel(proj_type)
        impl_cut = QW.QLabel(str(impl_cut.tolist()))
        cut_idx = QW.QLabel(str(cut_idx))

        # Get the labels for the grid shape and size
        if is_3D:
            grid_shape = QW.QLabel("{0:,}x{0:,}x{1:,}".format(res, depth))
            grid_size = QW.QLabel("{0:,}".format(res*res*depth))
        else:
            grid_shape = QW.QLabel("{0:,}x{1:,}".format(res, depth))
            grid_size = QW.QLabel("{0:,}".format(res*depth))

        # Convert res and depth as well
        res = QW.QLabel("{0:,}".format(res))
        depth = QW.QLabel("{0:,}".format(depth))

        # Create a layout for the details
        details_layout = QW.QVBoxLayout()

        # GENERAL
        # Create a group for the general details
        general_group = QW.QGroupBox("General")
        details_layout.addWidget(general_group)
        general_layout = QW.QFormLayout()
        general_group.setLayout(general_layout)

        # Add general details
        general_layout.addRow("Emulator iteration:", emul_i)
        general_layout.addRow("Parameters:", pars)
        general_layout.addRow("Projection type:", proj_type)
        general_layout.addRow("% of parameter space remaining:",
                              pl_space_rem)

        # PROJECTION DATA
        # Create a group for the projection data details
        data_group = QW.QGroupBox("Projection data")
        details_layout.addWidget(data_group)
        data_layout = QW.QFormLayout()
        data_group.setLayout(data_layout)

        # Add projection data details
        data_layout.addRow("Grid shape:", grid_shape)
        data_layout.addRow("Grid size:", grid_size)
        data_layout.addRow("# of implausibility wildcards:", cut_idx)
        data_layout.addRow("Implausibility cut-offs:", impl_cut)

        # Create a details message box for this projection figure
        details_box = QW.QDialog(self.main)
        details_box.setWindowModality(QC.Qt.NonModal)
        details_box.setWindowFlags(
            QC.Qt.WindowSystemMenuHint |
            QC.Qt.Window |
            QC.Qt.WindowCloseButtonHint |
            QC.Qt.MSWindowsOwnDC |
            QC.Qt.MSWindowsFixedSizeDialogHint)
        details_layout.setSizeConstraint(QW.QLayout.SetFixedSize)
        details_box.setWindowTitle("%s: %s details" % (APP_NAME, hcube_name))
        details_box.setLayout(details_layout)

        # Show the details message box
        details_box.show()


# Define class for options dialog
class OptionsDialog(QW.QDialog):
    def __init__(self, main_window_obj, *args, **kwargs):
        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.n_par = self.main.n_par
        self.set_proj_attr = self.main.set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr

        # Call super constructor
        super().__init__(self.main, *args, **kwargs)

        # Create the options window
        self.init()

    # This function creates the options window
    def init(self):
        # Create a window layout
        window_layout = QW.QVBoxLayout(self)
        window_layout.setSizeConstraint(QW.QLayout.SetFixedSize)

        # Create a tab widget
        window_tabs = QW.QTabWidget()
        window_layout.addWidget(window_tabs)

        # Create a options dict
        self.options_entries = sdict()

        # Obtain function that creates options entries
        options_entry = namedtuple('options_entry',
                                   ['box', 'default', 'value'])
        self.options_entry = partial(options_entry, value=None)

        # Define list with all tabs that should be available in what order
        options_tabs = ['general', 'appearance']

        # Include all tabs named in options_tabs
        for tab in options_tabs:
            getattr(self, 'add_tab_%s' % (tab))(window_tabs)

        # Also add the buttons
        self.add_group_buttons(window_layout)

        # Set default options
        self.reset_options()

        # Set a few properties of options window
        self.setModal(True)                                 # Modality
        self.setWindowTitle("Preferences")                  # Title

        # Add a new method to self.main
        self.main.get_option = self.get_option

    # This function shows the options window
    def __call__(self):
        # Show it
        self.show()

        # Move the options window to the center of the main window
        self.move(self.main.geometry().center()-self.rect().center())

    # This function overrides the closeEvent method
    def closeEvent(self, *args, **kwargs):
        # Close the window
        super().closeEvent(*args, **kwargs)

        # Set all option boxes back to their current values
        self.set_options()

    # This function returns the value of a specific option
    def get_option(self, name):
        return(self.options_entries[name].value)

    # This function creates a new tab
    def create_tab(self, name, tab_widget, *groups_list):
        # Create a tab
        options_tab = QW.QWidget()
        tab_layout = QW.QVBoxLayout()
        options_tab.setLayout(tab_layout)

        # Include all groups named in groups_list
        for group in groups_list:
            getattr(self, 'add_group_%s' % (group))(tab_layout)

        # Add a stretch
        tab_layout.addStretch()

        # Add tab to tab_widget
        tab_widget.addTab(options_tab, name)

    # This function creates a new group
    def create_group(self, name, tab_layout, *options_list):
        # Create a group
        options_group = QW.QGroupBox(name)
        group_layout = QW.QFormLayout()
        options_group.setLayout(group_layout)

        # Include all options named in options_list
        for option in options_list:
            getattr(self, 'add_option_%s' % (option))(group_layout)

        # Add group to tab
        tab_layout.addWidget(options_group)

    # GENERAL TAB
    def add_tab_general(self, *args):
        self.proj_defaults = sdict(self.get_proj_attr('proj_kwargs'))
        self.proj_keys = list(self.proj_defaults.keys())
        self.create_tab("General", *args,
                        'proj_grid', 'proj_kwargs')

    # INTERFACE TAB
    def add_tab_appearance(self, *args):
        self.create_tab("Appearance", *args, 'fonts', 'interface')

    # PROJ_GRID GROUP
    def add_group_proj_grid(self, *args):
        self.create_group("Projection grid", *args,
                          'proj_res', 'proj_depth')

    # PROJ_KWARGS GROUP
    def add_group_proj_kwargs(self, *args):
        self.create_group("Projection keywords", *args,
                          'align', 'show_cuts', 'smooth', 'kwargs_dicts')

    # INTERFACE GROUP
    def add_group_interface(self, *args):
        self.create_group("Interface", *args, 'auto_tile')

    # FONTS GROUP
    def add_group_fonts(self, *args):
        self.create_group("Fonts", *args, 'text_fonts')

    # TEXT_FONTS OPTION
    # TODO: Further implement this
    def add_option_text_fonts(self, group_layout):
        # PLAIN TEXT
        # Create a font families combobox
        plain_families_box = QW.QFontComboBox()
        plain_families_box.setFontFilters(QW.QFontComboBox.MonospacedFonts)
        plain_families_box.setEditable(True)
        plain_families_box.setInsertPolicy(QW.QComboBox.NoInsert)

        # Create a font size spinbox
        plain_size_box = QW.QSpinBox()
        plain_size_box.setRange(7, 9999999)

        # RICH TEXT
        # Create a font families combobox
        rich_families_box = QW.QFontComboBox()
        rich_families_box.setEditable(True)
        rich_families_box.setInsertPolicy(QW.QComboBox.NoInsert)

        # Create a font size spinbox
        rich_size_box = QW.QSpinBox()
        rich_size_box.setRange(7, 9999999)

        # Create a grid for the families and size boxes
        font_grid = QW.QGridLayout()
        group_layout.addRow(font_grid)

        # Add everything to this grid
        font_grid.addWidget(QW.QLabel("Plain text:"), 0, 0)
        font_grid.addWidget(plain_families_box, 0, 1)
        font_grid.addWidget(QW.QLabel("Size:"), 0, 2)
        font_grid.addWidget(plain_size_box, 0, 3)
        font_grid.addWidget(QW.QLabel("Rich text:"), 1, 0)
        font_grid.addWidget(rich_families_box, 1, 1)
        font_grid.addWidget(QW.QLabel("Size:"), 1, 2)
        font_grid.addWidget(rich_size_box, 1, 3)

    # AUTO_TILE OPTION
    def add_option_auto_tile(self, group_layout):
        # Make check box for auto tiling
        auto_tile_box = QW.QCheckBox("Auto-tile subwindows")
        auto_tile_box.setToolTip("Set this to automatically tile all "
                                 "projection subwindows whenever a new one is "
                                 "added")
        auto_tile_box.stateChanged.connect(self.enable_save_button)
        self.options_entries['auto_tile'] =\
            self.options_entry(auto_tile_box, True)
        group_layout.addRow(auto_tile_box)

    # PROJ_RES OPTION
    def add_option_proj_res(self, group_layout):
        # Make spinbox for option proj_res
        proj_res_box = QW.QSpinBox()
        proj_res_box.setRange(0, 9999999)
        proj_res_box.setToolTip(proj_res_doc)
        proj_res_box.valueChanged.connect(self.enable_save_button)
        self.options_entries['proj_res'] =\
            self.options_entry(proj_res_box, self.proj_defaults['proj_res'])
        group_layout.addRow('Resolution:', proj_res_box)

    # PROJ_DEPTH OPTION
    def add_option_proj_depth(self, group_layout):
        # Make spinbox for option proj_depth
        proj_depth_box = QW.QSpinBox()
        proj_depth_box.setRange(0, 9999999)
        proj_depth_box.setToolTip(proj_depth_doc)
        proj_depth_box.valueChanged.connect(self.enable_save_button)
        self.options_entries['proj_depth'] =\
            self.options_entry(proj_depth_box,
                               self.proj_defaults['proj_depth'])
        group_layout.addRow('Depth:', proj_depth_box)

    # EMUL_I OPTION
    def add_option_emul_i(self, group_layout):
        # Make spinbox for option emul_i
        emul_i_box = QW.QSpinBox()
        emul_i_box.setRange(0, self.pipe._emulator._emul_i)
        emul_i_box.valueChanged.connect(self.enable_save_button)
        self.options_entries['emul_i'] =\
            self.options_entry(emul_i_box, self.proj_defaults['emul_i'])
        group_layout.addRow('Iteration:', emul_i_box)

    # PROJ_TYPE OPTION
    def add_option_proj_type(self, group_layout):
        # Make check boxes for 2D and 3D projections
        # 2D projections
        proj_2D_box = QW.QCheckBox('2D')
        proj_2D_box.setEnabled(self.n_par > 2)
        proj_2D_box.stateChanged.connect(self.enable_save_button)
        self.options_entries['proj_2D'] =\
            self.options_entry(proj_2D_box, self.proj_defaults['proj_2D'])

        # 3D projections
        proj_3D_box = QW.QCheckBox('3D')
        proj_3D_box.setEnabled(self.n_par > 2)
        proj_3D_box.stateChanged.connect(self.enable_save_button)
        self.options_entries['proj_3D'] =\
            self.options_entry(proj_3D_box, self.proj_defaults['proj_3D'])

        # Create layout for proj_type and add it to options layout
        proj_type_box = QW.QHBoxLayout()
        proj_type_box.addWidget(proj_2D_box)
        proj_type_box.addWidget(proj_3D_box)
        proj_type_box.addStretch()
        group_layout.addRow('Projection type:', proj_type_box)

    # ALIGN OPTION
    def add_option_align(self, group_layout):
        # Make drop-down menu for align
        # Column align
        align_col_box = QW.QRadioButton('Column')
        align_col_box.toggled.connect(self.enable_save_button)
        self.options_entries['align_col'] =\
            self.options_entry(align_col_box,
                               self.proj_defaults['align'] == 'col')

        # Row align
        align_row_box = QW.QRadioButton('Row')
        align_row_box.toggled.connect(self.enable_save_button)
        self.options_entries['align_row'] =\
            self.options_entry(align_row_box,
                               self.proj_defaults['align'] == 'row')

        # Create layout for align and add it to options layout
        align_box = QW.QHBoxLayout()
        align_box.addWidget(align_col_box)
        align_box.addWidget(align_row_box)
        align_box.addStretch()
        group_layout.addRow('Alignment:', align_box)

    # SHOW_CUTS OPTION
    def add_option_show_cuts(self, group_layout):
        # Make check box for show_cuts
        show_cuts_box = QW.QCheckBox()
        show_cuts_box.stateChanged.connect(self.enable_save_button)
        self.options_entries['show_cuts'] =\
            self.options_entry(show_cuts_box, self.proj_defaults['show_cuts'])
        group_layout.addRow('Show cuts?', show_cuts_box)

    # SMOOTH OPTION
    def add_option_smooth(self, group_layout):
        # Make check box for smooth
        smooth_box = QW.QCheckBox()
        smooth_box.stateChanged.connect(self.enable_save_button)
        self.options_entries['smooth'] =\
            self.options_entry(smooth_box, self.proj_defaults['smooth'])
        group_layout.addRow('Smooth?', smooth_box)

    # KWARGS_DICTS OPTION
    def add_option_kwargs_dicts(self, group_layout):
        # Create a kwargs_dicts_box
        kwargs_dicts_box = KwargsDictBoxLayout(self)
        group_layout.addRow('Projection keyword dicts:', kwargs_dicts_box)

        # Add all kwargs_dicts to it
        # FIG_KWARGS
        kwargs_dicts_box.add_dict(
            "Figure", 'fig_kwargs',
            std_entries=['dpi', 'figsize'],
            banned_entries=[*self.get_proj_attr('pop_fig_kwargs')])

        # IMPL_KWARGS_2D
        kwargs_dicts_box.add_dict(
            "2D implausibility", 'impl_kwargs_2D',
            std_entries=['linestyle', 'marker', 'color'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs'), 'cmap'])

        # IMPL_KWARGS_3D
        kwargs_dicts_box.add_dict(
            "3D implausibility", 'impl_kwargs_3D',
            std_entries=['cmap'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs')])

        # LOS_KWARGS_2D
        kwargs_dicts_box.add_dict(
            "2D line-of-sight", 'los_kwargs_2D',
            std_entries=['linestyle', 'marker', 'color'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs'), 'cmap'])

        # LOS_KWARGS_3D
        kwargs_dicts_box.add_dict(
            "3D line-of-sight", 'los_kwargs_3D',
            std_entries=['cmap'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs')])

        # LINE_KWARGS_EST
        kwargs_dicts_box.add_dict(
            "Estimate lines", 'line_kwargs_est',
            std_entries=['linestyle', 'color'],
            banned_entries=[])

        # LINE_KWARGS_CUT
        kwargs_dicts_box.add_dict(
            "Cut-off lines", 'line_kwargs_cut',
            std_entries=['linestyle', 'color'],
            banned_entries=[])

    # BUTTONS GROUP
    def add_group_buttons(self, window_layout):
        # Create a buttons layout
        buttons_layout = QW.QHBoxLayout()
        window_layout.addLayout(buttons_layout)
        buttons_layout.addStretch()

        # Make a 'Reset' button
        reset_but = QW.QPushButton("&Reset")
        reset_but.setToolTip("Reset to defaults")
        reset_but.clicked.connect(self.reset_options)
        buttons_layout.addWidget(reset_but)

        # Make a 'Save' button
        save_but = QW.QPushButton("&Save")
        save_but.setToolTip("Save options")
        save_but.clicked.connect(self.save_options)
        save_but.setEnabled(False)
        self.save_but = save_but
        buttons_layout.addWidget(save_but)

        # Make a 'Close' button
        close_but = QW.QPushButton("&Close")
        close_but.setToolTip("Close without saving")
        close_but.setDefault(True)
        close_but.clicked.connect(self.close)
        buttons_layout.addWidget(close_but)

    # This function saves the new options values
    def save_options(self):
        # Save all new values
        for key, entry in self.options_entries.items():
            self.options_entries[key] =\
                entry._replace(value=get_box_value(entry.box))

            # If key is a projection parameter, save it in the Pipeline as well
            if key in self.proj_keys:
                # Make abbreviation for the new entry
                entry = self.options_entries[key]

                # Align
                if key in ['align_col', 'align_row']:
                    if entry.box.isChecked():
                        self.set_proj_attr('align', key[6:])
                else:
                    self.set_proj_attr(key, entry.value)

        # Disable the save button
        self.disable_save_button()
        for key, value in self.options_entries.items():
            print(key, value)

    # This function enables the save button
    def enable_save_button(self):
        self.save_but.setEnabled(True)

    # This function disables the save button
    def disable_save_button(self):
        self.save_but.setEnabled(False)

    # This function resets the default options
    def reset_options(self):
        # Reset all options to defaults
        for entry in self.options_entries.values():
            set_box_value(entry.box, entry.default)

        # Save current options
        self.save_options()

    # This function sets the current options
    def set_options(self):
        # Set all options to their current saved values
        for entry in self.options_entries.values():
            set_box_value(entry.box, entry.value)

        # Disable the save button
        self.disable_save_button()


# %% SUPPORT CLASSES
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


# Make a subclass that allows for kwargs dicts to be saved and edited
class KwargsDictBoxLayout(QW.QHBoxLayout):
    # This function creates an editable list of input boxes for the kwargs
    def __init__(self, options_dialog_obj, *args, **kwargs):
        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the kwargs dict window
        self.init()

    # This function creates the kwargs dict window
    def init(self):
        # Initialize the window for the kwargs dict
        self.dict_dialog = KwargsDictDialog(self.options)

        # Add a view button
        view_but = QW.QPushButton('View')
        view_but.setToolTip("View/Edit the projection keyword dicts")
        view_but.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        view_but.clicked.connect(self.dict_dialog)
        self.addWidget(view_but)

    # This function calls the create_tab()-method of dict_dialog
    def add_dict(self, *args, **kwargs):
        self.dict_dialog.create_tab(*args, **kwargs)


# Make a subclass that shows the kwargs dict entries window
class KwargsDictDialog(QW.QDialog):
    def __init__(self, options_dialog_obj, *args, **kwargs):
        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(self.options, *args, **kwargs)

        # Create the kwargs dict window
        self.init()

    # This function creates the kwargs dict window
    def init(self):
        # Create a layout for this window
        window_layout = QW.QVBoxLayout(self)
        window_layout.setSizeConstraint(QW.QLayout.SetFixedSize)

        # Create a tab widget
        # TODO: Use a combobox with stackedwidget instead?
        self.window_tabs = QW.QTabWidget()
        self.window_tabs.setUsesScrollButtons(False)
        window_layout.addWidget(self.window_tabs)
        window_layout.addStretch()

        # Add a close button
        buttons_layout = QW.QHBoxLayout()
        window_layout.addLayout(buttons_layout)
        close_but = QW.QPushButton("Close")
        close_but.clicked.connect(self.close)
        buttons_layout.addStretch()
        buttons_layout.addWidget(close_but)

        # Set some properties for this window
        self.setModal(True)                                         # Modality
        self.setWindowTitle("Viewing projection keyword dicts")     # Title

    # This function shows an editable window with the entries in the dict
    def __call__(self):
        # Show it
        self.show()

        # Move the kwargs_dicts window to the center of the main window
        self.move(self.options.geometry().center()-self.rect().center())

    # This function creates a new tab
    def create_tab(self, name, option_key, *args, **kwargs):
        # Create a tab
        kwargs_tab = KwargsDictDialogTab(self, name, *args, **kwargs)

        # Add this new tab to the options_entries
        self.options.options_entries[option_key] =\
            self.options.options_entry(kwargs_tab,
                                       self.options.proj_defaults[option_key])

        # Add it to the window tabs
        self.window_tabs.addTab(kwargs_tab, name)


# Make a class for describing a kwargs dict tab
class KwargsDictDialogTab(QW.QWidget):
    def __init__(self, kwargs_dict_dialog_obj, name, std_entries,
                 banned_entries, *args, **kwargs):
        # Save provided kwargs_dict_dialog_obj
        self.tab_dialog = kwargs_dict_dialog_obj
        self.options = self.tab_dialog.options
        self.name = name
        self.std_entries = sset(std_entries)
        self.banned_entries = sset(banned_entries)

        # Call super constructor
        super().__init__(self.tab_dialog, *args, **kwargs)

        # Create the kwargs dict window
        self.init()

    # This function creates the kwargs dict tab
    def init(self):
        # Create tab layout
        tab_layout = QW.QVBoxLayout(self)

        # Create a grid for this layout
        self.kwargs_grid = QW.QGridLayout()
        self.kwargs_grid.setColumnStretch(0, 1)
        self.kwargs_grid.setColumnStretch(1, 1)
        tab_layout.addLayout(self.kwargs_grid)

        # Make sure that '' is not in std_entries or banned_entries
        self.std_entries.discard('')
        self.banned_entries.discard('')

        # Create list of available entry types
        self.avail_entries = sset([attr[9:] for attr in dir(self)
                                   if attr.startswith('add_type_')])

        # Convert std_entries to solely contain valid available entry types
        self.std_entries.intersection_update(self.avail_entries)
        self.std_entries.difference_update(self.banned_entries)

        # Add an 'add' button at the bottom of this layout
        add_but = QW.QPushButton('+')
        add_but.setToolTip("Add a new entry")
        add_but.clicked.connect(self.add_editable_entry)
        add_but.clicked.connect(self.options.enable_save_button)
        tab_layout.addWidget(add_but)
        tab_layout.addStretch()

    # This function gets the dict value of a tab
    def get_box_value(self):
        # Create an empty dict to hold the values in
        tab_dict = sdict()

        # Loop over all items in grid and save them to tab_dict
        for row in range(self.kwargs_grid.count()//3):
            # Obtain the entry_type
            entry_type = get_box_value(
                self.kwargs_grid.itemAtPosition(row, 0).widget())

            # If the entry_type is empty, skip this row
            if(entry_type == '' or entry_type in self.banned_entries):
                continue

            # Obtain the value of the corresponding field box
            field_value = get_box_value(
                self.kwargs_grid.itemAtPosition(row, 1).widget())

            # Add this to the dict
            tab_dict[entry_type] = field_value

        # Return tab_dict
        return(tab_dict)

    # This function sets the dict value of a tab
    # OPTIMIZE: Reuse grid items that were already in the grid?
    def set_box_value(self, tab_dict):
        # Remove all items in the grid
        for _ in range(self.kwargs_grid.count()):
            item = self.kwargs_grid.takeAt(0)
            item.widget().close()
            del item

        # Add all items in tab_dict to kwargs_tab
        for row, (entry_type, field_value) in enumerate(tab_dict.items()):
            # Add a new entry to this tab
            self.add_editable_entry()

            # Set this entry to the proper type
            set_box_value(self.kwargs_grid.itemAtPosition(row, 0).widget(),
                          entry_type)

            # Set the value of the corresponding field
            set_box_value(self.kwargs_grid.itemAtPosition(row, 1).widget(),
                          field_value)

    # This function adds an editable entry
    def add_editable_entry(self):
        # Create a combobox with different standard kwargs
        kwargs_box = QW.QComboBox()
        kwargs_box.addItem('')
        kwargs_box.addItems(self.std_entries)
        kwargs_box.setToolTip("Select the standard keyword field to be added "
                              "or type it manually")
        kwargs_box.setEditable(True)
        kwargs_box.setInsertPolicy(QW.QComboBox.NoInsert)
        kwargs_box.completer().setCompletionMode(QW.QCompleter.PopupCompletion)
        kwargs_box.currentTextChanged.connect(
            lambda x: self.entry_type_selected(x, kwargs_box))
        kwargs_box.currentTextChanged.connect(self.options.enable_save_button)

        # Create a delete button
        delete_but = QW.QPushButton('X')
        delete_but.setToolTip("Delete this entry")
        delete_but.setMaximumSize(16, 16)
        delete_but.clicked.connect(
            lambda: self.remove_editable_entry(kwargs_box))
        delete_but.clicked.connect(self.options.enable_save_button)

        # Determine the number of entries currently in kwargs_grid
        n_rows = self.kwargs_grid.count()//3

        # Make a new editable entry
        self.kwargs_grid.addWidget(kwargs_box, n_rows, 0)
        self.kwargs_grid.addWidget(QW.QWidget(), n_rows, 1)
        self.kwargs_grid.addWidget(delete_but, n_rows, 2)

    # This function deletes an editable entry
    def remove_editable_entry(self, kwargs_box):
        # Determine at what index the provided kwargs_box currently is
        index = self.kwargs_grid.indexOf(kwargs_box)

        # As every row contains 3 items, remove item 3 times at this index
        for _ in range(3):
            # Take the current layoutitem at this index
            item = self.kwargs_grid.takeAt(index)

            # Close the widget in this item and delete the item
            item.widget().close()
            del item

    # This function is called when an item in the combobox is selected
    # TODO: Make sure that two fields cannot have the same name
    def entry_type_selected(self, entry_type, kwargs_box):
        # Determine at what index the provided kwargs_box currently is
        index = self.kwargs_grid.indexOf(kwargs_box)

        # Retrieve what the current field_box is
        cur_box = self.kwargs_grid.itemAt(index+1).widget()

        # Check what entry_type is given and act accordingly
        if(entry_type == ''):
            # If '' is selected, use an empty widget
            field_box = QW.QWidget()
        elif entry_type in self.banned_entries:
            # If one of the banned types is selected, show a warning message
            warn_msg = "%r is not a valid entry type!" % (entry_type)
            field_box = QW.QLabel(warn_msg)
        elif entry_type in self.std_entries:
            # If one of the standard types is selected, add its box
            field_box = getattr(self, 'add_type_%s' % (entry_type))()
        else:
            # If an unknown type is given, add default box if not used already
            if isinstance(cur_box, DefaultBox):
                return
            else:
                field_box = self.add_unknown_type()

        # Replace current field_box with new field_box
        cur_item = self.kwargs_grid.replaceWidget(cur_box, field_box)
        cur_item.widget().close()
        del cur_item

    # This function adds a cmap box
    def add_type_cmap(self):
        # Obtain a list with default colormaps that should be at the top
        std_cmaps = sset(['cividis', 'freeze', 'inferno', 'magma', 'plasma',
                          'rainforest', 'viridis'])
        std_cmaps.update([cmap+'_r' for cmap in std_cmaps])

        # Create a combobox for cmaps
        cmaps_box = QW.QComboBox()
        cmaps_box.addItems(std_cmaps)
        cmaps_box.insertSeparator(cmaps_box.count())
        cmaps_box.addItems(sset(cm.cmap_d))
        cmaps_box.setToolTip("Colormap to be used for the corresponding plot "
                             "type")
        cmaps_box.currentTextChanged.connect(self.options.enable_save_button)
        return(cmaps_box)

    # This function adds a dpi box
    def add_type_dpi(self):
        # Make spinbox for dpi
        dpi_box = QW.QSpinBox()
        dpi_box.setRange(1, 9999999)
        dpi_box.setSingleStep(10)
        dpi_box.setToolTip("DPI to use for the projection figure")
        dpi_box.valueChanged.connect(self.options.enable_save_button)
        return(dpi_box)

    # This function adds a figsize box
    def add_type_figsize(self):
        return(FigSizeBox(self.options))

    # This function adds a linestyle box
    def add_type_linestyle(self):
        # Obtain list with all supported linestyles
        linestyles_lst = [(key, value[6:]) for key, value in lineStyles.items()
                          if value != '_draw_nothing']
        linestyles_lst.sort(key=lambda x: x[0])

        # Make combobox for linestyles
        linestyle_box = QW.QComboBox()
        for i, (linestyle, tooltip) in enumerate(linestyles_lst):
            linestyle_box.addItem(linestyle)
            linestyle_box.setItemData(i, tooltip, QC.Qt.ToolTipRole)
        linestyle_box.setToolTip("Linestyle to be used for the corresponding "
                                 "plot type")
        linestyle_box.currentTextChanged.connect(
            self.options.enable_save_button)
        return(linestyle_box)

    # This function adds a marker box
    def add_type_marker(self):
        # Obtain list with all supported markers
        markers_lst = [(key, value) for key, value in lineMarkers.items()
                       if(value != 'nothing' and isinstance(key, str))]
        markers_lst.sort(key=lambda x: x[0])

        # Make combobox for markers
        marker_box = QW.QComboBox()
        for i, (marker, tooltip) in enumerate(markers_lst):
            marker_box.addItem(marker)
            marker_box.setItemData(i, tooltip, QC.Qt.ToolTipRole)
        marker_box.setToolTip("Marker to be used for the corresponding plot "
                              "type")
        marker_box.currentTextChanged.connect(
            self.options.enable_save_button)
        return(marker_box)

    # This function adds a color box
    def add_type_color(self):
        # Make combobox for colors
        color_box = QW.QComboBox()
        color_box.addItems(sset(BASE_COLORS))
        color_box.insertSeparator(color_box.count())
        color_box.addItems(sset(CSS4_COLORS))
        color_box.setToolTip("Select or type the color to be used for the "
                             "corresponding plot type")
        color_box.setEditable(True)
        color_box.setInsertPolicy(QW.QComboBox.NoInsert)
        color_box.completer().setCompletionMode(QW.QCompleter.PopupCompletion)
        color_box.currentTextChanged.connect(self.options.enable_save_button)
        return(color_box)

    # This function adds a default box
    def add_unknown_type(self):
        return(DefaultBox(self.options))


# Make class with a special box for setting the figsize
class FigSizeBox(QW.QWidget):
    def __init__(self, options_dialog_obj, *args, **kwargs):
        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the figsize box
        self.init()

    # This function creates the figsize box
    def init(self):
        # Create the box_layout
        box_layout = QW.QHBoxLayout(self)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self.setToolTip("Figure size dimensions to use for the projection "
                        "figure")

        # Create two double spinboxes for the width and height
        # WIDTH
        width_box = QW.QDoubleSpinBox()
        width_box.setRange(1, 9999999)
        width_box.setSingleStep(0.1)
        width_box.setToolTip("Width of projection figure")
        width_box.valueChanged.connect(self.options.enable_save_button)
        self.width_box = width_box

        # HEIGHT
        height_box = QW.QDoubleSpinBox()
        height_box.setRange(1, 9999999)
        height_box.setSingleStep(0.1)
        height_box.setToolTip("Height of projection figure")
        height_box.valueChanged.connect(self.options.enable_save_button)
        self.height_box = height_box

        # Also create a textlabel with 'X'
        x_label = QW.QLabel('X')
        x_label.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)

        # Add everything to the box_layout
        box_layout.addWidget(width_box)
        box_layout.addWidget(x_label)
        box_layout.addWidget(height_box)

    # This function retrieves a value of this special box
    def get_box_value(self):
        return((get_box_value(self.width_box), get_box_value(self.height_box)))

    # This function sets the value of this special box
    def set_box_value(self, value):
        set_box_value(self.width_box, value[0])
        set_box_value(self.height_box, value[1])


# Make class for the default lineedit box that allows for type to be selected
class DefaultBox(QW.QWidget):
    def __init__(self, options_dialog_obj, *args, **kwargs):
        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the defaultline box
        self.init()

    # This function creates a double box with type and lineedit
    def init(self):
        # Create the box_layout
        box_layout = QW.QHBoxLayout(self)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_layout = box_layout
        self.setToolTip("Enter the type and value for this unknown entry type")

        # Make a look-up dict for types
        self.type_dict = {
            float: 'float',
            int: 'int',
            str: 'str'}

        # Create a combobox for the type
        type_box = QW.QComboBox()
        type_box.addItems(self.type_dict.values())
        type_box.setToolTip("Type of the entered value")
        type_box.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        type_box.currentTextChanged.connect(self.create_field_box)
        type_box.currentTextChanged.connect(self.options.enable_save_button)
        self.type_box = type_box

        # Make value box corresponding to the current type
        value_box = getattr(self, "add_type_%s" % (type_box.currentText()))()
        self.value_box = value_box

        # Add everything to the box_layout
        box_layout.addWidget(type_box)
        box_layout.addWidget(value_box)

    # This function creates a field_box depending on the type that was selected
    def create_field_box(self, value_type):
        # Obtain a widget box for the specified value_type
        value_box = getattr(self, "add_type_%s" % (value_type))()

        # Set this value_box in the layout
        cur_item = self.box_layout.replaceWidget(self.value_box, value_box)
        cur_item.widget().close()
        del cur_item

        # Save new value_box
        self.value_box = value_box

    # This function creates the value box for integers
    def add_type_int(self):
        # Create a spinbox for integers
        int_box = QW.QSpinBox()
        int_box.setRange(-9999999, 9999999)
        int_box.setToolTip("Integer value for this entry type")
        int_box.valueChanged.connect(self.options.enable_save_button)
        return(int_box)

    # This function creates the value box for floats
    def add_type_float(self):
        # Create a spinbox for floats
        float_box = QW.QDoubleSpinBox()
        float_box.setRange(-9999999, 9999999)
        float_box.setToolTip("Float value for this entry type")
        float_box.valueChanged.connect(self.options.enable_save_button)
        return(float_box)

    # This function creates the value box for strings
    def add_type_str(self):
        # Create a lineedit for strings
        str_box = QW.QLineEdit()
        str_box.setToolTip("String value for this entry type")
        str_box.textEdited.connect(self.options.enable_save_button)
        return(str_box)

    # This function retrieves a value of this special box
    def get_box_value(self):
        return(get_box_value(self.value_box))

    # This function sets the value of this special box
    def set_box_value(self, value):
        set_box_value(self.type_box, self.type_dict[type(value)])
        set_box_value(self.value_box, value)


# %% SUPPORT FUNCTIONS
# This function gets the value of a provided widget_box
def get_box_value(widget_box):
    # Values (QAbstractSpinBox)
    if isinstance(widget_box, QW.QAbstractSpinBox):
        return(widget_box.value())
    # Bools (QAbstractButton)
    elif isinstance(widget_box, QW.QAbstractButton):
        return(widget_box.isChecked())
    # Items (QComboBox)
    elif isinstance(widget_box, QW.QComboBox):
        return(widget_box.currentText())
    # Strings (QLineEdit)
    elif isinstance(widget_box, QW.QLineEdit):
        return(widget_box.text())
    # Custom boxes (KwargsDictDialogTab, FigSizeBox, DefaultBox)
    elif isinstance(widget_box, (KwargsDictDialogTab, FigSizeBox, DefaultBox)):
        return(widget_box.get_box_value())

    # If none, raise error (such that I know to implement it)
    else:
        raise NotImplementedError


# This function sets the value of a provided widget_box
def set_box_value(widget_box, value):
    # Values (QAbstractSpinBox)
    if isinstance(widget_box, QW.QAbstractSpinBox):
        widget_box.setValue(value)
    # Bools (QAbstractButton)
    elif isinstance(widget_box, QW.QAbstractButton):
        widget_box.setChecked(value)
    # Items (QComboBox)
    elif isinstance(widget_box, QW.QComboBox):
        widget_box.setCurrentText(value)
    # Strings (QLineEdit)
    elif isinstance(widget_box, QW.QLineEdit):
        widget_box.setText(value)
    # Custom boxes (KwargsDictDialogTab, FigSizeBox, DefaultBox)
    elif isinstance(widget_box, (KwargsDictDialogTab, FigSizeBox, DefaultBox)):
        widget_box.set_box_value(value)

    # If none, raise error (such that I know to implement it)
    else:
        raise NotImplementedError


# %% FUNCTION DEFINITIONS GUI
def open_gui(pipeline_obj):
    # Wrap entire execution in switch_backend of MPL
    # TODO: Currently, this does not properly switch the backend back
    with switch_backend('Agg'):
        # Set some application attributes
        QW.QApplication.setAttribute(QC.Qt.AA_DontShowIconsInMenus, False)
        QW.QApplication.setAttribute(QC.Qt.AA_EnableHighDpiScaling, True)
        QW.QApplication.setAttribute(QC.Qt.AA_UseHighDpiPixmaps, True)

        # Initialize a new QApplication
        qapp = QW.QApplication([APP_NAME])

        # Set application icon
        qapp.setWindowIcon(QG.QIcon(path.join(DIR_PATH, 'data/app_icon.ico')))
        qapp.setApplicationName(APP_NAME)

        # Make sure that the application quits when the last window closes
        qapp.lastWindowClosed.connect(qapp.quit, QC.Qt.QueuedConnection)

        # Initialize main window and draw (show) it
        main_window = MainViewerWindow(qapp, pipeline_obj)
        main_window.show()

        # Replace the KeyboardInterrupt error by the system's default handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Start application
        qapp.exec_()


# %% Main program
# TODO: Remove this when development has been finished
if __name__ == '__main__':
    import os
    os.chdir("../../PRISM_Root")
    from prism._pipeline import Pipeline
    from prism.modellink import GaussianLink

    try:
        pipe
    except NameError:
        modellink_obj = GaussianLink(model_data='data/data_gaussian.txt')
        pipe = Pipeline(modellink_obj, root_dir='tests',
                        working_dir='projection_gui',
                        prism_par='data/prism_gaussian.txt')

        pipe.construct(1)

    open_gui(pipe)
