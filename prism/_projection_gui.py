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
from os import path
import signal
import sys
from textwrap import dedent

# Package imports
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
from matplotlib.backend_bases import _default_filetypes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
import matplotlib.pyplot as plt
import numpy as np
from pytest_mpl.plugin import switch_backend
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism.__version__ import __version__
from prism._docstrings import proj_depth_doc, proj_res_doc

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

        # Prepare projections to be made
        self.get_proj_attr('prepare_projections')(None, None)

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

        # OPTIONS
        # Create options menu
        options_menu = self.menubar.addMenu('&Options')

        # Add settings action to options menu
        self.settings = SettingsDialog(self)
        options_act = QW_QAction('&Preferences', self)
        options_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.Key_P,
            statustip="Projection drawing preferences")
        options_act.triggered.connect(self.settings)
        options_menu.addAction(options_act)

        # HELP
        # Create help menu
        help_menu = self.menubar.addMenu('&Help')

        # Add about action to help menu
        about_act = QW_QAction('&About', self)
        about_act.setDetails(
            statustip="About %s" % (APP_NAME))
        about_act.triggered.connect(self.about)
        help_menu.addAction(about_act)

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
        self.settings.reset_settings()

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


# Define class for the projection viewing area dock widget
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

        # Settings for proj_area
        self.proj_area.setViewMode(0)                   # Use subwindow mode
        self.proj_area.setStatusTip("Main projection viewing area")

        # Settings for area_window
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

        # Add action for resetting the view
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
        # Close all currently opened figures
        for fig in self.proj_fig_registry.values():
            plt.close(fig)

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
    # TODO: Add action for showing the details/properties of a single figure
    def create_drawn_context_menu(self):
        # Create context menu
        menu = QW.QMenu('Drawn')

        # Make shortcut for obtaining selected items
        list_items = self.proj_list_d.selectedItems

        # Add show action to menu
        # TODO: Make sure only a single subwindow with a figure can exist
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
        # TODO: Make sure that closing a figure also closes its subwindow
        close_act = QW_QAction('&Close', self)
        close_act.setDetails(
            statustip="Close selected projection figure(s)")
        close_act.triggered.connect(
            lambda: self.close_projection_figures(list_items()))
        menu.addAction(close_act)

        # Save made menu as an attribute
        self.context_menu_d = menu

    # This function shows the context menu for drawn projections
    def show_drawn_context_menu(self):
        # If there is currently at least one item selected, show context menu
        if len(self.proj_list_d.selectedItems()):
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

        # Save made menu as an attribute
        self.context_menu_a = menu

    # This function shows the context menu for available projections
    def show_available_context_menu(self):
        # If there is currently at least one item selected, show context menu
        if len(self.proj_list_a.selectedItems()):
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

            # Obtain the corresponding figure
            fig = self.proj_fig_registry[hcube_name]

            # Create a FigureCanvas instance
            figure_canvas = FigureCanvas(fig)

            # Create a new subwindow
            subwindow = QW.QMdiSubWindow()
            subwindow.setWindowTitle(hcube_name)
            subwindow.setWidget(figure_canvas)

            # Add new subwindow to viewing area
            self.main.area_dock.proj_area.addSubWindow(subwindow)
            subwindow.show()

    # This function removes a projection figure permanently from the register
    def close_projection_figures(self, list_items):
        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()

            # Pop the figure from the registry
            fig = self.proj_fig_registry.pop(hcube_name)

            # Close the figure
            plt.close(fig)

            # Move figure from drawn to available
            item = self.proj_list_d.takeItem(
                self.proj_list_d.row(list_item))
            self.proj_list_a.addItem(item)

    # This function draws a projection figure
    # OPTIMIZE: (Re)Drawing a 3D projection figure takes up to 15 seconds
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
            self.proj_fig_registry[hcube_name] = fig

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
            fig = self.proj_fig_registry[hcube_name]

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


# Define class for settings dialog
class SettingsDialog(QW.QDialog):
    def __init__(self, main_window_obj, *args, **kwargs):
        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.n_par = self.main.n_par
        self.set_proj_attr = self.main.set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr

        # Call super constructor
        super().__init__(self.main, *args, **kwargs)

        # Create the settings window
        self.init()

    # This function creates the settings window
    def init(self):
        # Create a window layout
        self.window_layout = QW.QVBoxLayout(self)

        # Create settings layout
        self.settings_layout = QW.QFormLayout()
        self.window_layout.addLayout(self.settings_layout)

        # Save projection defaults here
        self.defaults = sdict(self.get_proj_attr('proj_kwargs'))

        # Initialize empty dict of setting boxes
        self.settings_dict = sdict()

        # Define list with all options that should be available in what order
        setting_items = ['proj_grid', 'proj_type', 'align', 'show_cuts',
                         'smooth', 'buttons']

        # Include all options named in setting_items
        for item in setting_items:
            getattr(self, 'add_option_%s' % (item))()

        # Set a few properties of settings window
        self.setGeometry(0, 0, 0, 0)                        # Resolution
        self.setModal(True)                                 # Modality
        self.setWindowTitle("Preferences")                  # Title

    # This function shows the settings window
    def __call__(self):
        # Move the settings window to the center of the main window
        self.move(self.main.geometry().center()-self.rect().center())

        # Show it
        self.show()

    # PROJ_PAR
    def add_option_proj_grid(self):
        # Make spinbox for setting proj_res
        proj_res_box = QW.QSpinBox()
        self.settings_dict['res'] = proj_res_box
        proj_res_box.setRange(0, 9999999)
        proj_res_box.setValue(self.get_proj_attr('res'))
        proj_res_box.setToolTip(proj_res_doc)
        proj_res_box.valueChanged.connect(
            lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('Resolution:', proj_res_box)

        # Make spinbox for setting proj_depth
        proj_depth_box = QW.QSpinBox()
        self.settings_dict['depth'] = proj_depth_box
        proj_depth_box.setRange(0, 9999999)
        proj_depth_box.setValue(self.get_proj_attr('depth'))
        proj_depth_box.setToolTip(proj_depth_doc)
        proj_depth_box.valueChanged.connect(
            lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('Depth:', proj_depth_box)

    # EMUL_I
    def add_option_emul_i(self):
        # Make spinbox for setting emul_i
        emul_i_box = QW.QSpinBox()
        self.settings_dict['emul_i'] = emul_i_box
        emul_i_box.setRange(0, self.pipe._emulator._emul_i)
        emul_i_box.setValue(self.get_proj_attr('emul_i'))
        emul_i_box.valueChanged.connect(lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('Iteration:', emul_i_box)

    # PROJ_TYPE
    def add_option_proj_type(self):
        # Make check boxes for 2D and 3D projections
        # 2D projections
        proj_2D_box = QW.QCheckBox('2D')
        self.settings_dict['proj_2D'] = proj_2D_box
        proj_2D_box.setChecked(self.get_proj_attr('proj_2D'))
        proj_2D_box.setEnabled(self.n_par > 2)
        proj_2D_box.stateChanged.connect(
            lambda: self.save_but.setEnabled(True))

        # 3D projections
        proj_3D_box = QW.QCheckBox('3D')
        self.settings_dict['proj_3D'] = proj_3D_box
        proj_3D_box.setChecked(self.get_proj_attr('proj_3D'))
        proj_3D_box.setEnabled(self.n_par > 2)
        proj_3D_box.stateChanged.connect(
            lambda: self.save_but.setEnabled(True))

        # Create layout for proj_type and add it to settings layout
        proj_type_box = QW.QHBoxLayout()
        proj_type_box.addWidget(proj_2D_box)
        proj_type_box.addWidget(proj_3D_box)
        proj_type_box.addStretch()
        self.settings_layout.addRow('Projection type:', proj_type_box)

    # ALIGN
    def add_option_align(self):
        # Make drop-down menu for align
        # Column align
        align_col_box = QW.QRadioButton('Column')
        self.settings_dict['align_col'] = align_col_box
        align_col_box.setChecked(self.get_proj_attr('align') == 'col')
        align_col_box.toggled.connect(lambda: self.save_but.setEnabled(True))

        # Row align
        align_row_box = QW.QRadioButton('Row')
        self.settings_dict['align_row'] = align_row_box
        align_row_box.setChecked(self.get_proj_attr('align') == 'row')
        align_row_box.toggled.connect(lambda: self.save_but.setEnabled(True))

        # Create layout for align and add it to settings layout
        align_box = QW.QHBoxLayout()
        align_box.addWidget(align_col_box)
        align_box.addWidget(align_row_box)
        align_box.addStretch()
        self.settings_layout.addRow('Alignment:', align_box)

    # SHOW_CUTS
    def add_option_show_cuts(self):
        # Make check box for show_cuts
        show_cuts_box = QW.QCheckBox()
        self.settings_dict['show_cuts'] = show_cuts_box
        show_cuts_box.setChecked(self.get_proj_attr('show_cuts'))
        show_cuts_box.stateChanged.connect(
            lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('Show cuts?', show_cuts_box)

    # SMOOTH
    def add_option_smooth(self):
        # Make check box for smooth
        smooth_box = QW.QCheckBox()
        self.settings_dict['smooth'] = smooth_box
        smooth_box.setChecked(self.get_proj_attr('smooth'))
        smooth_box.stateChanged.connect(lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('Smooth?', smooth_box)

    # KWARGS
    def add_option_kwargs(self):
        # IMPL_KWARGS_2D
        self.create_kwargs_box_layout('impl_kwargs_2D')

    # BUTTONS
    def add_option_buttons(self):
        # Create a buttons layout
        buttons_layout = QW.QHBoxLayout()
        self.window_layout.addLayout(buttons_layout)
        buttons_layout.addStretch()

        # Make a 'Reset' button
        reset_but = QW.QPushButton("&Reset")
        reset_but.setToolTip("Reset to defaults")
        reset_but.clicked.connect(self.reset_settings)
        reset_but.clicked.connect(lambda: save_but.setEnabled(False))
        buttons_layout.addWidget(reset_but)

        # Make a 'Save' button
        save_but = QW.QPushButton("&Save")
        save_but.setToolTip("Save settings")
        save_but.clicked.connect(self.save_settings)
        save_but.clicked.connect(lambda: save_but.setEnabled(False))
        save_but.setEnabled(False)
        self.save_but = save_but
        buttons_layout.addWidget(save_but)

        # Make a 'Close' button
        close_but = QW.QPushButton("&Close")
        close_but.setToolTip("Close without saving")
        close_but.setDefault(True)
        close_but.clicked.connect(self.close)
        buttons_layout.addWidget(close_but)

    # This function saves the new default settings
    def save_settings(self):
        # Save all new defaults
        for key, box in self.settings_dict.items():
            # Values (QSpinBox)
            if isinstance(box, QW.QSpinBox):
                self.set_proj_attr(key, box.value())
            # Align (special QRadioButton)
            elif key in ['align_col', 'align_row']:
                if box.isChecked():
                    self.set_proj_attr('align', key[6:])
            # Bools (QCheckBox/QRadioButton)
            elif isinstance(box, (QW.QCheckBox, QW.QRadioButton)):
                self.set_proj_attr(key, int(box.isChecked()))
            # Items (QComboBox)
            elif isinstance(box, QW.QComboBox):
                self.set_proj_attr(key, box.currentText())

    # This function resets the default settings
    def reset_settings(self):
        # Reset all settings to defaults
        for key, box in self.settings_dict.items():
            # Values (QSpinBox)
            if isinstance(box, QW.QSpinBox):
                box.setValue(self.defaults[key])
            # Align (special QRadioButton)
            elif key in ['align_col', 'align_row']:
                box.setChecked(self.defaults['align'] == key[6:])
            # Bools (QCheckBox/QRadioButton)
            elif isinstance(box, (QW.QCheckBox, QW.QRadioButton)):
                box.setChecked(self.defaults[key])
            # Items (QComboBox)
            elif isinstance(box, QW.QComboBox):
                box.setCurrentText(self.defaults[key])

        # Save current settings
        self.save_settings()

    # This function creates an editable list of input boxes for the kwargs
    def create_kwargs_box_layout(self, name):
        # Create a kwargs layout
        kwargs_box = QW.QVBoxLayout()
        edit_button = QW.QPushButton('Edit')
        edit_button.clicked.connect(
            lambda: self.add_editable_entry(kwargs_box))
        kwargs_box.addWidget(edit_button)
        self.settings_layout.addRow(name, kwargs_box)

    # This function adds an editable entry to a given box
    def add_editable_entry(self, box):
        # Make a new editable line
        line1 = QW.QLineEdit()
        line2 = QW.QLineEdit()
        line_box = QW.QHBoxLayout()
        line_box.addWidget(line1)
        line_box.addWidget(line2)
        box.addLayout(line_box)


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


# %% FUNCTION DEFINITIONS GUI
def open_gui(pipeline_obj):
    # Wrap entire execution in switch_backend of MPL
    # TODO: Currently, this does not properly switch the backend back
    with switch_backend('Agg'):
        # Initialize a new QApplication
        qapp = QW.QApplication([APP_NAME])

        # Set application icon
        qapp.setWindowIcon(QG.QIcon(path.join(DIR_PATH, 'data/app_icon.ico')))
        qapp.setApplicationName(APP_NAME)

        # Make sure that the application quits when the last window closes
        qapp.lastWindowClosed.connect(qapp.quit, QC.Qt.QueuedConnection)

        # Set some application attributes
        qapp.setAttribute(QC.Qt.AA_DontShowIconsInMenus, False)
        qapp.setAttribute(QC.Qt.AA_EnableHighDpiScaling, True)
        qapp.setAttribute(QC.Qt.AA_UseHighDpiPixmaps, True)

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
