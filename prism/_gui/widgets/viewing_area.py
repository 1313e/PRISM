# -*- coding: utf-8 -*-

"""
GUI Projection Viewing Area
===========================
Provides the viewing area dock widget for the Projection GUI.

"""


# %% IMPORTS
# Built-in imports
from os import path
import sys

# Package imports
from PyQt5 import QtCore as QC, QtWidgets as QW
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._gui.widgets import QW_QAction, QW_QToolBar

# All declaration
__all__ = ['ViewingAreaDockWidget']


# %% CLASS DEFINITIONS
# Define class for the projection viewing area dock widget
# TODO: Allow for multiple viewing areas to co-exist?
class ViewingAreaDockWidget(QW.QDockWidget):
    def __init__(self, main_window_obj, *args, **kwargs):
        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.set_proj_attr = self.main.set_proj_attr
        self.all_set_proj_attr = self.main.all_set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr
        self.call_proj_attr = self.main.call_proj_attr
        self.all_call_proj_attr = self.main.all_call_proj_attr

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

        # Options for proj_area
        self.proj_area.setViewMode(QW.QMdiArea.SubWindowView)
        self.proj_area.setOption(QW.QMdiArea.DontMaximizeSubWindowOnActivation)
        self.proj_area.setActivationOrder(QW.QMdiArea.StackingOrder)
        self.proj_area.setStatusTip("Main projection viewing area")

        # Options for area_window
        self.area_window.setAttribute(QC.Qt.WA_DeleteOnClose)
        self.area_window.setContextMenuPolicy(QC.Qt.NoContextMenu)

        # Obtain dict of default docking positions
        self.default_pos = self.get_default_dock_positions()

        # Add toolbar to the projection viewer
        self.create_projection_toolbar()

    # This function saves the current state of the viewer to file
    # TODO: See if the window frames can be removed from the saved image
    @QC.pyqtSlot()
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
        self.area_window.close()

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
    @QC.pyqtSlot()
    def set_default_dock_positions(self):
        # Set the dock widgets and toolbars to their default positions
        # TOOLS TOOLBAR
        self.proj_toolbar.setVisible(True)
        self.area_window.addToolBar(self.default_pos['Tools'],
                                    self.proj_toolbar)

    # This function creates the toolbar of the projection viewing area
    def create_projection_toolbar(self):
        # Create toolbar for projection viewer
        self.proj_toolbar = QW_QToolBar(self, "Tools")

        # Create an action for enabling/disabling the toolbar
        proj_toolbar_act = self.proj_toolbar.toggleViewAction()
        proj_toolbar_act.setText("Tools toolbar")
        proj_toolbar_act.setStatusTip("Enable/disable the 'Tools' toolbar")
        self.main.toolbars_menu.addAction(proj_toolbar_act)

        # Add action for cascading all subwindows
        cascade_act = QW_QAction(
            self, "&Cascade",
            shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_C,
            statustip="Cascade all subwindows",
            triggered=self.proj_area.cascadeSubWindows)
        self.proj_toolbar.addAction(cascade_act)

        # Add action for tiling all subwindows
        tile_act = QW_QAction(
            self, "&Tile",
            shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_T,
            statustip="Tile all subwindows",
            triggered=self.proj_area.tileSubWindows)
        self.proj_toolbar.addAction(tile_act)

        # Add action for closing all subwindows
        close_act = QW_QAction(
            self, "Close all",
            shortcut=QC.Qt.CTRL + QC.Qt.SHIFT + QC.Qt.Key_X,
            statustip="Close all subwindows",
            triggered=self.proj_area.closeAllSubWindows)
        self.proj_toolbar.addAction(close_act)
