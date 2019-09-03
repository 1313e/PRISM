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
from textwrap import dedent

# Package imports
from matplotlib.backends.qt_compat import (
    QtCore as QC, QtGui as QG, QtWidgets as QW, _getSaveFileName)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
import matplotlib.pyplot as plt
import numpy as np
from pytest_mpl.plugin import switch_backend
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism.__version__ import __version__

# All declaration
__all__ = ['open_gui']


# %% GLOBALS
APP_NAME = "PRISM Projection Viewer"        # Name of application
DIR_PATH = path.dirname(__file__)           # Path to directory of this file


# %% CLASS DEFINITIONS GUI
# Define class for main viewer window
# TODO: This cannot run from IPython console on Windows
# However, on some it apparently does work properly
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
        self.pipe._Projection__prepare_projections(None, None)

        # Save some statistics about pipeline and modellink
        self.n_par = self.pipe._modellink._n_par

        # Make sure that the viewer is deleted when window is closed
        self.setAttribute(QC.Qt.WA_DeleteOnClose)

        # Create statusbar
        self.create_statusbar()

        # Create main projection viewing area
        self.proj_viewer = ProjectionViewer(self)
        self.proj_viewer.setFocus()
        self.setCentralWidget(self.proj_viewer)

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
        save_act = QW_QAction('&Save view', self)
        save_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.Key_S,
            statustip="Save current projection viewing area as an image")
        save_act.triggered.connect(self.proj_viewer.save_view)
        file_menu.addAction(save_act)

        # Add quit action to file menu
        quit_act = QW_QAction('&Quit', self)
        quit_act.setDetails(
            shortcut=QC.Qt.CTRL + QC.Qt.Key_Q,
            statustip="Quit viewer")
        quit_act.triggered.connect(self.qapp.closeAllWindows)
        file_menu.addAction(quit_act)

        # VIEW
        # Create view menu, which includes all actions in the proj_toolbar
        view_menu = self.menubar.addMenu('&View')
        view_menu.addActions(self.proj_viewer.proj_toolbar.actions())

        # OPTIONS
        # Create options menu
        options_menu = self.menubar.addMenu('&Options')

        # Add settings action to options menu
        self.settings = SettingsWindow(self)
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
        QW.QMessageBox.about(self, "About %s" % (APP_NAME), dedent(r"""
            <b>PRISM v%s | %s</b><br>
            Copyright (C) 2019 Ellert van der Velden
            """ % (__version__, APP_NAME)))

    # This function is called when the viewer is closed
    def closeEvent(self, *args, **kwargs):
        # Call the closeEvent of ProjectionViewer
        self.proj_viewer.closeEvent(*args, **kwargs)

        # Save that Projection GUI is no longer being used
        self.set_proj_attr('use_GUI', 0)

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


# Define class for main projection viewing area
# TODO: Allow Figure instance to be saved from the GUI as if made by project()
# Look at matplotlib/backends/backend_qt5/SaveFigureQT() on how to do this
class ProjectionViewer(QW.QMainWindow):
    def __init__(self, main_window_obj, *args, **kwargs):
        # Call super constructor
        super().__init__(*args, **kwargs)

        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.set_proj_attr = self.main.set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr

        # Make sure that the viewer is deleted when window is closed
        self.setAttribute(QC.Qt.WA_DeleteOnClose)

        # Create the projection viewing area
        self.create_projection_area()
        self.create_projection_overview()

    # This function saves the current state of the viewer to file
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

        # Open the file saving system
        filename, _ = _getSaveFileName(
            parent=self.main,
            caption="Save view as...",
            directory=path.join(self.pipe._working_dir, "proj_area.png"),
            filter=file_filters,
            initialFilter=default_filter)

        # If filename was provided, save image
        if(filename != ''):
            # Grab the current state of the projection area as a Pixmap
            pixmap = self.proj_area.grab()

            # Save pixmap with chosen filename
            pixmap.save(filename)

    # This function is called when the main window is closed
    def closeEvent(self, *args, **kwargs):
        # Close all currently opened figures
        for fig in self.proj_fig_registry.values():
            plt.close(fig)

        # Close the projection viewer
        super().closeEvent(*args, **kwargs)

    # This function creates the main projection viewing area
    def create_projection_area(self):
        # Create a MdiArea for the viewer
        self.proj_area = QW.QMdiArea(self)
        self.proj_area.setViewMode(0)                   # Use subwindow mode
        self.proj_area.setStatusTip("Main projection viewing area")
        self.setCentralWidget(self.proj_area)

        # Create empty dict containing all projection figure instances
        self.proj_fig_registry = {}

        # Add toolbar to the projection viewer
        self.create_projection_toolbar()

    # This function creates the toolbar of the projection viewing area
    def create_projection_toolbar(self):
        # Create toolbar for projection viewer
        self.proj_toolbar = QW.QToolBar("Tools", self)
        self.addToolBar(self.proj_toolbar)

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

    # This function creates the projection list overview
    # TODO: Should this be a QToolBar or a QDockWidget?
    def create_projection_overview(self):
        # Create an overview (QToolBar)
        self.proj_overview = QW.QToolBar("Projections", self.main)
        self.proj_overview.setAllowedAreas(
            QC.Qt.LeftToolBarArea | QC.Qt.RightToolBarArea)
        self.proj_overview.setFloatable(False)
        self.main.addToolBar(QC.Qt.LeftToolBarArea, self.proj_overview)

#        # Create an overview (QDockWidget)
#        self.proj_dock = QW.QDockWidget("Projections", self.main)
#        self.proj_dock.setAllowedAreas(
#            QC.Qt.LeftDockWidgetArea | QC.Qt.RightDockWidgetArea)
#        self.proj_dock.setFeatures(
#            QW.QDockWidget.DockWidgetMovable)
#        self.main.addDockWidget(QC.Qt.LeftDockWidgetArea, self.proj_dock)
#        proj_widget = QW.QWidget()
#        self.proj_dock.setWidget(proj_widget)
#        self.proj_overview = QW.QVBoxLayout()
#        proj_widget.setLayout(self.proj_overview)

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

        # Add list for drawn projections
        self.proj_overview.addWidget(QW.QLabel("Drawn:"))
        self.proj_list_d = QW.QListWidget()
        self.proj_list_d.addItems(drawn_hcubes)
        self.proj_list_d.setStatusTip("Lists all projections that have been "
                                      "drawn")
        self.proj_list_d.setAlternatingRowColors(True)
        self.proj_list_d.setSortingEnabled(True)
        self.proj_list_d.itemDoubleClicked.connect(self.add_projection_figure)
        self.proj_overview.addWidget(self.proj_list_d)

        # Add list for available projections
        self.proj_overview.addWidget(QW.QLabel("Available:"))
        self.proj_list_a = QW.QListWidget()
        self.proj_list_a.addItems(avail_hcubes)
        self.proj_list_a.setStatusTip("Lists all projections that have been "
                                      "calculated but not drawn")
        self.proj_list_a.setAlternatingRowColors(True)
        self.proj_list_a.setSortingEnabled(True)
        self.proj_list_a.itemDoubleClicked.connect(self.add_projection_figure)
        self.proj_overview.addWidget(self.proj_list_a)

        # Add list for projections that can be created
        self.proj_overview.addWidget(QW.QLabel("Unavailable:"))
        self.proj_list_u = QW.QListWidget()
        self.proj_list_u.addItems(unavail_hcubes)
        self.proj_list_u.setStatusTip("Lists all projections that have not "
                                      "been calculated")
        self.proj_list_u.setAlternatingRowColors(True)
        self.proj_list_u.setSortingEnabled(True)
        self.proj_list_u.itemDoubleClicked.connect(
            self.create_projection_figure)
        self.proj_overview.addWidget(self.proj_list_u)

    # This function adds a projection figure to the viewing area
    # OPTIMIZE: (Re)Drawing a 3D projection figure takes up to 15 seconds
    def add_projection_figure(self, list_item):
        # Retrieve text of list_item
        hcube_name = list_item.text()
        hcube = self.hcubes[self.names.index(hcube_name)]

        # Check if this figure is already stored in memory
        if hcube_name in self.proj_fig_registry.keys():
            # Obtain the corresponding figure
            fig = self.proj_fig_registry[hcube_name]

        # If not, create it
        else:
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
            item = self.proj_list_a.takeItem(self.proj_list_a.currentRow())
            self.proj_list_d.addItem(item)

        # Create a FigureCanvas instance
        figure_canvas = FigureCanvas(fig)

        # Create a new subwindow
        subwindow = QW.QMdiSubWindow()
        subwindow.setWindowTitle(hcube_name)
        subwindow.setWidget(figure_canvas)

        # Add new subwindow to viewing area
        self.proj_area.addSubWindow(subwindow)
        subwindow.show()

    # This function removes a projection figure permanently from the register
    def remove_projection_figure(self, list_item):
        # Retrieve text of list_item
        hcube_name = list_item.text()

        # Check if this figure is already stored in memory
        if hcube_name in self.proj_fig_registry.keys():
            # Pop the figure from the registry
            fig = self.proj_fig_registry.pop(hcube_name)

            # Close the figure
            plt.close(fig)

            # Move figure from drawn to available
            item = self.proj_list_d.takeItem(self.proj_list_d.currentRow())
            self.proj_list_a.addItem(item)

    # This function creates a projection figure and adds it to the viewing area
    def create_projection_figure(self, list_item):
        # Retrieve text of list_item
        hcube_name = list_item.text()
        hcube = self.hcubes[self.names.index(hcube_name)]

        # Calculate projection data
        _, _ = self.get_proj_attr('analyze_proj_hcube')(hcube)

        # Move figure from unavailable to available
        item = self.proj_list_u.takeItem(self.proj_list_u.currentRow())
        self.proj_list_a.addItem(item)
        self.proj_list_a.setCurrentItem(item)

        # Add the figure to the viewing area
        self.add_projection_figure(item)


# Define class for settings window
class SettingsWindow(object):
    def __init__(self, main_window_obj):
        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.n_par = self.main.n_par
        self.set_proj_attr = self.main.set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr

        # Save projection defaults here
        self.defaults = sdict(self.pipe._Projection__proj_kwargs)

    # This function creates a settings window
    def __call__(self):
        # Create settings window
        self.window = QW.QDialog(self.main)

        # Create settings layout
        self.settings_layout = QW.QFormLayout(self.window)

        # Initialize empty dict of setting boxes
        self.settings_dict = sdict()

        # Define list with all options that should be available in what order
        setting_items = ['proj_type', 'align', 'show_cuts', 'smooth',
                         'buttons']

        # Include all options named in setting_items
        for item in setting_items:
            getattr(self, 'add_option_%s' % (item))()

        # Set a few properties of settings window
        self.window.setGeometry(0, 0, 0, 0)                    # Resolution
        self.window.setWindowModality(QC.Qt.ApplicationModal)  # Modality
        self.window.setWindowTitle("Preferences")              # Title

        # Show it
        self.window.show()

    # EMUL_I
    def add_option_emul_i(self):
        # Make spinbox for setting emul_i
        emul_i_box = QW.QSpinBox()
        self.settings_dict['emul_i'] = emul_i_box
        emul_i_box.setRange(0, self.pipe._emulator._emul_i)
        emul_i_box.setValue(self.get_proj_attr('emul_i'))
        emul_i_box.valueChanged.connect(lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('emul_i:', emul_i_box)

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
        self.settings_layout.addRow('proj_type:', proj_type_box)

    # FIGURE
    def add_option_figure(self):
        # Make check box for figure
        figure_box = QW.QCheckBox()
        self.settings_dict['figure'] = figure_box
        figure_box.setChecked(self.get_proj_attr('figure'))
        figure_box.stateChanged.connect(lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('figure:', figure_box)

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
        self.settings_layout.addRow('align:', align_box)

    # SHOW_CUTS
    def add_option_show_cuts(self):
        # Make check box for show_cuts
        show_cuts_box = QW.QCheckBox()
        self.settings_dict['show_cuts'] = show_cuts_box
        show_cuts_box.setChecked(self.get_proj_attr('show_cuts'))
        show_cuts_box.stateChanged.connect(
            lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('show_cuts:', show_cuts_box)

    # SMOOTH
    def add_option_smooth(self):
        # Make check box for smooth
        smooth_box = QW.QCheckBox()
        self.settings_dict['smooth'] = smooth_box
        smooth_box.setChecked(self.get_proj_attr('smooth'))
        smooth_box.stateChanged.connect(lambda: self.save_but.setEnabled(True))
        self.settings_layout.addRow('smooth:', smooth_box)

    # KWARGS
    def add_option_kwargs(self):
        # IMPL_KWARGS_2D
        self.create_kwargs_box_layout('impl_kwargs_2D')

    # BUTTONS
    def add_option_buttons(self):
        # Create a buttons layout
        buttons_layout = QW.QHBoxLayout()
        self.settings_layout.addRow(buttons_layout)

        # Make a 'Reset' button
        reset_but = QW.QPushButton("Reset")
        reset_but.clicked.connect(self.reset_settings)
        reset_but.clicked.connect(lambda: save_but.setEnabled(False))
        buttons_layout.addWidget(reset_but)

        # Make a 'Save' button
        save_but = QW.QPushButton("Save")
        save_but.clicked.connect(self.save_settings)
        save_but.clicked.connect(lambda: save_but.setEnabled(False))
        save_but.setEnabled(False)
        self.save_but = save_but
        buttons_layout.addWidget(save_but)

        # Make a 'Close' button
        close_but = QW.QPushButton("Close")
        close_but.setDefault(True)
        close_but.clicked.connect(self.window.close)
        buttons_layout.addWidget(close_but)

    # This function saves the new default settings
    def save_settings(self):
        # Save all new defaults
        for key, val in self.settings_dict.items():
            # Values (QSpinBox)
            if key in ['emul_i']:
                self.set_proj_attr(key, val.value())
            # Bools (QCheckBox/QRadioButton)
            elif key in ['proj_2D', 'proj_3D', 'figure', 'show_cuts',
                         'smooth']:
                self.set_proj_attr(key, int(val.isChecked()))
            # Align
            elif key in ['align_col', 'align_row']:
                if val.isChecked():
                    self.set_proj_attr('align', key[6:])
            # Items (QComboBox)
            elif key in []:
                self.set_proj_attr(key, val.currentText())

    # This function resets the default settings
    def reset_settings(self):
        # Reset all settings to defaults
        for key, val in self.settings_dict.items():
            # Values (QSpinBox)
            if key in ['emul_i']:
                val.setValue(self.defaults[key])
            # Bools (QCheckBox/QRadioButton)
            elif key in ['proj_2D', 'proj_3D', 'figure', 'show_cuts',
                         'smooth']:
                val.setChecked(self.defaults[key])
            # Align
            elif key in ['align_col', 'align_row']:
                val.setChecked(self.defaults['align'] == key[6:])
            # Items (QComboBox)
            elif key in []:
                val.setCurrentText(self.defaults[key])

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

    pipe.open_gui()
