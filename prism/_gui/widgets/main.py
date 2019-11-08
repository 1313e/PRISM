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
import sys
from textwrap import dedent
import warnings

# Package imports
from e13tools.utils import docstring_substitute
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW

# PRISM imports
from prism import __version__
from prism._docstrings import kwargs_doc, qt_slot_doc
from prism._internal import RequestError, RequestWarning
from prism._gui import APP_ICON_PATH, APP_NAME
from prism._gui.widgets import QW_QAction, QW_QMenu, show_exception_details
from prism._gui.widgets.overview import OverviewDockWidget
from prism._gui.widgets.preferences import OptionsDialog
from prism._gui.widgets.viewing_area import ViewingAreaDockWidget

# All declaration
__all__ = ['MainViewerWindow']


# %% CLASS DEFINITIONS
# Define class for main viewer window
# TODO: Refactor entire GUI to account for functionalities ported to GuiPy
class MainViewerWindow(QW.QMainWindow):
    """
    Defines the :class:`~MainViewerWindow` class for the Projection GUI.

    This class provides the main window for the GUI and combines all other
    widgets; layouts; and elements together.

    """

    # Create signal for exception that are raised
    exception = QC.pyqtSignal()

    # Initialize MainViewerWindow class
    @docstring_substitute(optional=kwargs_doc.format(
        'PyQt5.QtWidgets.QMainWindow'))
    def __init__(self, pipeline_obj, *args, **kwargs):
        """
        Initialize an instance of the :class:`~MainViewerWindow` class.

        Parameters
        ----------
        pipeline_obj : :obj:`~prism.Pipeline` object
            Instance of the :class:`~prism.Pipeline` class for which the GUI
            needs to be initialized.

        %(optional)s

        """

        # Save pipeline_obj as pipe
        self.pipe = pipeline_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Set up the main window
        self.init()

    # This function sets up the main window
    def init(self):
        """
        Sets up the main window after it has been initialized.

        This function is mainly responsible for initializing all other widgets
        that are required to make the GUI work, and connecting them together.

        """

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
        self.setWindowIcon(QG.QIcon(APP_ICON_PATH))

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
        self.menubar.setFocus()

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
        """
        Creates the top-level menubar of the main window.

        Other widgets can modify this menubar to add additional actions to it.

        """

        # Obtain menubar
        self.menubar = self.menuBar()

        # FILE
        # Create file menu
        file_menu = self.menubar.addMenu('&File')

        # Add save action to file menu
        save_act = QW_QAction(
            self, '&Save view as...',
            shortcut=QG.QKeySequence.Save,
            statustip="Save current projection viewing area as an image",
            triggered=self.area_dock.save_view,
            role=QW_QAction.ApplicationSpecificRole)
        file_menu.addAction(save_act)

        # Add quit action to file menu
        quit_act = QW_QAction(
            self, '&Quit',
            shortcut=QG.QKeySequence.Quit,
            statustip="Quit %s" % (APP_NAME),
            triggered=self.close,
            role=QW_QAction.QuitRole)
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
            triggered=self.options,
            role=QW_QAction.PreferencesRole)
        help_menu.addAction(options_act)

        # Add details action to help menu
        details_act = QW_QAction(
            self, '&Details',
            shortcut=QC.Qt.CTRL + QC.Qt.Key_D,
            statustip=("Show the pipeline details overview of a specified "
                       "iteration"),
            triggered=self.show_pipeline_details_overview,
            role=QW_QAction.ApplicationSpecificRole)
        help_menu.addAction(details_act)

        # Add a separator
        help_menu.addSeparator()

        # Add API reference action to help menu
        apiref_act = QW_QAction(
            self, 'API reference',
            statustip="Open %s's API reference in a webbrowser" % (APP_NAME),
            triggered=self.api_reference,
            role=QW_QAction.ApplicationSpecificRole)
        help_menu.addAction(apiref_act)

        # Add a separator
        help_menu.addSeparator()

        # Add about action to help menu
        about_act = QW_QAction(
            self, '&About...',
            statustip="About %s" % (APP_NAME),
            triggered=self.about,
            role=QW_QAction.AboutRole)
        help_menu.addAction(about_act)

        # Add aboutQt action to help menu
        aboutqt_act = QW_QAction(
            self, 'About &Qt...',
            statustip="About Qt framework",
            triggered=QW.QApplication.aboutQt,
            role=QW_QAction.AboutQtRole)
        help_menu.addAction(aboutqt_act)

    # This function creates the statusbar in the viewer
    def create_statusbar(self):
        """
        Creates the bottom-level statusbar of the main window, primarily used
        for displaying extended descriptions of actions.

        """

        # Obtain statusbar
        self.statusbar = self.statusBar()

    # This function creates a message box with the 'about' information
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def about(self):
        """
        Displays a small section with information about the GUI.

        %(qt_slot)s

        """

        # Make shortcuts for certain links
        github_repo = "https://github.com/1313e/PRISM"

        # Create the text for the 'about' dialog
        text = dedent(r"""
            <b>{name} | <a href="{github}">PRISM</a> v{version}</b><br>
            Copyright &copy; 2019 Ellert van der Velden<br>
            Distributed under the
            <a href="{github}/raw/master/LICENSE">BSD-3 License</a>.
            """.format(name=APP_NAME,
                       version=__version__,
                       github=github_repo))

        # Create the 'about' dialog
        QW.QMessageBox.about(self, "About %s" % (APP_NAME), text)

    # This function opens the RTD API reference documentation in a webbrowser
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def api_reference(self):
        """
        Opens the API reference documentation of the GUI in a webbrowser.

        %(qt_slot)s

        """

        # Open webbrowser
        QG.QDesktopServices.openUrl(QC.QUrl(
            "https://prism-tool.readthedocs.io/en/latest/api/prism._gui.html"))

    # This function is called when the viewer is closed
    def closeEvent(self, *args, **kwargs):
        """
        Special :meth:`~PyQt5.QtWidgets.QWidget.closeEvent` event that
        automatically performs some clean-up operations before the main window
        closes.

        """

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
        """
        Sets the requested :class:`~prism._projection.Projection` attribute
        `name` to `value` on the controller rank.

        """

        # Set the attribute
        setattr(self.pipe, '_Projection__%s' % (name), value)

    # This function is an MPI-version of set_proj_attr
    def all_set_proj_attr(self, name, value):
        """
        Sets the requested :class:`~prism._projection.Projection` attribute
        `name` to `value` on all ranks.

        """

        # Set the attribute on all ranks
        self.pipe._make_call('__setattr__', '_Projection__%s' % (name), value)

    # This function allows for projection attributes to be read more easily
    def get_proj_attr(self, name):
        """
        Gets the value of the requested :class:`~prism._projection.Projection`
        attribute `name` on the controller rank.

        """

        # Retrieve the attribute
        return(getattr(self.pipe, '_Projection__%s' % (name)))

    # This function allows for projection attributes to be called more easily
    def call_proj_attr(self, name, *args, **kwargs):
        """
        Calls the requested :class:`~prism._projection.Projection` attribute
        `name` using the provided `args` and `kwargs` on the controller rank.

        """

        # Call the attribute
        return(getattr(self.pipe, '_Projection__%s' % (name))(*args, **kwargs))

    # This function is an MPI-version of call_proj_attr
    def all_call_proj_attr(self, name, *args, **kwargs):
        """
        Calls the requested :class:`~prism._projection.Projection` attribute
        `name` using the provided `args` and `kwargs` on all ranks.

        """

        # Call the attribute on all ranks
        return(self.pipe._make_call('_Projection__%s' % (name),
                                    *args, **kwargs))

    # This function returns the default positions of dock widgets and toolbars
    def get_default_dock_positions(self):
        """
        Returns the default positions of all dock widgets connected to the main
        window.

        """

        # Make dict including the default docking positions
        default_pos = {
            'Viewing area': QC.Qt.RightDockWidgetArea,
            'Overview': QC.Qt.LeftDockWidgetArea}

        # Return it
        return(default_pos)

    # This function sets dock widgets and toolbars to their default position
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def set_default_dock_positions(self):
        """
        Sets the positions of all dock widgets connected to the main window to
        their default positions.

        %(qt_slot)s

        """

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
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def show_pipeline_details_overview(self):
        """
        Creates and shows a dialog containing the output of the
        :meth:`~prism.Pipeline.details` method for all emulator iterations.

        %(qt_slot)s

        """

        # Make a details dialog
        details_box = QW.QDialog(self)
        details_box.setWindowModality(QC.Qt.NonModal)
        details_box.setAttribute(QC.Qt.WA_DeleteOnClose)
        details_box.setWindowFlags(
            QC.Qt.MSWindowsOwnDC |
            QC.Qt.Window |
            QC.Qt.WindowTitleHint |
            QC.Qt.WindowSystemMenuHint |
            QC.Qt.WindowCloseButtonHint)
        details_box.setWindowTitle("%s: Pipeline details" % (APP_NAME))

        # Create a layout for this dialog
        layout = QW.QVBoxLayout(details_box)

        # Obtain the latest emul_i
        emul_i = self.pipe._emulator._emul_i

        # Create a details tab widget
        tab_widget = QW.QTabWidget()
        tab_widget.setCornerWidget(QW.QLabel("Emulator iteration:"),
                                   QC.Qt.TopLeftCorner)
        tab_widget.setUsesScrollButtons(True)
        tab_widget.setStyleSheet(
            """
            QTabWidget::pane {
                padding: 1px;}
            QTabWidget::tab-bar {
                left: 10px;}
            """)
        layout.addWidget(tab_widget)

        # Loop over all emulator iterations and add their pages
        for i in range(1, emul_i+1):
            # Initialize a StringIO stream to capture the output with
            with StringIO() as string_stream:
                # Use this stream to capture the overview of details()
                with redirect_stdout(string_stream):
                    # Obtain the details at specified emulator iteration
                    self.pipe._make_call('details', i)

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
            details_tab = QW.QWidget(details_box)
            details_tab.setLayout(details_layout)

            # Add the tab to a scrollarea
            scrollarea = QW.QScrollArea(details_box)
            scrollarea.setFrameStyle(QW.QFrame.NoFrame)
            scrollarea.setContentsMargins(0, 0, 0, 0)
            scrollarea.setWidgetResizable(True)
            scrollarea.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
            scrollarea.horizontalScrollBar().setEnabled(False)
            scrollarea.setWidget(details_tab)

            # Set size constraints on the scrollarea
            scrollarea.setMaximumHeight(details_tab.height() + 2)

            # Add it to the tab_widget
            tab_widget.addTab(scrollarea, "&%i" % (i))

        # Set size constraints on the details box
        details_box.setFixedWidth(tab_widget.sizeHint().width() +
                                  scrollarea.verticalScrollBar().width() +
                                  layout.contentsMargins().left() +
                                  layout.contentsMargins().right())
        details_box.setMaximumHeight(scrollarea.maximumHeight() +
                                     tab_widget.tabBar().sizeHint().height() +
                                     layout.contentsMargins().top() +
                                     layout.contentsMargins().bottom())

        # Show the details message box
        details_box.show()
