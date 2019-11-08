# -*- coding: utf-8 -*-

"""
GUI Projection Overview
=======================
Provides the overview dock widget for the Projection GUI.

"""


# %% IMPORTS
# Built-in imports
import os
from os import path
from sys import platform
import time

# Package imports
from e13tools.utils import docstring_substitute
import matplotlib.pyplot as plt
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._docstrings import (
    kwargs_doc, list_items_optional_doc, list_item_optional_doc, qt_slot_doc)
from prism._gui import APP_NAME
from prism._gui.widgets import (
    FigureCanvas, QW_QAction, QW_QMenu, OverviewListWidget,
    ThreadedProgressDialog)

# All declaration
__all__ = ['OverviewDockWidget']


# %% CLASS DEFINITIONS
# Create class for the projection overview dock widget
class OverviewDockWidget(QW.QDockWidget):
    """
    Defines the :class:`~OverviewDockWidget` class for the Projection GUI.

    This class provides the user with the ability to quickly create; draw;
    view; and save projection figures.

    """

    # TODO: Allow for the lists to be sorted differently?
    @docstring_substitute(optional=kwargs_doc.format(
        'PyQt5.QtWidgets.QDockWidget'))
    def __init__(self, main_window_obj, *args, **kwargs):
        """
        Initialize an instance of the :class:`~OverviewDockWidget` class.

        Parameters
        ----------
        main_window_obj : :obj:`~prism._gui.widgets.MainViewerWindow` object
            Instance of the :class:`~prism._gui.widgets.MainViewerWindow` class
            that acts as the parent of this dock widget.

        %(optional)s

        """

        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.set_proj_attr = self.main.set_proj_attr
        self.all_set_proj_attr = self.main.all_set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr
        self.call_proj_attr = self.main.call_proj_attr
        self.all_call_proj_attr = self.main.all_call_proj_attr

        # Call the super constructor
        super().__init__("Overview", self.main, *args, **kwargs)

        # Create the overview widget
        self.init()

    # This function is called when the main window is closed
    def closeEvent(self, *args, **kwargs):
        """
        Special :meth:`~PyQt5.QtWidgets.QWidget.closeEvent` event that
        automatically performs some clean-up operations before the overview
        dock widget closes.

        """

        # Close all currently opened figures and subwindows
        for fig, subwindow in self.proj_fig_registry.values():
            plt.close(fig)
            subwindow.close()

        # Close the projection overview
        super().closeEvent(*args, **kwargs)

    # This function creates the projection overview
    def init(self):
        """
        Sets up the projection overview dock widget after it has been
        initialized.

        This function is mainly responsible for creating the different overview
        lists and menus that allow the user to manipulate projection figures.

        """

        # Create an overview
        overview_widget = QW.QWidget()
        self.proj_overview = QW.QVBoxLayout()
        overview_widget.setLayout(self.proj_overview)
        self.setWidget(overview_widget)

        # Set the contents margins at the bottom to zero
        contents_margins = self.proj_overview.getContentsMargins()
        self.proj_overview.setContentsMargins(*contents_margins[:3], 0)

        # Create empty dict containing all projection figure instances
        self.proj_fig_registry = {}

        # Make lists of all hcubes and their names
        self.hcubes = list(self.get_proj_attr('hcubes'))
        self.names = [self.call_proj_attr('get_hcube_name', hcube)
                      for hcube in self.hcubes]

        # Divide all hcubes up into three different lists
        # Drawn; available; unavailable
        unavail_hcubes = [self.call_proj_attr('get_hcube_name', hcube)
                          for hcube in self.get_proj_attr('create_hcubes')]
        avail_hcubes = [name for name in self.names
                        if name not in unavail_hcubes]
        drawn_hcubes = []

        # DRAWN PROJECTIONS
        # Add list for drawn projections
        self.proj_list_d = OverviewListWidget(
            hcubes_list=drawn_hcubes,
            status_tip="Lists all projections that have been drawn",
            context_menu=self.show_drawn_context_menu,
            activated=self.show_projection_figures)
        self.proj_overview.addWidget(QW.QLabel("Drawn:"))
        self.proj_overview.addWidget(self.proj_list_d)
        self.create_drawn_context_menu()

        # AVAILABLE PROJECTIONS
        # Add list for available projections
        self.proj_list_a = OverviewListWidget(
            hcubes_list=avail_hcubes,
            status_tip=("Lists all projections that have been calculated but "
                        "not drawn"),
            context_menu=self.show_available_context_menu,
            activated=self.draw_projection_figures)
        self.proj_overview.addWidget(QW.QLabel("Available:"))
        self.proj_overview.addWidget(self.proj_list_a)
        self.create_available_context_menu()

        # UNAVAILABLE PROJECTIONS
        # Add list for projections that can be created
        self.proj_list_u = OverviewListWidget(
            hcubes_list=unavail_hcubes,
            status_tip="Lists all projections that have not been calculated",
            context_menu=self.show_unavailable_context_menu,
            activated=self.create_projection_figures)
        self.proj_overview.addWidget(QW.QLabel("Unavailable:"))
        self.proj_overview.addWidget(self.proj_list_u)
        self.create_unavailable_context_menu()

    # This function creates the context menu for drawn projections
    def create_drawn_context_menu(self):
        """
        Creates the context (right-click) menu for the 'Drawn' overview list.

        This menu contains all actions that are available for drawn projection
        figures.

        """

        # Create context menu
        menu = QW_QMenu(self, 'Drawn')

        # Add show action to menu
        show_act = QW_QAction(
            self, 'S&how',
            statustip="Show selected projection figure(s)",
            triggered=self.show_projection_figures)
        menu.addAction(show_act)

        # Add save action to menu
        save_act = QW_QAction(
            self, '&Save',
            statustip="Save selected projection figure(s) to file",
            triggered=self.save_projection_figures)
        menu.addAction(save_act)

        # Add save as action to menu
        save_as_act = QW_QAction(
            self, 'Save &as...',
            statustip="Save selected projection figure(s) to chosen file",
            triggered=self.save_as_projection_figures)
        menu.addAction(save_as_act)

        # Add redraw action to menu
        redraw_act = QW_QAction(
            self, '&Redraw',
            statustip="Redraw selected projection figure(s)",
            triggered=self.redraw_projection_figures)
        menu.addAction(redraw_act)

        # Add close action to menu
        close_act = QW_QAction(
            self, '&Close',
            statustip="Close selected projection figure(s)",
            triggered=self.close_projection_figures)
        menu.addAction(close_act)

        # Add details action to menu (single item only)
        self.details_u_act = QW_QAction(
            self, 'De&tails',
            statustip="Show details about selected projection figure",
            triggered=self.details_drawn_projection_figure)
        menu.addAction(self.details_u_act)

        # Save made menu as an attribute
        self.context_menu_d = menu

    # This function shows the context menu for drawn projections
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def show_drawn_context_menu(self):
        """
        Shows the 'Drawn' context menu, giving the user access to its actions.

        %(qt_slot)s

        """

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
        """
        Creates the context (right-click) menu for the 'Available' overview
        list.

        This menu contains all actions that are available for created (but not
        drawn) projection figures.

        """

        # Create context menu
        menu = QW_QMenu(self, 'Available')

        # Add draw action to menu
        draw_act = QW_QAction(
            self, '&Draw',
            statustip="Draw selected projection figure(s)",
            triggered=self.draw_projection_figures)
        menu.addAction(draw_act)

        # Add draw&save action to menu
        draw_save_act = QW_QAction(
            self, 'Draw && &Save',
            statustip="Draw & save selected projection figure(s)",
            triggered=self.draw_save_projection_figures)
        menu.addAction(draw_save_act)

        # Add recreate action to menu
        recreate_act = QW_QAction(
            self, '&Recreate',
            statustip="Recreate selected projection figure(s)",
            triggered=self.recreate_projection_figures)
        menu.addAction(recreate_act)

        # Add delete action to menu
        delete_act = QW_QAction(
            self, 'D&elete',
            statustip="Delete selected projection figure(s)",
            triggered=self.delete_projection_figures)
        menu.addAction(delete_act)

        # Add details action to menu (single item only)
        self.details_a_act = QW_QAction(
            self, 'De&tails',
            statustip="Show details about selected projection figure",
            triggered=self.details_available_projection_figure)
        menu.addAction(self.details_a_act)

        # Save made menu as an attribute
        self.context_menu_a = menu

    # This function shows the context menu for available projections
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def show_available_context_menu(self):
        """
        Shows the 'Available' context menu, giving the user access to its
        actions.

        %(qt_slot)s

        """

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
        """
        Creates the context (right-click) menu for the 'Unavailable' overview
        list.

        This menu contains all actions that are available for non-existing
        projection figures.

        """

        # Create context menu
        menu = QW_QMenu(self, 'Unavailable')

        # Add create action to menu
        create_act = QW_QAction(
            self, '&Create',
            statustip="Create selected projection figure(s)",
            triggered=self.create_projection_figures)
        menu.addAction(create_act)

        # Add create&draw action to menu
        create_draw_act = QW_QAction(
            self, 'Create && &Draw',
            statustip="Create & draw selected projection figure(s)",
            triggered=self.create_draw_projection_figures)
        menu.addAction(create_draw_act)

        # Add create, draw & save action to menu
        create_draw_save_act = QW_QAction(
            self, 'Create, Draw && &Save',
            statustip="Create, draw & save selected projection figure(s)",
            triggered=self.create_draw_save_projection_figures)
        menu.addAction(create_draw_save_act)

        # Save made menu as an attribute
        self.context_menu_u = menu

    # This function shows the context menu for unavailable projections
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def show_unavailable_context_menu(self):
        """
        Shows the 'Unavailable' context menu, giving the user access to its
        actions.

        %(qt_slot)s

        """

        # If there is currently at least one item selected, show context menu
        if len(self.proj_list_u.selectedItems()):
            self.context_menu_u.popup(QG.QCursor.pos())

    # This function shows a list of projection figures in the viewing area
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def show_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and shows them in the projection viewing area.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_d.selectedItems()

        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()

            # Obtain the corresponding figure and subwindow
            fig, subwindow = self.proj_fig_registry[hcube_name]

            # If subwindow is None, create a new one
            if subwindow is None:
                # Create a new subwindow
                subwindow = QW.QMdiSubWindow(self.main.area_dock.proj_area)
                subwindow.setWindowTitle(hcube_name)

                # Set a few properties of the subwindow
                # TODO: Make subwindow frameless when not being hovered
                subwindow.setOption(QW.QMdiSubWindow.RubberBandResize)

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

    # This function closes a list of projection figures permanently
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def close_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and closes their :obj:`~matplotlib.figure.Figure` objects.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_d.selectedItems()

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

    # This function draws a list of projection figures
    # OPTIMIZE: Reshaping a 3D projection figure takes up to 15 seconds
    # TODO: Figure out if there is a way to make a figure static, and only
    # resize when explicitly told to do so
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def draw_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and draws them, creating their :class:`~matplotlib.figure.Figure`
        instances.

        If the `auto_show` option is *True*, drawn figures will be shown
        automatically as well.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_a.selectedItems()

        # Draw projections
        result = self.use_progress_dialog("Drawing projection figures...",
                                          self._draw_projection_figure,
                                          list_items)

        # Show all drawn projection figures if the dialog was not cancelled
        if result and self.main.get_option('auto_show'):
            self.show_projection_figures(list_items)

        # Return result
        return(result)

    # This function draws a projection figure
    def _draw_projection_figure(self, list_item):
        """
        Draws the projection figure requested in the provided `list_item`,
        creating its :class:`~matplotlib.figure.Figure` instance.

        This function is used iteratively by :meth:`~draw_projection_figures`.

        Parameters
        ----------
        list_item : :obj:`~PyQt5.QtWidgets.QListWidgetItem` object
            The item that contains the requested projection figure.

        """

        # Retrieve text of list_item
        hcube_name = list_item.text()
        hcube = self.hcubes[self.names.index(hcube_name)]

        # Load in the data corresponding to the requested figure
        impl_min, impl_los, proj_res, _ =\
            self.call_proj_attr('get_proj_data', hcube)

        # Call the proper function for drawing the projection figure
        if(len(hcube) == 2):
            fig = self.call_proj_attr('draw_2D_proj_fig',
                                      hcube, impl_min, impl_los, proj_res)
        else:
            fig = self.call_proj_attr('draw_3D_proj_fig',
                                      hcube, impl_min, impl_los, proj_res)

        # Register figure in the registry
        self.proj_fig_registry[hcube_name] = [fig, None]

        # Move figure from available to drawn
        item = self.proj_list_a.takeItem(
            self.proj_list_a.row(list_item))
        self.proj_list_d.addItem(item)

    # This function deletes a list of projection figures
    # TODO: Avoid reimplementing the __get_req_hcubes() logic here
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def delete_projection_figures(self, list_items=None, *,
                                  skip_warning=False):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and delete them, permanently removing their corresponding projection
        data.

        %(qt_slot)s

        %(optional)s
        skip_warning : bool. Default: False
            Whether or not to skip showing the warning asking the user if they
            are sure they want to permanently delete all items in `list_items`.
            If *True*, the answer is taken to be *True*.

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_a.selectedItems()

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
                    self.call_proj_attr('get_fig_path', hcube)
                if path.exists(fig_path):
                    os.remove(fig_path)
                if path.exists(fig_path_s):
                    os.remove(fig_path_s)

                # Move figure from available to unavailable
                item = self.proj_list_a.takeItem(
                    self.proj_list_a.row(list_item))
                self.proj_list_u.addItem(item)

    # This function creates a list of projection figures
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def create_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and creates them, calculating their corresponding projection data.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_u.selectedItems()

        # Create projections
        result = self.use_progress_dialog("Creating projection figures...",
                                          self._create_projection_figure,
                                          list_items)

        # Return result
        return(result)

    # This function creates a projection figure
    def _create_projection_figure(self, list_item):
        """
        Creates the projection figure requested in the provided `list_item`,
        calculating its projection data.

        This function is used iteratively by
        :meth:`~create_projection_figures`.

        Parameters
        ----------
        list_item : :obj:`~PyQt5.QtWidgets.QListWidgetItem` object
            The item that contains the requested projection figure.

        """

        # Retrieve text of list_item
        hcube_name = list_item.text()
        hcube = self.hcubes[self.names.index(hcube_name)]

        # Calculate projection data
        _, _ = self.all_call_proj_attr('analyze_proj_hcube', hcube)

        # Move figure from unavailable to available
        item = self.proj_list_u.takeItem(self.proj_list_u.row(list_item))
        self.proj_list_a.addItem(item)

    # This function saves a list of projection figures to file
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def save_projection_figures(self, list_items=None, *, choose=False):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and saves their :obj:`~matplotlib.figure.Figure` objects.

        %(qt_slot)s

        %(optional)s
        choose : bool. Default: False
            Whether or not the user is allowed to choose where the projection
            figure is saved to.
            If *False*, it uses the default filename as defined by
            :meth:`~prism.Pipeline._Projection__get_fig_path`.

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_d.selectedItems()

        # Loop over all items in list_items
        for list_item in list_items:
            # Retrieve text of list_item
            hcube_name = list_item.text()
            hcube = self.hcubes[self.names.index(hcube_name)]

            # Obtain the corresponding figure
            fig, _ = self.proj_fig_registry[hcube_name]

            # Obtain the default figure path
            fig_paths = self.call_proj_attr('get_fig_path', hcube)
            fig_path = fig_paths[self.get_proj_attr('smooth')]

            # If choose, save using non-default figure path
            if choose:
                # Get the supported filetypes
                filetypes = FigureCanvas.get_supported_filetypes_grouped()

                # Get dict of all supported file extensions in MPL
                ext_dict = sdict()
                for name, exts in filetypes.items():
                    ext_dict[name] = ' '.join(['*.%s' % (ext) for ext in exts])

                # Set default extension
                default_ext = '*.png'

                # Initialize empty list of filters and default filter
                file_filters = []
                default_filter = None

                # Obtain list with the different file filters
                for name, ext in ext_dict.items():
                    # Create proper string layout for this filter
                    file_filter = "%s (%s)" % (name, ext)
                    file_filters.append(file_filter)

                    # If this extension is the default one, save it as such
                    if default_ext in file_filter:
                        default_filter = file_filter

                # Add 'All (Image) Files' filter to the list of filters
                file_filters.append("All Image Files (%s)"
                                    % (' '.join(ext_dict.values())))
                file_filters.append("All Files (*)")

                # Combine list into a single string
                file_filters = ';;'.join(file_filters)

                # Create an OS-dependent options dict
                options = {}

                # Do not use Linux' native dialog as it is bad on some dists
                if platform.startswith('linux'):
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

    # This function saves a list of projection figures to file
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def save_as_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and saves their :obj:`~matplotlib.figure.Figure` objects, asking the
        user where to save each one.

        This function basically calls :meth:`~save_projection_figures` with
        `choose` set to *True*.

        %(qt_slot)s

        %(optional)s

        """

        self.save_projection_figures(list_items, choose=True)

    # This function redraws a list of projection figures
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def redraw_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and redraws them, closing and recreating their
        :obj:`~matplotlib.figure.Figure` objects.

        This function is basically a combination of
        :meth:`~close_projection_figures` and :meth:`~draw_projection_figures`.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_d.selectedItems()

        # Close and redraw all projection figures in list_items
        self.close_projection_figures(list_items)
        self.draw_projection_figures(list_items)

    # This function draws and saves a list of projection figures
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def draw_save_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided
        `list_items`, draws their :obj:`~matplotlib.figure.Figure` objects and
        saves them afterward.

        This function is basically a combination of
        :meth:`~draw_projection_figures` and :meth:`~save_projection_figures`.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_a.selectedItems()

        # Draw and save all projection figures in list_items
        if self.draw_projection_figures(list_items):
            self.save_projection_figures(list_items)

    # This function recreates a list of projection figures
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def recreate_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided `list_items`
        and recreates them, permanently removing their corresponding projection
        data and recalculating it.

        This function is basically a combination of
        :meth:`~delete_projection_figures` and
        :meth:`~create_projection_figures`.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_a.selectedItems()

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

    # This function creates and draws a list of projection figures
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def create_draw_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided
        `list_items`, calculates their projection data and draws their
        :obj:`~matplotlib.figure.Figure` objects afterward.

        This function is basically a combination of
        :meth:`~create_projection_figures` and
        :meth:`~draw_projection_figures`.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_u.selectedItems()

        # Create and draw all projection figures in list_items
        if self.create_projection_figures(list_items):
            self.draw_projection_figures(list_items)

    # This function creates, draws and saves a list of projection figures
    @QC.pyqtSlot()
    @QC.pyqtSlot(list)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_items_optional_doc)
    def create_draw_save_projection_figures(self, list_items=None):
        """
        Retrieves the projection figures requested in the provided
        `list_items`, calculates their projection data, draws their
        :obj:`~matplotlib.figure.Figure` objects and saves them afterward.

        This function is basically a combination of
        :meth:`~create_projection_figures`; :meth:`~draw_projection_figures`
        and :meth:`~save_projection_figures`.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_items
        if list_items is None:
            list_items = self.proj_list_u.selectedItems()

        # Create, draw and save all projection figures in list_items
        if self.create_projection_figures(list_items):
            if self.draw_projection_figures(list_items):
                self.save_projection_figures(list_items)

    # This function shows a details overview of a drawn projection figure
    @QC.pyqtSlot()
    @QC.pyqtSlot(QW.QListWidgetItem)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_item_optional_doc)
    def details_drawn_projection_figure(self, list_item=None):
        """
        Retrieves the projection figure requested in the provided `list_item`,
        gathers its properties and shows a details dialog listing them.

        This function is used for projections in the 'Drawn' list.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_item
        if list_item is None:
            list_item = self.proj_list_d.selectedItems()[0]

        # Show details
        self._details_projection_figure(list_item)

    # This function shows a details overview of an available projection figure
    @QC.pyqtSlot()
    @QC.pyqtSlot(QW.QListWidgetItem)
    @docstring_substitute(qt_slot=qt_slot_doc,
                          optional=list_item_optional_doc)
    def details_available_projection_figure(self, list_item=None):
        """
        Retrieves the projection figure requested in the provided `list_item`,
        gathers its properties and shows a details dialog listing them.

        This function is used for projections in the 'Available' list.

        %(qt_slot)s

        %(optional)s

        """

        # Obtain the list_item
        if list_item is None:
            list_item = self.proj_list_a.selectedItems()[0]

        # Show details
        self._details_projection_figure(list_item)

    # This function shows a details overview of a projection figure
    # TODO: Add section on how the figure was drawn for drawn projections?
    def _details_projection_figure(self, list_item):
        """
        Creates and shows a details dialog for the projection figure requested
        in the provided `list_item`.

        Parameters
        ----------
        list_item : :obj:`~PyQt5.QtWidgets.QListWidgetItem` object
            The item that contains the requested projection figure.

        """

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
        general_layout.addRow("Emulator iteration", emul_i)
        general_layout.addRow("Parameters", pars)
        general_layout.addRow("Projection type", proj_type)
        general_layout.addRow("% of parameter space remaining",
                              pl_space_rem)

        # PROJECTION DATA
        # Create a group for the projection data details
        data_group = QW.QGroupBox("Projection data")
        details_layout.addWidget(data_group)
        data_layout = QW.QFormLayout()
        data_group.setLayout(data_layout)

        # Add projection data details
        data_layout.addRow("Grid shape", grid_shape)
        data_layout.addRow("Grid size", grid_size)
        data_layout.addRow("# of implausibility wildcards", cut_idx)
        data_layout.addRow("Implausibility cut-offs", impl_cut)

        # Create a details message box for this projection figure
        details_box = QW.QDialog(self.main)
        details_box.setWindowModality(QC.Qt.NonModal)
        details_box.setAttribute(QC.Qt.WA_DeleteOnClose)
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

    # This function creates the proper progress dialog for given operation
    def use_progress_dialog(self, label, func, *iterables):
        """
        Creates a progress dialog with the given `label`, and executes the
        requested `func` using the provided `iterables`.

        Depending on the current settings, this function will either create a
        :obj:`~prism._gui.widgets.helpers.ThreadedProgressDialog` object that
        allows the user to abort the operation (but is slower), or a static
        dialog that cannot be interrupted.

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

        Returns
        -------
        result : bool
            Whether or not the operations ended successfully, which can be used
            by other functions to determine if it should continue.

        """

        # Set result to False
        result = False

        # Use a progress dialog if one was requested
        if self.main.get_option('use_progress_dialog'):
            # Create a threaded progress dialog for creating projections
            progress_dialog = ThreadedProgressDialog(
                self.main, label, func, *iterables)

            # Wrap in try-statement to make sure the dialog is always closed
            try:
                # Execute the function provided to the progress dialog
                result = progress_dialog()
            # Close dialog
            finally:
                progress_dialog.close()

        # Else, do not use one and execute on main thread
        else:
            # Create a dialog showing that an operation is being executed
            dialog = QW.QDialog(self.main)
            dialog.setWindowModality(QC.Qt.ApplicationModal)
            dialog.setWindowTitle(APP_NAME)
            dialog.setAttribute(QC.Qt.WA_DeleteOnClose)
            dialog.setWindowFlags(
                QC.Qt.WindowTitleHint |
                QC.Qt.Dialog |
                QC.Qt.CustomizeWindowHint)
            layout = QW.QVBoxLayout(dialog)
            layout.addWidget(QW.QLabel(label))

            # Show the dialog
            dialog.show()

            # Wait for 0.1 seconds and then process events to update dialog
            time.sleep(0.1)
            QW.QApplication.instance().processEvents()

            # Wrap in try-statement to make sure the dialog is always closed
            try:
                # Loop over all iterables
                for items in zip(*iterables):
                    func(*items)

                # If this finishes successfully, set result to True
                else:
                    result = True

            # Close dialog
            finally:
                dialog.close()

        # Return result
        return(result)
