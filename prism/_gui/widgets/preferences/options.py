# -*- coding: utf-8 -*-

"""
GUI Options
===========
Provides the main :class:`~PyQt5.QtWidgets.QDialog` subclass that creates the
preferences menu and keeps track of all internally saved options.
The window used for the kwargs dicts is defined in
:mod:`~prism._gui.widgets.preferences.kwargs_dicts`.

"""


# %% IMPORTS
# Package imports
from e13tools.utils import docstring_substitute
from PyQt5 import QtCore as QC, QtWidgets as QW
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._docstrings import (
    kwargs_doc, proj_depth_doc, proj_res_doc, qt_slot_doc)
from prism._gui.widgets import (
    QW_QDoubleSpinBox, QW_QSpinBox, get_box_value, get_modified_box_signal,
    set_box_value)
from prism._gui.widgets.preferences.kwargs_dicts import KwargsDictBoxLayout

# All declaration
__all__ = ['OptionsDialog', 'OptionsEntry']


# %% CLASS DEFINITIONS
# Define class for options dialog
class OptionsDialog(QW.QDialog):
    """
    Defines the :class:`~OptionsDialog` class for the Projection GUI.

    This class provides both the 'Preferences' dialog and the functions that
    are required to load; save; set; and change them.

    """

    # Create saving, resetting and discarding signals
    saving = QC.pyqtSignal()
    resetting = QC.pyqtSignal()
    discarding = QC.pyqtSignal()

    @docstring_substitute(optional=kwargs_doc.format(
        'PyQt5.QtWidgets.QDialog'))
    def __init__(self, main_window_obj, *args, **kwargs):
        """
        Initialize an instance of the :class:`~OptionsDialog` class.

        Parameters
        ----------
        main_window_obj : :obj:`~prism._gui.widgets.MainViewerWindow` object
            Instance of the :class:`~prism._gui.widgets.MainViewerWindow` class
            that acts as the parent of this dialog.

        %(optional)s

        """

        # Save provided MainWindow object
        self.main = main_window_obj
        self.pipe = self.main.pipe
        self.n_par = self.main.n_par
        self.set_proj_attr = self.main.set_proj_attr
        self.all_set_proj_attr = self.main.all_set_proj_attr
        self.get_proj_attr = self.main.get_proj_attr
        self.call_proj_attr = self.main.call_proj_attr
        self.all_call_proj_attr = self.main.all_call_proj_attr

        # Call super constructor
        super().__init__(self.main, *args, **kwargs)

        # Create the options window
        self.init()

    # This function shows the options window
    @QC.pyqtSlot()
    def __call__(self):
        """
        Qt slot that shows the options dialog in the center of the main window.

        """

        # Show it
        self.show()

        # Move the options window to the center of the main window
        self.move(self.main.geometry().center()-self.rect().center())

    # This function overrides the closeEvent method
    def closeEvent(self, *args, **kwargs):
        """
        Special :meth:`~PyQt5.QtWidgets.QWidget.closeEvent` event that makes
        sure that all dialogs will be closed related to the options menu, and
        discards all changes made.

        """

        # Make sure the kwargs dict dialog is closed
        self.dict_dialog.close()

        # Close the window
        super().closeEvent(*args, **kwargs)

        # Set all option boxes back to their saved values
        self.discard_options()

    # This function creates the options window
    def init(self):
        """
        Sets up the options dialog after it has been initialized.

        This function is mainly responsible for initializing all option entries
        that the GUI has, and creating a database for them. It also creates the
        layout of the options dialog.

        """

        # Create a window layout
        window_layout = QW.QVBoxLayout(self)

        # Create a tab widget
        window_tabs = QW.QTabWidget()
        window_layout.addWidget(window_tabs)

        # Create a options dict
        self.option_entries = sdict()

        # Define list with all tabs that should be available in what order
        option_tabs = ['general', 'appearance']

        # Include all tabs named in options_tabs
        for tab in option_tabs:
            window_tabs.addTab(*getattr(self, 'create_tab_%s' % (tab))())

        # Also add the buttons
        self.create_group_buttons(window_layout)

        # Set a few properties of options window
        self.setWindowModality(QC.Qt.WindowModal)           # Modality
        self.setWindowTitle("Preferences")                  # Title

        # Add a new method to self.main
        self.main.get_option = self.get_option

    # This function returns the value of a specific option
    def get_option(self, name):
        """
        Returns the value of the option entry associated with the given `name`.

        """

        return(self.option_entries[name].value)

    # This function creates a new options entry
    def create_entry(self, name, box, default):
        """
        Creates a new :class:`~OptionsEntry` instance, using the provided
        `name`, `box` and `default`, and registers it in the options dialog.

        Parameters
        ----------
        name : str
            The name of this options entry.
        box : :obj:`~PyQt5.QtWidgets.QWidget` object
            The widget that will hold the values of this entry.
        default : object
            The default value of this entry.

        """

        # Create new options entry
        entry = OptionsEntry(name, box, default)

        # Connect box signals
        get_modified_box_signal(box).connect(self.enable_save_button)

        # Connect entry slots
        self.saving.connect(entry.save_value)
        self.resetting.connect(entry.reset_value)
        self.discarding.connect(entry.discard_value)

        # Add new entry to option_entries
        self.option_entries[name] = entry

    # This function creates a new tab
    def create_tab(self, name, groups_list):
        """
        Creates a new options tab with the given `name` and adds the groups
        defined in `groups_list` to it.

        This function acts as a base function called by `create_tab_`
        functions.

        Parameters
        ----------
        name : str
            The name of this options tab.
        groups_list : list of str
            A list containing the names of all option groups that need to be
            added to this tab.

        Returns
        -------
        tab : :obj:`~PyQt5.QtWidgets.QWidget` object
            The created options tab.
        name : str
            The name of this options tab as provided with `name`.
            This variable is mainly returned such that it is easier to pass tab
            names between functions.

        """

        # Create a tab
        tab = QW.QWidget()
        layout = QW.QVBoxLayout()
        tab.setLayout(layout)

        # Include all groups named in groups_list
        for group in groups_list:
            layout.addWidget(getattr(self, 'create_group_%s' % (group))())

        # Add a stretch
        layout.addStretch()

        # Return tab
        return(tab, name)

    # This function creates a new group
    def create_group(self, name, options_list):
        """
        Creates a new option group with the given `name` and adds the options
        defined in `options_list` to it.

        This function acts as a base function called by `create_group_`
        functions.

        Parameters
        ----------
        name : str
            The name of this option group.
        options_list : list of str
            A list containing the names of all options that need to be added to
            this group.

        Returns
        -------
        group : :obj:`~PyQt5.QtWidgets.QGroupBox` object
            The created option group.

        """

        # Create a group
        group = QW.QGroupBox(name)
        layout = QW.QFormLayout()
        group.setLayout(layout)

        # Include all options named in options_list
        for option in options_list:
            layout.addRow(*getattr(self, 'create_option_%s' % (option))())

        # Return group
        return(group)

    # GENERAL TAB
    def create_tab_general(self):
        """
        Creates the 'General' tab and returns it.

        """

        self.proj_defaults = sdict(self.get_proj_attr('proj_kwargs'))
        self.proj_keys = list(self.proj_defaults.keys())
        self.proj_keys.remove('align')
        self.proj_keys.extend(['align_col', 'align_row'])
        return(self.create_tab("General", ['proj_grid', 'proj_kwargs']))

    # INTERFACE TAB
    def create_tab_appearance(self):
        """
        Creates the 'Appearance' tab and returns it.

        """

        return(self.create_tab("Appearance", ['interface']))

    # PROJ_GRID GROUP
    def create_group_proj_grid(self):
        """
        Creates the 'Projection grid' group and returns it.

        """

        return(self.create_group("Projection grid",
                                 ['proj_res', 'proj_depth']))

    # PROJ_KWARGS GROUP
    def create_group_proj_kwargs(self):
        """
        Creates the 'Projection keywords' group and returns it.

        """

        return(self.create_group("Projection keywords",
                                 ['align', 'show_cuts', 'smooth',
                                  'kwargs_dicts']))

    # INTERFACE GROUP
    def create_group_interface(self):
        """
        Creates the 'Interface' group and returns it.

        """

        return(self.create_group("Interface", ['auto_show', 'auto_tile',
                                               'progress_dialog']))

    # FONTS GROUP
    def create_group_fonts(self):   # pragma: no cover
        """
        Creates the 'Fonts' group and returns it.

        """

        return(self.create_group("Fonts", ['text_fonts']))

    # TEXT_FONTS OPTION
    # TODO: Further implement this
    def create_option_text_fonts(self):     # pragma: no cover
        """
        Creates the 'text_fonts' option and returns it.

        This option allows for the fonts used in the GUI to be modified.

        """

        # PLAIN TEXT
        # Create a font families combobox
        plain_box = QW.QFontComboBox()
        plain_box.setFontFilters(QW.QFontComboBox.MonospacedFonts)
        plain_box.setEditable(True)
        plain_box.setInsertPolicy(plain_box.NoInsert)
        plain_box.completer().setCompletionMode(QW.QCompleter.PopupCompletion)

        # Create a font size spinbox
        plain_size = QW_QSpinBox()
        plain_size.setRange(7, 9999999)
        plain_size.setSuffix(" pts")

        # RICH TEXT
        # Create a font families combobox
        rich_box = QW.QFontComboBox()
        rich_box.setEditable(True)
        rich_box.setInsertPolicy(rich_box.NoInsert)
        rich_box.completer().setCompletionMode(QW.QCompleter.PopupCompletion)

        # Create a font size spinbox
        rich_size = QW_QSpinBox()
        rich_size.setRange(7, 9999999)
        rich_size.setSuffix(" pts")

        # Create a grid for the families and size boxes
        font_grid = QW.QGridLayout()
        font_grid.setColumnStretch(1, 2)
        font_grid.setColumnStretch(3, 1)

        # Add everything to this grid
        font_grid.addWidget(QW.QLabel("Plain text:"), 0, 0)
        font_grid.addWidget(plain_box, 0, 1)
        font_grid.addWidget(QW.QLabel("Size:"), 0, 2)
        font_grid.addWidget(plain_size, 0, 3)
        font_grid.addWidget(QW.QLabel("Rich text:"), 1, 0)
        font_grid.addWidget(rich_box, 1, 1)
        font_grid.addWidget(QW.QLabel("Size:"), 1, 2)
        font_grid.addWidget(rich_size, 1, 3)

        font_grid.addWidget(QW.QLabel("NOTE: Does not work yet"), 2, 0, 1, 4)

        # Return the grid
        return(font_grid,)

    # DPI OPTION
    # TODO: Further implement this one as well
    def create_option_dpi(self):    # pragma: no cover
        """
        Creates the 'dpi' option and returns it.

        This option allows for the DPI used in the GUI to be modified.

        """

        # Make a checkbox for setting a custom DPI scaling
        dpi_check = QW.QCheckBox("Custom DPI scaling:")
        dpi_check.setToolTip("Set this to enable custom DPI scaling of the "
                             "GUI")
        self.create_entry('dpi_flag', dpi_check, False)

        # Make a spinbox for setting the DPI scaling
        dpi_box = QW_QDoubleSpinBox()
        dpi_box.setRange(0, 100)
        dpi_box.setSuffix("x")
        dpi_box.setSpecialValueText("Auto")
        dpi_box.setToolTip("Custom DPI scaling factor to use. "
                           "'1.0' is no scaling. "
                           "'Auto' is automatic scaling.")
        dpi_check.toggled.connect(dpi_box.setEnabled)
        dpi_box.setEnabled(False)
        self.create_entry('dpi_scaling', dpi_box, 1.0)

        # Return DPI box
        return(dpi_check, dpi_box)

    # AUTO_TILE OPTION
    def create_option_auto_tile(self):
        """
        Creates the 'auto_tile' option and returns it.

        This option sets whether the projection subwindows are automatically
        tiled.

        """

        # Make check box for auto tiling
        auto_tile_box = QW.QCheckBox("Auto-tile subwindows")
        auto_tile_box.setToolTip("Set this to automatically tile all "
                                 "projection subwindows whenever a new one is "
                                 "added")
        self.create_entry('auto_tile', auto_tile_box, True)

        # Return auto_tile box
        return(auto_tile_box,)

    # AUTO_SHOW OPTION
    def create_option_auto_show(self):
        """
        Creates the 'auto_show' option and returns it.

        This option sets whether the projection subwindows are automatically
        shown whenever created.

        """

        # Make check box for auto showing projection figures/subwindows
        auto_show_box = QW.QCheckBox("Auto-show subwindows")
        auto_show_box.setToolTip("Set this to automatically show a projection "
                                 "subwindow after it has been drawn")
        self.create_entry('auto_show', auto_show_box, True)

        # Return auto_show box
        return(auto_show_box,)

    # PROGRESS_DIALOG OPTION
    def create_option_progress_dialog(self):
        """
        Creates the 'progress_dialog' option and returns it.

        This option sets whether a threaded progress dialog is used for some
        operations.

        """

        # Make check box for using a threaded progress dialog
        progress_dialog_box = QW.QCheckBox("Use threaded progress dialog")
        progress_dialog_box.setToolTip(
            "Set this to use a threaded progress dialog whenever projections "
            "are created or drawn.\nThis allows for the operation to be "
            "monitored and/or aborted, but also slows down the execution")
        self.create_entry('use_progress_dialog', progress_dialog_box, True)

        # Return progress_dialog box
        return(progress_dialog_box,)

    # PROJ_RES OPTION
    def create_option_proj_res(self):
        """
        Creates the 'proj_res' option and returns it.

        This option sets the value of the 'proj_res' projection parameter.

        """

        # Make spinbox for option proj_res
        proj_res_box = QW_QSpinBox()
        proj_res_box.setRange(0, 9999999)
        proj_res_box.setToolTip(proj_res_doc)
        self.create_entry('proj_res', proj_res_box,
                          self.proj_defaults['proj_res'])

        # Return resolution box
        return('Resolution:', proj_res_box)

    # PROJ_DEPTH OPTION
    def create_option_proj_depth(self):
        """
        Creates the 'proj_depth' option and returns it.

        This option sets the value of the 'proj_depth' projection parameter.

        """

        # Make spinbox for option proj_depth
        proj_depth_box = QW_QSpinBox()
        proj_depth_box.setRange(0, 9999999)
        proj_depth_box.setToolTip(proj_depth_doc)
        self.create_entry('proj_depth', proj_depth_box,
                          self.proj_defaults['proj_depth'])

        # Return depth box
        return('Depth:', proj_depth_box)

    # ALIGN OPTION
    def create_option_align(self):
        """
        Creates the 'align' option and returns it.

        This option sets the value of the 'align' projection parameter.

        """

        # Column align
        align_col_box = QW.QRadioButton('Column')
        align_col_box.setToolTip("Align the projection subplots in a single "
                                 "column")
        self.create_entry('align_col', align_col_box,
                          self.proj_defaults['align'] == 'col')

        # Row align
        align_row_box = QW.QRadioButton('Row')
        align_row_box.setToolTip("Align the projection subplots in a single "
                                 "row")
        self.create_entry('align_row', align_row_box,
                          self.proj_defaults['align'] == 'row')

        # Create layout for align and add it to options layout
        align_box = QW.QHBoxLayout()
        align_box.addWidget(align_col_box)
        align_box.addWidget(align_row_box)
        align_box.addStretch()

        # Return alignment box
        return('Alignment:', align_box)

    # SHOW_CUTS OPTION
    def create_option_show_cuts(self):
        """
        Creates the 'show_cuts' option and returns it.

        This option sets the value of the 'show_cuts' projection parameter.

        """

        # Make check box for show_cuts
        show_cuts_box = QW.QCheckBox()
        show_cuts_box.setToolTip("Enable/disable showing all implausibility "
                                 "cut-off lines in 2D projections")
        self.create_entry('show_cuts', show_cuts_box,
                          self.proj_defaults['show_cuts'])

        # Return shot_cuts box
        return('Show cuts?', show_cuts_box)

    # SMOOTH OPTION
    def create_option_smooth(self):
        """
        Creates the 'smooth' option and returns it.

        This option sets the value of the 'smooth' projection parameter.

        """

        # Make check box for smooth
        smooth_box = QW.QCheckBox()
        smooth_box.setToolTip("Enable/disable smoothing the projections. When "
                              "smoothed, the minimum implausibility is forced "
                              "to be above the first cut-off for implausible "
                              "regions")
        self.create_entry('smooth', smooth_box, self.proj_defaults['smooth'])

        # Return smooth box
        return('Smooth?', smooth_box)

    # KWARGS_DICTS OPTION
    def create_option_kwargs_dicts(self):
        """
        Creates the 'kwargs_dicts' option and returns it.

        This option allows for the
        :class:`~prism._gui.widgets.preferences.KwargsDictDialog` to be shown
        to the user.
        This dialog is able to set the values of all 'XXX_kwargs' projection
        parameters.

        """

        # Create a kwargs_dict_box
        kwargs_dict_box = KwargsDictBoxLayout(self)
        self.kwargs_dict_box = kwargs_dict_box

        # Add all kwargs_dicts to it
        # FIG_KWARGS
        kwargs_dict_box.add_dict(
            "Figure", 'fig_kwargs',
            std_entries=['dpi', 'figsize'],
            banned_entries=self.get_proj_attr('pop_fig_kwargs'))

        # IMPL_KWARGS_2D
        kwargs_dict_box.add_dict(
            "2D implausibility", 'impl_kwargs_2D',
            std_entries=['linestyle', 'linewidth', 'marker', 'markersize',
                         'color', 'alpha'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs'), 'cmap'])

        # IMPL_KWARGS_3D
        kwargs_dict_box.add_dict(
            "3D implausibility", 'impl_kwargs_3D',
            std_entries=['cmap', 'alpha', 'xscale', 'yscale'],
            banned_entries=self.get_proj_attr('pop_plt_kwargs'))

        # LOS_KWARGS_2D
        kwargs_dict_box.add_dict(
            "2D line-of-sight", 'los_kwargs_2D',
            std_entries=['linestyle', 'linewidth', 'marker', 'markersize',
                         'color', 'alpha'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs'), 'cmap'])

        # LOS_KWARGS_3D
        kwargs_dict_box.add_dict(
            "3D line-of-sight", 'los_kwargs_3D',
            std_entries=['cmap', 'alpha', 'xscale', 'yscale'],
            banned_entries=self.get_proj_attr('pop_plt_kwargs'))

        # LINE_KWARGS_EST
        kwargs_dict_box.add_dict(
            "Estimate lines", 'line_kwargs_est',
            std_entries=['linestyle', 'color', 'alpha', 'linewidth'],
            banned_entries=[])

        # LINE_KWARGS_CUT
        kwargs_dict_box.add_dict(
            "Cut-off lines", 'line_kwargs_cut',
            std_entries=['linestyle', 'color', 'alpha', 'linewidth'],
            banned_entries=[])

        # Return kwargs_dict box
        return('Projection keyword dicts:', kwargs_dict_box)

    # BUTTONS GROUP
    def create_group_buttons(self, window_layout):
        """
        Creates the button box that is shown at the bottom of the options
        dialog and registers it in the provided `window_layout`.

        """

        # Create a button_box
        button_box = QW.QDialogButtonBox()
        window_layout.addWidget(button_box)

        # Make a 'Reset' button
        reset_but = button_box.addButton(button_box.Reset)
        reset_but.setToolTip("Reset to defaults")
        reset_but.clicked.connect(self.reset_options)
        self.reset_but = reset_but

        # Make an 'Apply' button
        save_but = button_box.addButton(button_box.Apply)
        save_but.setToolTip("Apply changes")
        save_but.clicked.connect(self.save_options)
        save_but.setEnabled(False)
        self.save_but = save_but

        # Make a 'Close' button
        close_but = button_box.addButton(button_box.Close)
        close_but.setToolTip("Close without saving")
        close_but.clicked.connect(self.close)
        close_but.setDefault(True)
        self.close_but = close_but

    # This function saves the new options values
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def save_options(self):
        """
        Saves all current values of all option entries.

        Option entries that affect projection parameters are automatically
        modified as well.

        %(qt_slot)s

        """

        # Emit the saving signal
        self.saving.emit()

        # Save all new values
        for key, entry in self.option_entries.items():
            # If key is a projection parameter, save it in the Pipeline as well
            if key in self.proj_keys:
                # Align
                if key in ['align_col', 'align_row']:
                    if entry.box.isChecked():
                        self.set_proj_attr('align', key[6:])
                else:
                    self.set_proj_attr(key, entry.value)

        # Disable the save button
        self.disable_save_button()

    # This function enables the save button
    @QC.pyqtSlot()
    def enable_save_button(self):
        """
        Qt slot that enables the save button at the bottom of the options
        dialog.
        The save button is enabled if at least one change has been made to any
        option entry.

        """

        self.save_but.setEnabled(True)

    # This function disables the save button
    @QC.pyqtSlot()
    def disable_save_button(self):
        """
        Qt slot that disables the save button at the bottom of the options
        dialog.
        The save button is disabled whenever no changes have been made to any
        option entry.

        """

        self.save_but.setEnabled(False)

    # This function resets the options to default
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def reset_options(self):
        """
        Resets the saved and current values of all option entries back to their
        default values.

        %(qt_slot)s

        """

        # Emit the resetting signal
        self.resetting.emit()

        # Save current options
        self.save_options()

    # This function discards all changes to the options
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def discard_options(self):
        """
        Discards the current values of all option entries and sets them back to
        their saved values.

        %(qt_slot)s

        """

        # Emit the discarding signal
        self.discarding.emit()

        # Disable the save button
        self.disable_save_button()


# Define class used as a container for options entries
class OptionsEntry(QC.QObject):
    """
    Defines the :class:`~OptionsEntry` class.

    This class is used as a container for making option entries in the
    :class:`~OptionsDialog` class.

    """

    # Initialize options entry
    def __init__(self, name, box, default):
        """
        Initialize an instance of the :class:`~OptionsEntry` class.

        Parameters
        ----------
        name : str
            The name of this options entry.
        box : :obj:`~PyQt5.QtWidgets.QWidget` object
            The widget that will hold the values of this entry.
        default : object
            The default value of this entry.

        """

        # Save provided name, box and default
        self._name = name
        self._box = box
        self._default = default

        # Call super constructor
        super().__init__()

        # Initialize the options entry
        self.init()

    # Create a representation of this entry
    def __repr__(self):
        return("OptionsEntry(name=%s, box=%s, default=%s, value=%s)"
               % (self.name, self.box, self.default, self.value))

    # This function creates the options entry
    def init(self):
        """
        Sets up the options entry after it has been initialized.

        This function is mainly responsible for making sure that the current
        and saved values of this entry are set to its default value.

        """

        # Set the box value to the default value and save it
        self.reset_value()
        self.save_value()

    # This property contains the name of this entry
    @property
    def name(self):
        """
        str: The name of this options entry.

        """

        return(self._name)

    # This property contains the associated QWidget object of this entry
    @property
    def box(self):
        """
        :obj:`~PyQt5.QtWidgets.QWidget` object: The widget box that contains
        this options entry.

        """

        return(self._box)

    # This property contains the default value of this entry
    @property
    def default(self):
        """
        object: The default value of this options entry.

        """

        return(self._default)

    # This property contains the currently saved value of this entry
    @property
    def value(self):
        """
        object: The currently saved value of this options entry.

        """

        return(self._value)

    # This function saves the value of this entry's box
    @QC.pyqtSlot()
    def save_value(self):
        """
        Qt slot that saves the current value of this options entry.

        """

        self._value = get_box_value(self._box)

    # This function resets the value of this entry's box to the default value
    @QC.pyqtSlot()
    def reset_value(self):
        """
        Qt slot that resets the current value of this options entry to its
        default value.

        """

        set_box_value(self._box, self._default)

    # This function sets the value of this entry's box to the saved value
    @QC.pyqtSlot()
    def discard_value(self):
        """
        Qt slot that discards the current value and sets it back to its saved
        value.

        """

        set_box_value(self._box, self._value)
