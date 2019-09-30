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
from PyQt5 import QtCore as QC, QtWidgets as QW
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._docstrings import proj_depth_doc, proj_res_doc
from prism._gui.widgets import QW_QDoubleSpinBox, QW_QSpinBox
from prism._gui.widgets.preferences.helpers import (
    get_box_value, options_entry, set_box_value)
from prism._gui.widgets.preferences.kwargs_dicts import KwargsDictBoxLayout

# All declaration
__all__ = ['OptionsDialog']


# %% CLASS DEFINITIONS
# Define class for options dialog
class OptionsDialog(QW.QDialog):
    def __init__(self, main_window_obj, *args, **kwargs):
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

    # This function creates the options window
    def init(self):
        # Create a window layout
        window_layout = QW.QVBoxLayout(self)

        # Create a tab widget
        window_tabs = QW.QTabWidget()
        window_layout.addWidget(window_tabs)

        # Create a options dict
        self.options_entries = sdict()

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
        self.setWindowModality(QC.Qt.WindowModal)           # Modality
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
        # Make sure the kwargs dict dialog is closed
        self.dict_dialog.close()

        # Close the window
        super().closeEvent(*args, **kwargs)

        # Set all option boxes back to their current values
        self.set_options()

    # This function returns the value of a specific option
    def get_option(self, name):
        return(self.options_entries[name].value)

    # This function creates a new tab
    def create_tab(self, name, page_widget, *groups_list):
        # Create a tab
        options_tab = QW.QWidget()
        page_layout = QW.QVBoxLayout()
        options_tab.setLayout(page_layout)

        # Include all groups named in groups_list
        for group in groups_list:
            getattr(self, 'add_group_%s' % (group))(page_layout)

        # Add a stretch
        page_layout.addStretch()

        # Add tab to page_widget
        page_widget.addTab(options_tab, name)

    # This function creates a new group
    def create_group(self, name, page_layout, *options_list):
        # Create a group
        options_group = QW.QGroupBox(name)
        group_layout = QW.QFormLayout()
        options_group.setLayout(group_layout)

        # Include all options named in options_list
        for option in options_list:
            getattr(self, 'add_option_%s' % (option))(group_layout)

        # Add group to tab
        page_layout.addWidget(options_group)

    # GENERAL TAB
    def add_tab_general(self, *args):
        self.proj_defaults = sdict(self.get_proj_attr('proj_kwargs'))
        self.proj_keys = list(self.proj_defaults.keys())
        self.proj_keys.remove('align')
        self.proj_keys.extend(['align_col', 'align_row'])
        self.create_tab("General", *args,
                        'proj_grid', 'proj_kwargs')

    # INTERFACE TAB
    def add_tab_appearance(self, *args):
        self.create_tab("Appearance", *args, 'interface')

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
        self.create_group("Interface", *args, 'auto_tile', 'auto_show')

    # FONTS GROUP
    def add_group_fonts(self, *args):
        self.create_group("Fonts", *args, 'text_fonts')

    # TEXT_FONTS OPTION
    # TODO: Further implement this
    def add_option_text_fonts(self, group_layout):
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
        group_layout.addRow(font_grid)

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

    # DPI OPTION
    # TODO: Further implement this one as well
    def add_option_dpi(self, group_layout):
        # Make a checkbox for setting a custom DPI scaling
        dpi_check = QW.QCheckBox("Custom DPI scaling:")
        dpi_check.setToolTip("Set this to enable custom DPI scaling of the "
                             "GUI")
        dpi_check.toggled.connect(self.enable_save_button)
        self.options_entries['dpi_flag'] = options_entry(dpi_check, False)

        # Make a spinbox for setting the DPI scaling
        dpi_box = QW_QDoubleSpinBox()
        dpi_box.setRange(0, 100)
        dpi_box.setSuffix("x")
        dpi_box.setSpecialValueText("Auto")
        dpi_box.setToolTip("Custom DPI scaling factor to use. "
                           "'1.0' is no scaling. "
                           "'Auto' is automatic scaling.")
        dpi_box.valueChanged.connect(self.enable_save_button)
        dpi_check.toggled.connect(dpi_box.setEnabled)
        dpi_box.setEnabled(False)
        self.options_entries['dpi_scaling'] = options_entry(dpi_box, 1.0)
        group_layout.addRow(dpi_check, dpi_box)

    # AUTO_TILE OPTION
    def add_option_auto_tile(self, group_layout):
        # Make check box for auto tiling
        auto_tile_box = QW.QCheckBox("Auto-tile subwindows")
        auto_tile_box.setToolTip("Set this to automatically tile all "
                                 "projection subwindows whenever a new one is "
                                 "added")
        auto_tile_box.toggled.connect(self.enable_save_button)
        self.options_entries['auto_tile'] = options_entry(auto_tile_box, True)
        group_layout.addRow(auto_tile_box)

    # AUTO_SHOW OPTIONS
    def add_option_auto_show(self, group_layout):
        # Make check box for auto showing projection figures/subwindows
        auto_show_box = QW.QCheckBox("Auto-show subwindows")
        auto_show_box.setToolTip("Set this to automatically show a projection "
                                 "subwindow after it has been drawn")
        auto_show_box.toggled.connect(self.enable_save_button)
        self.options_entries['auto_show'] = options_entry(auto_show_box, True)
        group_layout.addRow(auto_show_box)

    # PROJ_RES OPTION
    def add_option_proj_res(self, group_layout):
        # Make spinbox for option proj_res
        proj_res_box = QW_QSpinBox()
        proj_res_box.setRange(0, 9999999)
        proj_res_box.setToolTip(proj_res_doc)
        proj_res_box.valueChanged.connect(self.enable_save_button)
        self.options_entries['proj_res'] =\
            options_entry(proj_res_box, self.proj_defaults['proj_res'])
        group_layout.addRow('Resolution:', proj_res_box)

    # PROJ_DEPTH OPTION
    def add_option_proj_depth(self, group_layout):
        # Make spinbox for option proj_depth
        proj_depth_box = QW_QSpinBox()
        proj_depth_box.setRange(0, 9999999)
        proj_depth_box.setToolTip(proj_depth_doc)
        proj_depth_box.valueChanged.connect(self.enable_save_button)
        self.options_entries['proj_depth'] =\
            options_entry(proj_depth_box, self.proj_defaults['proj_depth'])
        group_layout.addRow('Depth:', proj_depth_box)

    # EMUL_I OPTION
    def add_option_emul_i(self, group_layout):
        # Make spinbox for option emul_i
        emul_i_box = QW_QSpinBox()
        emul_i_box.setRange(0, self.pipe._emulator._emul_i)
        emul_i_box.valueChanged.connect(self.enable_save_button)
        self.options_entries['emul_i'] =\
            options_entry(emul_i_box, self.proj_defaults['emul_i'])
        group_layout.addRow('Iteration:', emul_i_box)

    # PROJ_TYPE OPTION
    def add_option_proj_type(self, group_layout):
        # Make check boxes for 2D and 3D projections
        # 2D projections
        proj_2D_box = QW.QCheckBox('2D')
        proj_2D_box.setEnabled(self.n_par > 2)
        proj_2D_box.toggled.connect(self.enable_save_button)
        self.options_entries['proj_2D'] =\
            options_entry(proj_2D_box, self.proj_defaults['proj_2D'])

        # 3D projections
        proj_3D_box = QW.QCheckBox('3D')
        proj_3D_box.setEnabled(self.n_par > 2)
        proj_3D_box.toggled.connect(self.enable_save_button)
        self.options_entries['proj_3D'] =\
            options_entry(proj_3D_box, self.proj_defaults['proj_3D'])

        # Create layout for proj_type and add it to options layout
        proj_type_box = QW.QHBoxLayout()
        proj_type_box.addWidget(proj_2D_box)
        proj_type_box.addWidget(proj_3D_box)
        proj_type_box.addStretch()
        group_layout.addRow('Projection type:', proj_type_box)

    # ALIGN OPTION
    def add_option_align(self, group_layout):
        # Column align
        align_col_box = QW.QRadioButton('Column')
        align_col_box.setToolTip("Align the projection subplots in a single "
                                 "column")
        align_col_box.toggled.connect(self.enable_save_button)
        self.options_entries['align_col'] =\
            options_entry(align_col_box, self.proj_defaults['align'] == 'col')

        # Row align
        align_row_box = QW.QRadioButton('Row')
        align_row_box.setToolTip("Align the projection subplots in a single "
                                 "row")
        align_row_box.toggled.connect(self.enable_save_button)
        self.options_entries['align_row'] =\
            options_entry(align_row_box, self.proj_defaults['align'] == 'row')

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
        show_cuts_box.setToolTip("Enable/disable showing all implausibility "
                                 "cut-off lines in 2D projections")
        show_cuts_box.toggled.connect(self.enable_save_button)
        self.options_entries['show_cuts'] =\
            options_entry(show_cuts_box, self.proj_defaults['show_cuts'])
        group_layout.addRow('Show cuts?', show_cuts_box)

    # SMOOTH OPTION
    def add_option_smooth(self, group_layout):
        # Make check box for smooth
        smooth_box = QW.QCheckBox()
        smooth_box.setToolTip("Enable/disable smoothing the projections. When "
                              "smoothed, the minimum implausibility is forced "
                              "to be above the first cut-off for implausible "
                              "regions")
        smooth_box.toggled.connect(self.enable_save_button)
        self.options_entries['smooth'] =\
            options_entry(smooth_box, self.proj_defaults['smooth'])
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
            banned_entries=self.get_proj_attr('pop_fig_kwargs'))

        # IMPL_KWARGS_2D
        kwargs_dicts_box.add_dict(
            "2D implausibility", 'impl_kwargs_2D',
            std_entries=['linestyle', 'linewidth', 'marker', 'markersize',
                         'color', 'alpha'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs'), 'cmap'])

        # IMPL_KWARGS_3D
        kwargs_dicts_box.add_dict(
            "3D implausibility", 'impl_kwargs_3D',
            std_entries=['cmap', 'alpha', 'xscale', 'yscale'],
            banned_entries=self.get_proj_attr('pop_plt_kwargs'))

        # LOS_KWARGS_2D
        kwargs_dicts_box.add_dict(
            "2D line-of-sight", 'los_kwargs_2D',
            std_entries=['linestyle', 'linewidth', 'marker', 'markersize',
                         'color', 'alpha'],
            banned_entries=[*self.get_proj_attr('pop_plt_kwargs'), 'cmap'])

        # LOS_KWARGS_3D
        kwargs_dicts_box.add_dict(
            "3D line-of-sight", 'los_kwargs_3D',
            std_entries=['cmap', 'alpha', 'xscale', 'yscale'],
            banned_entries=self.get_proj_attr('pop_plt_kwargs'))

        # LINE_KWARGS_EST
        kwargs_dicts_box.add_dict(
            "Estimate lines", 'line_kwargs_est',
            std_entries=['linestyle', 'color', 'alpha', 'linewidth'],
            banned_entries=[])

        # LINE_KWARGS_CUT
        kwargs_dicts_box.add_dict(
            "Cut-off lines", 'line_kwargs_cut',
            std_entries=['linestyle', 'color', 'alpha', 'linewidth'],
            banned_entries=[])

    # BUTTONS GROUP
    def add_group_buttons(self, window_layout):
        # Create a button_box
        button_box = QW.QDialogButtonBox()
        window_layout.addWidget(button_box)

        # Make a 'Reset' button
        reset_but = button_box.addButton(button_box.Reset)
        reset_but.setToolTip("Reset to defaults")
        reset_but.clicked.connect(self.reset_options)

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