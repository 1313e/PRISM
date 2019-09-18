# -*- coding: utf-8 -*-

"""
GUI Custom Option Boxes
=======================
Provides a collection of :class:`~PyQt5.QtWidgets.QWidget` subclasses to be
used as custom option entry boxes in the
:class:`~prism._gui.widgets.preferences.OptionsDialog` class or
:class:`~prism._gui.widgets.preferences.KwargsDictDialogPage` class.

"""


# %% IMPORTS
# Package imports
from PyQt5 import QtWidgets as QW

# PRISM imports
from prism._gui.widgets.preferences.helpers import get_box_value, set_box_value

# All declaration
__all__ = ['DefaultBox', 'FigSizeBox']


# %% CLASS DEFINITIONS
# Make class for the default lineedit box that allows for type to be selected
class DefaultBox(QW.QWidget):
    def __init__(self, options_dialog_obj, *args, **kwargs):
        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the default box
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
            bool: 'bool',
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

    # This function creates the value box for bools
    def add_type_bool(self):
        # Create a checkbox for bools
        bool_box = QW.QCheckBox()
        bool_box.setToolTip("Boolean value for this entry type")
        bool_box.stateChanged.connect(self.options.enable_save_button)
        return(bool_box)

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
        width_box.setToolTip("Width (in inches) of projection figure")
        width_box.valueChanged.connect(self.options.enable_save_button)
        self.width_box = width_box

        # HEIGHT
        height_box = QW.QDoubleSpinBox()
        height_box.setRange(1, 9999999)
        height_box.setSingleStep(0.1)
        height_box.setToolTip("Height (in inches) of projection figure")
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
