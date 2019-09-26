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
# Built-in imports
from itertools import chain

# Package imports
from matplotlib import rcParamsDefault as rcParams
from matplotlib.colors import BASE_COLORS, CSS4_COLORS, to_rgba
import numpy as np
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW

# PRISM imports
from prism._gui.widgets import QW_QComboBox, QW_QLabel
from prism._gui.widgets.preferences.helpers import get_box_value, set_box_value

# All declaration
__all__ = ['ColorBox', 'DefaultBox', 'FigSizeBox']


# %% CLASS DEFINITIONS
# Make class with a special box for setting the color of a plotted line
class ColorBox(QW.QWidget):
    def __init__(self, options_dialog_obj, *args, **kwargs):
        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the color box
        self.init()

    # This function creates the color box
    def init(self):
        # Create the box layout
        box_layout = QW.QHBoxLayout(self)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self.setToolTip("Color to be used for the corresponding plot type")

        # Declare the default color
        self.default_color = rcParams['lines.color']

        # Create a color label
        color_label = self.create_color_label()
        self.color_label = color_label
        box_layout.addWidget(color_label)

        # Create a color combobox
        color_combobox = self.create_color_combobox()
        box_layout.addWidget(color_combobox)
        self.color_combobox = color_combobox

        # Set the starting color of the color box
        self.set_box_value(self.default_color)

    # This function creates the color label
    def create_color_label(self):
        # Create the color label
        color_label = QW_QLabel()

        # Set some properties
        color_label.setFrameShape(QW.QFrame.StyledPanel)
        color_label.setScaledContents(True)
        color_label.setToolTip("Click to open the custom color picker")
        color_label.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        color_label.mousePressed.connect(self.show_colorpicker)

        # Return it
        return(color_label)

    # This function creates the color combobox
    def create_color_combobox(self):
        # Obtain the CN colors
        n_cyclic = len(rcParams['axes.prop_cycle'])
        CN_COLORS = [("C%i" % (i), "This is MPL cyclic color #%i" % (i))
                     for i in range(n_cyclic)]

        # Make tuple of all colors
        colors = (CN_COLORS, BASE_COLORS, CSS4_COLORS)

        # Determine the cumulative lengths of all four sets
        cum_len = np.cumsum(list(map(len, colors)))

        # Make combobox for colors
        color_box = QW_QComboBox()

        # Fill combobox with all colors
        for i, color in enumerate(chain(*colors)):
            # If color is a tuple, it consists of (color, tooltip)
            if isinstance(color, tuple):
                color_box.addItem(color[0])
                color_box.setItemData(i, color[1], QC.Qt.ToolTipRole)
            else:
                color_box.addItem(color)

        # Add some separators
        for i in reversed(cum_len[:-1]):
            color_box.insertSeparator(i)

        # Set remaining properties
        color_box.setToolTip("Select or type (in HEX) the color")
        color_box.setEditable(True)
        color_box.setInsertPolicy(QW.QComboBox.NoInsert)
        color_box.completer().setCompletionMode(QW.QCompleter.PopupCompletion)
        color_box.highlighted[str].connect(self.set_color_label)
        color_box.popup_hidden[str].connect(self.set_color_label)
        color_box.currentTextChanged.connect(self.options.enable_save_button)
        color_box.currentTextChanged.connect(self.set_color)
        return(color_box)

    # This function shows the custom color picker dialog
    def show_colorpicker(self):
        # Obtain current qcolor
        qcolor = convert_to_qcolor(self.get_box_value())

        # Show color dialog
        color = QW.QColorDialog.getColor(
            qcolor, parent=self,
            options=QW.QColorDialog.DontUseNativeDialog)

        # If the returned color is valid, save it
        if color.isValid():
            self.set_color(convert_to_mpl_color(color))

    # This function sets a given color as the current color
    def set_color(self, color):
        # If color can be converted to a hex integer, do so and add hash to it
        try:
            int(color, 16)
        except ValueError:
            pass
        else:
            # Make sure that color has a length of 6
            if(len(color) == 6):
                color = "#%s" % (color)

        # Set the color label
        default_flag = self.set_color_label(color)

        # If default was not used, set the combobox to the proper value as well
        if not default_flag:
            set_box_value(self.color_combobox, color)

    # This function sets the color of the colorlabel
    def set_color_label(self, color):
        # Try to create the pixmap of the colorlabel
        try:
            pixmap = create_color_pixmap(color,
                                         (70, self.color_combobox.height()-2))
            default_flag = False
        # If that cannot be done, create the default instead
        except ValueError:
            pixmap = create_color_pixmap(self.default_color,
                                         (70, self.color_combobox.height()-2))
            default_flag = True

        # Set the colorlabel
        self.color_label.setPixmap(pixmap)

        # Return if default was used or not
        return(default_flag)

    # This function retrieves a value of this special box
    def get_box_value(self):
        # Obtain the value
        color = get_box_value(self.color_combobox)

        # Try to convert this to QColor
        try:
            convert_to_qcolor(color)
        # If this fails, return the default color
        except ValueError:
            return(self.default_color)
        # Else, return the retrieved color
        else:
            return(color)

    # This function sets the value of this special box
    def set_box_value(self, value):
        self.set_color(value)
        self.default_color = value
        self.color_combobox.lineEdit().setPlaceholderText(value)


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

        # Set default value
        set_box_value(self, rcParams['figure.figsize'])

    # This function retrieves a value of this special box
    def get_box_value(self):
        return((get_box_value(self.width_box), get_box_value(self.height_box)))

    # This function sets the value of this special box
    def set_box_value(self, value):
        set_box_value(self.width_box, value[0])
        set_box_value(self.height_box, value[1])


# %% FUNCTION DEFINITIONS
# This function converts an MPL color to a QColor
def convert_to_qcolor(color):
    # If the color can be converted to a float, raise a ValueError
    # This is because MPL accepts float strings as valid colors
    try:
        float(color)
    except ValueError:
        pass
    else:
        raise ValueError

    # Obtain the RGBA values of an MPL color
    r, g, b, a = to_rgba(color)

    # Convert to Qt RGBA values
    color = QG.QColor(
        int(r*255),
        int(g*255),
        int(b*255),
        int(a*255))

    # Return color
    return(color)


# This function converts a QColor to an MPL color
def convert_to_mpl_color(qcolor):
    hexid = qcolor.name()
    return str(hexid)


# This function creates a pixmap of an MPL color
def create_color_pixmap(color, size):
    # Obtain the RGBA values of an MPL color
    color = convert_to_qcolor(color)

    # Create an image object
    image = QG.QImage(*size, QG.QImage.Format_RGB32)

    # Fill the entire image with the same color
    image.fill(color)

    # COnvert the image to a pixmap
    pixmap = QG.QPixmap.fromImage(image)

    # Return the pixmap
    return(pixmap)
