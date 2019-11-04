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
from e13tools.utils import docstring_substitute
from matplotlib import rcParams
from matplotlib.colors import BASE_COLORS, CSS4_COLORS, to_rgba
import numpy as np
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW

# PRISM imports
from prism._docstrings import kwargs_doc, qt_slot_doc
from prism._gui.widgets import (
    BaseBox, QW_QComboBox, QW_QDoubleSpinBox, QW_QEditableComboBox, QW_QLabel,
    QW_QSpinBox, get_box_value, set_box_value)

# All declaration
__all__ = ['ColorBox', 'DefaultBox', 'FigSizeBox']


# %% CLASS DEFINITIONS
# Make class with a special box for setting the color of a plotted line
class ColorBox(BaseBox):
    """
    Defines the :class:`~ColorBox` class.

    This class is used for making the 'color' entry in the
    :class:`~prism._gui.widgets.preferences.KwargsDictDialogPage` class.

    """

    @docstring_substitute(optional=kwargs_doc.format(
        'prism._gui.widgets.core.BaseBox'))
    def __init__(self, *args, **kwargs):
        """
        Initialize an instance of the :class:`~ColorBox` class.

        %(optional)s

        """

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the color box
        self.init()

    # This function creates the color box
    def init(self):
        """
        Sets up the color box entry after it has been initialized.

        This function is mainly responsible for creating the color wheel and
        color label, that allow the user to quickly cycle through different
        color options.

        """

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
        """
        Creates a special label that shows the currently selected or hovered
        color, and returns it.

        """

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
        """
        Creates a combobox that holds all default colors accepted by matplotlib
        and returns it.

        """

        # Obtain the CN colors
        n_cyclic = len(rcParams['axes.prop_cycle'])
        CN_COLORS = [("C%i" % (i), "This is MPL cyclic color #%i" % (i))
                     for i in range(n_cyclic)]

        # Make tuple of all colors
        colors = (CN_COLORS, BASE_COLORS, CSS4_COLORS)

        # Determine the cumulative lengths of all four sets
        cum_len = np.cumsum(list(map(len, colors)))

        # Make combobox for colors
        color_box = QW_QEditableComboBox()

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
        color_box.highlighted[str].connect(self.set_color_label)
        color_box.popup_hidden[str].connect(self.set_color_label)
        color_box.currentTextChanged.connect(self.set_color)
        return(color_box)

    # This function converts an MPL color to a QColor
    @staticmethod
    def convert_to_qcolor(color):
        """
        Converts a provided matplotlib color `color` to a
        :obj:`~PyQt5.QtGui.QColor` object.

        Parameters
        ----------
        color : str
            The matplotlib color that must be converted.
            If `color` is a float string, an error will be raised, as Qt5 does
            not accept those.

        Returns
        -------
        qcolor : :obj:`~PyQt5.QtGui.QColor` object
            The instance of the :class:`~PyQt5.QtGui.QColor` class that
            corresponds to the provided `color`.

        """

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
    @staticmethod
    def convert_to_mpl_color(qcolor):
        """
        Converts a provided :obj:`~PyQt5.QtGui.QColor` object `color` to a
        matplotlib color.

        Parameters
        ----------
        qcolor : :obj:`~PyQt5.QtGui.QColor` object
            The instance of the :class:`~PyQt5.QtGui.QColor` class must be
            converted to a matplotlib color.

        Returns
        -------
        color : str
            The corresponding matplotlib color.
            The returned `color` is always written in HEX.

        """

        hexid = qcolor.name()
        return str(hexid)

    # This function creates a pixmap of an MPL color
    @staticmethod
    def create_color_pixmap(color, size):
        """
        Creates a :obj:`~PyQt5.QtGui.QPixmap` object consisting of the given
        `color` with the provided `size`.

        Parameters
        ----------
        color : str
            The matplotlib color that must be used for the pixmap.
        size : tuple
            The width and height dimension values of the pixmap to be created.

        Returns
        -------
        pixmap : :obj:`~PyQt5.QtGui.QPixmap` object
            The instance of the :class:`~PyQt5.QtGui.QPixmap` class that was
            created from the provided `color` and `size`.

        """

        # Obtain the RGBA values of an MPL color
        color = ColorBox.convert_to_qcolor(color)

        # Create an image object
        image = QG.QImage(*size, QG.QImage.Format_RGB32)

        # Fill the entire image with the same color
        image.fill(color)

        # Convert the image to a pixmap
        pixmap = QG.QPixmap.fromImage(image)

        # Return the pixmap
        return(pixmap)

    # This function shows the custom color picker dialog
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def show_colorpicker(self):
        """
        Shows the colorwheel picker dialog to the user, allowing for any color
        option to be selected.

        %(qt_slot)s

        """

        # Obtain current qcolor
        qcolor = self.convert_to_qcolor(self.get_box_value())

        # Show color dialog
        color = QW.QColorDialog.getColor(
            qcolor, parent=self,
            options=QW.QColorDialog.DontUseNativeDialog)

        # If the returned color is valid, save it
        if color.isValid():
            self.set_color(self.convert_to_mpl_color(color))

    # This function sets a given color as the current color
    @QC.pyqtSlot(str)
    @docstring_substitute(qt_slot=qt_slot_doc)
    def set_color(self, color):
        """
        Sets the current color to the provided `color`, and updates the entry
        in the combobox and the label accordingly.

        %(qt_slot)s

        Parameters
        ----------
        color : str
            The color that needs to be used as the current color. The provided
            `color` can be any string that is accepted as a color by
            matplotlib.
            If `color` is invalid, it is set to the current default color
            instead.

        """

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
    @QC.pyqtSlot(str)
    @docstring_substitute(qt_slot=qt_slot_doc)
    def set_color_label(self, color):
        """
        Sets the current color label to the provided `color`.

        %(qt_slot)s

        Parameters
        ----------
        color : str
            The color that needs to be used as the current color label. The
            provided `color` can be any string that is accepted as a color by
            matplotlib.
            If `color` is invalid, it is set to the current default color
            instead.

        Returns
        -------
        default_flag : bool
            Whether or not the color label is currently set to the default
            color. This happens when `color` is an invalid color.

        """

        # Try to create the pixmap of the colorlabel
        try:
            pixmap = self.create_color_pixmap(color,
                                              (70,
                                               self.color_combobox.height()-2))
            default_flag = False
        # If that cannot be done, create the default instead
        except ValueError:
            pixmap = self.create_color_pixmap(self.default_color,
                                              (70,
                                               self.color_combobox.height()-2))
            default_flag = True

        # Set the colorlabel
        self.color_label.setPixmap(pixmap)

        # Return if default was used or not
        return(default_flag)

    # This function retrieves a value of this special box
    def get_box_value(self):
        """
        Returns the current (valid) color value of the color combobox.

        Returns
        -------
        color : str
            The current valid matplotlib color value.

        """

        # Obtain the value
        color = get_box_value(self.color_combobox)

        # Try to convert this to QColor
        try:
            self.convert_to_qcolor(color)
        # If this fails, return the default color
        except ValueError:
            return(self.default_color)
        # Else, return the retrieved color
        else:
            return(color)

    # This function sets the value of this special box
    def set_box_value(self, value):
        """
        Sets the current (default) color value to `value`.

        Parameters
        ----------
        value : str
            The matplotlib color value that must be set for this colorbox.

        """

        self.set_color(value)
        self.default_color = value
        self.color_combobox.lineEdit().setPlaceholderText(value)


# Make class for the default non-standard box that allows type to be selected
class DefaultBox(BaseBox):
    """
    Defines the :class:`~ColorBox` class.

    This class is used for making a non-standard entry in the
    :class:`~prism._gui.widgets.preferences.KwargsDictDialogPage` class.
    It currently supports inputs of type bool; float; int; and str.

    """

    @docstring_substitute(optional=kwargs_doc.format(
        'prism._gui.widgets.core.BaseBox'))
    def __init__(self, *args, **kwargs):
        """
        Initialize an instance of the :class:`~DefaultBox` class.

        %(optional)s

        """

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the default box
        self.init()

    # This function creates a double box with type and lineedit
    def init(self):
        """
        Sets up the non-standard default box entry after it has been
        initialized.

        This function is mainly responsible for creating the type combobox and
        allowing for different field boxes to be used for different value
        types.

        """

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
        type_box = QW_QComboBox()
        type_box.addItems(self.type_dict.values())
        type_box.setToolTip("Type of the entered value")
        type_box.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        type_box.currentTextChanged.connect(self.create_field_box)
        self.type_box = type_box

        # Make value box corresponding to the current type
        val_box = getattr(self, "create_type_%s" % (type_box.currentText()))()
        self.value_box = val_box

        # Add everything to the box_layout
        box_layout.addWidget(type_box)
        box_layout.addWidget(val_box)

    # This function creates a field_box depending on the type that was selected
    @QC.pyqtSlot(str)
    @docstring_substitute(qt_slot=qt_slot_doc)
    def create_field_box(self, value_type):
        """
        Creates a field box for the provided type `value_type` and replaces the
        current field box with it.

        %(qt_slot)s

        Parameters
        ----------
        value_type : {'bool'; 'float'; 'int'; 'str'}
            The string that defines what type of field box is requested.

        """

        # Create a widget box for the specified value_type
        value_box = getattr(self, "create_type_%s" % (value_type))()

        # Set this value_box in the layout
        cur_item = self.box_layout.replaceWidget(self.value_box, value_box)
        cur_item.widget().close()
        del cur_item

        # Save new value_box
        self.value_box = value_box

    # This function creates the value box for bools
    def create_type_bool(self):
        """
        Creates the field box for values of type 'bool' and returns it.

        """

        # Create a checkbox for bools
        bool_box = QW.QCheckBox()
        bool_box.setToolTip("Boolean value for this entry type")
        return(bool_box)

    # This function creates the value box for floats
    def create_type_float(self):
        """
        Creates the field box for values of type 'float' and returns it.

        """

        # Create a spinbox for floats
        float_box = QW_QDoubleSpinBox()
        float_box.setRange(-9999999, 9999999)
        float_box.setDecimals(6)
        float_box.setToolTip("Float value for this entry type")
        return(float_box)

    # This function creates the value box for integers
    def create_type_int(self):
        """
        Creates the field box for values of type 'int' and returns it.

        """

        # Create a spinbox for integers
        int_box = QW_QSpinBox()
        int_box.setRange(-9999999, 9999999)
        int_box.setToolTip("Integer value for this entry type")
        return(int_box)

    # This function creates the value box for strings
    def create_type_str(self):
        """
        Creates the field box for values of type 'str' and returns it.

        """

        # Create a lineedit for strings
        str_box = QW.QLineEdit()
        str_box.setToolTip("String value for this entry type")
        return(str_box)

    # This function retrieves a value of this special box
    def get_box_value(self):
        """
        Returns the current value of the field box.

        Returns
        -------
        value : bool, float, int or str
            The current value of this default box.

        """

        return(get_box_value(self.value_box))

    # This function sets the value of this special box
    def set_box_value(self, value):
        """
        Sets the value type to `type(value)` and the field value to `value`.

        Parameters
        ----------
        value : bool, float, int or str
            The value to use for this default box. The type of `value`
            determines which field box must be used.

        """

        set_box_value(self.type_box, self.type_dict[type(value)])
        set_box_value(self.value_box, value)


# Make class with a special box for setting the figsize
class FigSizeBox(BaseBox):
    """
    Defines the :class:`~FigSizeBox` class.

    This class is used for making the 'figsize' entry in the
    :class:`~prism._gui.widgets.preferences.KwargsDictDialogPage` class.

    """

    @docstring_substitute(optional=kwargs_doc.format(
        'prism._gui.widgets.core.BaseBox'))
    def __init__(self, *args, **kwargs):
        """
        Initialize an instance of the :class:`~FigSizeBox` class.

        %(optional)s

        """

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the figsize box
        self.init()

    # This function creates the figsize box
    def init(self):
        """
        Sets up the figure size entry after it has been initialized.

        This function is mainly responsible for simply creating the two double
        spinboxes that allow for the width and height to be set.

        """

        # Create the box_layout
        box_layout = QW.QHBoxLayout(self)
        box_layout.setContentsMargins(0, 0, 0, 0)
        self.setToolTip("Figure size dimensions to use for the projection "
                        "figure")

        # Create two double spinboxes for the width and height
        # WIDTH
        width_box = QW_QDoubleSpinBox()
        width_box.setRange(1, 9999999)
        width_box.setSuffix("'")
        width_box.setStepType(width_box.AdaptiveDecimalStepType)
        width_box.setToolTip("Width (in inches) of projection figure")
        self.width_box = width_box

        # HEIGHT
        height_box = QW_QDoubleSpinBox()
        height_box.setRange(1, 9999999)
        height_box.setSuffix("'")
        height_box.setToolTip("Height (in inches) of projection figure")
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
        """
        Returns the current width and height of this figsize box and returns
        it.

        Returns
        -------
        figsize : tuple
            A tuple containing the width and height values of the figsize,
            formatted as `(width, height)`.

        """

        return((get_box_value(self.width_box), get_box_value(self.height_box)))

    # This function sets the value of this special box
    def set_box_value(self, value):
        """
        Sets the current value of the figsize box to `value`.

        Parameters
        ----------
        value : tuple
            A tuple containing the width and height values of the figsize,
            formatted as `(width, height)`.

        """

        set_box_value(self.width_box, value[0])
        set_box_value(self.height_box, value[1])
