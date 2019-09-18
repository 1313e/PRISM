# -*- coding: utf-8 -*-

"""
GUI Options Helpers
===================
Provides a collection of utility functions used for making the options window
in the Projection GUI.

"""


# %% IMPORTS
# Built-in imports
from collections import namedtuple
from functools import partial

# Package imports
from PyQt5 import QtWidgets as QW

# All declaration
__all__ = ['get_box_value', 'options_entry', 'set_box_value']


# %% FUNCTIONS DEFINITIONS
# This function gets the value of a provided widget_box
def get_box_value(widget_box):
    # Values (QAbstractSpinBox)
    if isinstance(widget_box, QW.QAbstractSpinBox):
        return(widget_box.value())
    # Bools (QAbstractButton)
    elif isinstance(widget_box, QW.QAbstractButton):
        return(widget_box.isChecked())
    # Items (QComboBox)
    elif isinstance(widget_box, QW.QComboBox):
        return(widget_box.currentText())
    # Strings (QLineEdit)
    elif isinstance(widget_box, QW.QLineEdit):
        return(widget_box.text())
    # Custom boxes (preferences.custom_boxes.xxx, KwargsDictDialogPage)
    else:
        # Try to obtain the value directly from the box
        try:
            return(widget_box.get_box_value())
        # If that does not work, raise a NotImplementedError
        except AttributeError:
            raise NotImplementedError(widget_box.__class__)


# This function sets the value of a provided widget_box
def set_box_value(widget_box, value):
    # Values (QAbstractSpinBox)
    if isinstance(widget_box, QW.QAbstractSpinBox):
        widget_box.setValue(value)
    # Bools (QAbstractButton)
    elif isinstance(widget_box, QW.QAbstractButton):
        widget_box.setChecked(value)
    # Items (QComboBox)
    elif isinstance(widget_box, QW.QComboBox):
        index = widget_box.findText(value)
        if(index != -1):
            widget_box.setCurrentIndex(index)
        else:
            widget_box.setCurrentText(value)
    # Strings (QLineEdit)
    elif isinstance(widget_box, QW.QLineEdit):
        widget_box.setText(value)
    # Custom boxes (preferences.custom_boxes.xxx, KwargsDictDialogPage)
    else:
        # Try to set the value directly to the box
        try:
            widget_box.set_box_value(value)
        # If that does not work, raise a NotImplementedError
        except AttributeError:
            raise NotImplementedError(widget_box.__class__)


# %% OTHER DEFINITIONS
# Create a named_tuple to be used for options entries
options_entry = namedtuple('options_entry', ['box', 'default', 'value'])
options_entry = partial(options_entry, value=None)
