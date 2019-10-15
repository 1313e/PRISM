# -*- coding: utf-8 -*-

"""
GUI Options Helpers
===================
Provides a collection of utility functions used for making the options window
in the Projection GUI.

"""


# %% IMPORTS
# Package imports
from PyQt5 import QtCore as QC, QtWidgets as QW

# All declaration
__all__ = ['BaseBox', 'get_box_value', 'get_modified_box_signal',
           'set_box_value']


# %% CLASS DEFINITIONS
# Make base class for custom boxes
# As QW.QWidget is a strict class (in C++), this cannot be an ABC
class BaseBox(QW.QWidget):
    # Define modified signal
    modified = QC.pyqtSignal()

    def __init__(self, *args, **kwargs):
        # Call super constructor
        super().__init__(*args, **kwargs)

    # Initialize base box
    def init(self):
        # Get the modified signal of this box
        signal = get_modified_box_signal(self)

        # If this signal is not the modified signal, connect the two signals
        if 'modified' not in signal.signal:
            signal.connect(self.modified)

    # Override childEvent to automatically connect signals if child is BaseBox
    def childEvent(self, event):
        # If this event involved a child being added, check child object
        if(event.type() == QC.QEvent.ChildAdded):
            # If added child is an instance of BaseBox, connect signals
            child = event.child()
            if isinstance(child, BaseBox):
                child.modified.connect(self.modified)

        # Call and return super method
        return(super().childEvent(event))

    # Define get_box_value method
    def get_box_value(self):
        raise NotImplementedError(self.__class__)

    # Define set_box_value method
    def set_box_value(self, value):
        raise NotImplementedError(self.__class__)


# %% FUNCTIONS DEFINITIONS
# This function gets the value of a provided box
def get_box_value(box):
    # Values (QAbstractSpinBox)
    if isinstance(box, QW.QAbstractSpinBox):
        return(box.value())
    # Bools (QAbstractButton)
    elif isinstance(box, QW.QAbstractButton):
        return(box.isChecked())
    # Items (QComboBox)
    elif isinstance(box, QW.QComboBox):
        return(box.currentText())
    # Strings (QLineEdit)
    elif isinstance(box, QW.QLineEdit):
        return(box.text())
    # Custom boxes (BaseBox)
    elif isinstance(box, BaseBox):
        return(box.get_box_value())
    # If none applies, raise error
    else:
        raise NotImplementedError("Custom boxes must be a subclass of BaseBox")


# This function gets the emitted signal when a provided box is modified
def get_modified_box_signal(box):
    # Values (QAbstractSpinBox)
    if isinstance(box, QW.QAbstractSpinBox):
        return(box.valueChanged)
    # Bools (QAbstractButton)
    elif isinstance(box, QW.QAbstractButton):
        return(box.toggled if box.isCheckable() else box.clicked)
    # Items (QComboBox)
    elif isinstance(box, QW.QComboBox):
        return(box.currentTextChanged)
    # Strings (QLineEdit)
    elif isinstance(box, QW.QLineEdit):
        return(box.textEdited)
    # Custom boxes (BaseBox)
    elif isinstance(box, BaseBox):
        return(box.modified)
    # If none applies, raise error
    else:
        raise NotImplementedError("Custom boxes must be a subclass of BaseBox")


# This function sets the value of a provided box
def set_box_value(box, value):
    # Values (QAbstractSpinBox)
    if isinstance(box, QW.QAbstractSpinBox):
        box.setValue(value)
    # Bools (QAbstractButton)
    elif isinstance(box, QW.QAbstractButton):
        box.setChecked(value)
    # Items (QComboBox)
    elif isinstance(box, QW.QComboBox):
        index = box.findText(value)
        if(index != -1):
            box.setCurrentIndex(index)
        else:
            box.setCurrentText(value)
    # Strings (QLineEdit)
    elif isinstance(box, QW.QLineEdit):
        box.setText(value)
    # Custom boxes (BaseBox)
    elif isinstance(box, BaseBox):
        box.set_box_value(value)
    # If none applies, raise error
    else:
        raise NotImplementedError("Custom boxes must be a subclass of BaseBox")
