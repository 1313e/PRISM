# -*- coding: utf-8 -*-

"""
GUI Base Widget Classes
=======================
Provides a collection of custom :class:`~PyQt5.QtWidgets.QWidget` base classes
that allow for certain widgets to be standardized.

"""


# %% IMPORTS
# Built-in imports
from sys import platform

# Package imports
from PyQt5 import QtCore as QC, QtWidgets as QW

# All declaration
__all__ = ['QW_QAction', 'QW_QComboBox', 'QW_QDoubleSpinBox',
           'QW_QEditableComboBox', 'QW_QLabel', 'QW_QMenu', 'QW_QSpinBox',
           'QW_QToolBar']


# %% CLASS DEFINITIONS
# Make subclass of QW.QAction that automatically sets details based on status
class QW_QAction(QW.QAction):
    # Override constructor
    def __init__(self, parent, text, *, shortcut=None, tooltip=None,
                 statustip=None, icon=None, triggered=None, role=None):
        # Call super constructor
        if icon is None:
            super().__init__(text, parent)
        else:
            super().__init__(icon, text, parent)

        # Set all the details
        self.setDetails(shortcut=shortcut,
                        tooltip=tooltip,
                        statustip=statustip)

        # Set the signal trigger
        if triggered is not None:
            self.triggered.connect(triggered)

        # Set the action menu role
        self.setMenuRole(self.NoRole if role is None else role)

    # Make new method that automatically sets Shortcut, ToolTip and StatusTip
    def setDetails(self, *, shortcut=None, tooltip=None, statustip=None):
        # If shortcut is not None, set it
        if shortcut is not None:
            super().setShortcut(shortcut)
            shortcut = self.shortcut().toString()

        # If tooltip is None, its base is set to the action's name
        if tooltip is None:
            base_tooltip = self.text().replace('&', '')
            tooltip = base_tooltip
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


# Create custom combobox class with more signals
class QW_QComboBox(QW.QComboBox):
    popup_shown = QC.pyqtSignal([int], [str])
    popup_hidden = QC.pyqtSignal([int], [str])

    # Override the showPopup to emit a signal whenever it is triggered
    def showPopup(self, *args, **kwargs):
        self.popup_shown[int].emit(self.currentIndex())
        self.popup_shown[str].emit(self.currentText())
        return(super().showPopup(*args, **kwargs))

    # Override the hidePopup to emit a signal whenever it is triggered.
    def hidePopup(self, *args, **kwargs):
        self.popup_hidden[int].emit(self.currentIndex())
        self.popup_hidden[str].emit(self.currentText())
        return(super().hidePopup(*args, **kwargs))


# Create custom QComboBox class that is editable
class QW_QEditableComboBox(QW_QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEditable(True)
        self.setInsertPolicy(self.NoInsert)
        self.completer().setCompletionMode(QW.QCompleter.PopupCompletion)
        self.completer().setFilterMode(QC.Qt.MatchContains)


# Create custom QAbstractSpinBox that automatically sets some properties
class QW_QAbstractSpinBox(QW.QAbstractSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStepType(self.AdaptiveDecimalStepType)
        self.setAccelerated(True)
        self.setGroupSeparatorShown(True)
        self.setStyleSheet(
            """
            QAbstractSpinBox {{
                margin: {0}px 0px {0}px 0px;
                max-height: 24px;}}
            """.format("-1" if platform.startswith('linux') else '0'))


# Create custom QDoubleSpinBox
class QW_QDoubleSpinBox(QW.QDoubleSpinBox, QW_QAbstractSpinBox):
    pass


# Create custom QSpinBox
class QW_QSpinBox(QW.QSpinBox, QW_QAbstractSpinBox):
    pass


# Create custom label class with more signals
class QW_QLabel(QW.QLabel):
    mousePressed = QC.pyqtSignal()

    # Override the mousePressEvent to emit a signal whenever it is triggered
    def mousePressEvent(self, event):
        self.mousePressed.emit()
        event.accept()


# Create custom QMenu class that swaps the order of inputs
class QW_QMenu(QW.QMenu):
    def __init__(self, parent, title):
        super().__init__(title, parent)


# Create custom QToolbar class that swaps the order of inputs
class QW_QToolBar(QW.QToolBar):
    def __init__(self, parent, window_title):
        super().__init__(window_title, parent)
