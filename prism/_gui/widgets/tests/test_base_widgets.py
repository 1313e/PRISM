# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from mpi4pyd import MPI
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
import pytest

# PRISM imports
from prism._gui import APP_ICON_PATH
from prism._gui.widgets.base_widgets import (
    QW_QAction, QW_QComboBox, QW_QDoubleSpinBox, QW_QEditableComboBox,
    QW_QLabel, QW_QMenu, QW_QSpinBox, QW_QToolBar)


# Skip this entire module for any rank that is not the controller
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.rank,
                                reason="Worker ranks cannot test this")


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for custom QAction
class TestQW_QAction(object):
    # Test default action
    def test_default(self, qtbot):
        # Create action
        action = QW_QAction(None, 'test')
        qtbot.addWidget(action)

    # Test with all optional arguments
    def test_args(self, qtbot):
        # Create action
        action = QW_QAction(
            None, 'test',
            shortcut=QC.Qt.CTRL + QC.Qt.Key_T,
            tooltip="This is a test action",
            statustip="This is a test statustip",
            icon=QG.QIcon(APP_ICON_PATH),
            triggered=print)
        qtbot.addWidget(action)

    # Test if setting the shortcut manually raises an error
    def test_set_shortcut(self, qtbot):
        # Create action
        action = QW_QAction(None, 'test')
        qtbot.addWidget(action)

        # Try to set the shortcut
        with pytest.raises(AttributeError):
            action.setShortcut()

    # Test if setting the tooltip manually raises an error
    def test_set_tooltip(self, qtbot):
        # Create action
        action = QW_QAction(None, 'test')
        qtbot.addWidget(action)

        # Try to set the shortcut
        with pytest.raises(AttributeError):
            action.setToolTip()

    # Test if setting the statustip manually raises an error
    def test_set_statustip(self, qtbot):
        # Create action
        action = QW_QAction(None, 'test')
        qtbot.addWidget(action)

        # Try to set the shortcut
        with pytest.raises(AttributeError):
            action.setStatusTip()


# Pytest for custom QComboBox
class TestQW_QComboBox(object):
    # Test new combobox signals
    def test_signals(self, qtbot):
        # Create combobox
        combobox = QW_QComboBox()
        qtbot.addWidget(combobox)

        # Test signals
        with qtbot.waitSignal(combobox.popup_shown):
            combobox.showPopup()
        with qtbot.waitSignal(combobox.popup_hidden):
            combobox.hidePopup()


# Pytest for custom editable QComboBox
class TestQW_QEditableComboBox(object):
    # Test default
    def test_default(self, qtbot):
        # Create combobox
        combobox = QW_QEditableComboBox()
        qtbot.addWidget(combobox)

        # Check its settings
        assert combobox.isEditable()
        assert (combobox.insertPolicy() == QW.QComboBox.NoInsert)

    # Test signals (same as for QW_QComboBox)
    def test_signals(self, qtbot):
        # Create combobox
        combobox = QW_QEditableComboBox()
        qtbot.addWidget(combobox)

        # Test signals
        with qtbot.waitSignal(combobox.popup_shown):
            combobox.showPopup()
        with qtbot.waitSignal(combobox.popup_hidden):
            combobox.hidePopup()


# Pytest for custom QDoubleSpinBox
class TestQW_QDoubleSpinBox(object):
    # Test default
    def test_default(self, qtbot):
        # Create doublespinbox
        spinbox = QW_QDoubleSpinBox()
        qtbot.addWidget(spinbox)

        # Check its settings
        assert (spinbox.stepType() == QW.QSpinBox.AdaptiveDecimalStepType)
        assert spinbox.isAccelerated()
        assert spinbox.isGroupSeparatorShown()


# Pytest for custom QSpinBox
class TestQW_QSpinBox(object):
    # Test default
    def test_default(self, qtbot):
        # Create spinbox
        spinbox = QW_QSpinBox()
        qtbot.addWidget(spinbox)

        # Check its settings
        assert (spinbox.stepType() == QW.QSpinBox.AdaptiveDecimalStepType)
        assert spinbox.isAccelerated()
        assert spinbox.isGroupSeparatorShown()


# Pytest for custom QLabel
class TestQW_QLabel(object):
    # Test signals
    def test_signals(self, qtbot):
        # Create label
        label = QW_QLabel()
        qtbot.addWidget(label)

        # Test signals
        with qtbot.waitSignal(label.mousePressed):
            qtbot.mouseClick(label, QC.Qt.LeftButton)


# Pytest for custom QMenu
class TestQW_QMenu(object):
    # Test default
    def test_default(self, qtbot):
        menu = QW_QMenu(None, "test")
        qtbot.addWidget(menu)

        # Check title
        assert (menu.title() == 'test')


# Pytest for custom QToolBar
class TestQW_QToolBar(object):
    # Test default
    def test_default(self, qtbot):
        toolbar = QW_QToolBar(None, "test")
        qtbot.addWidget(toolbar)

        # Check window title
        assert (toolbar.windowTitle() == 'test')
