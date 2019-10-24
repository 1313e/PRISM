# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from mpi4pyd import MPI
from numpy.random import randint
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
import pytest

# PRISM imports
from prism._gui.widgets.core import get_box_value, set_box_value
from prism._gui.widgets.preferences.custom_boxes import (
    ColorBox, DefaultBox, FigSizeBox)


# Skip this entire module for any rank that is not the controller
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.rank,
                                reason="Worker ranks cannot test this")


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for ColorBox
class TestColorBox(object):
    # Test a color box
    @pytest.fixture(scope='function')
    def box(self, qtbot):
        # Create a ColorBox instance
        box = ColorBox()
        qtbot.addWidget(box)

        # Return box
        return(box)

    # Test if the proper boxes are present
    def test_boxes(self, box):
        # Test if there is a color combobox
        assert hasattr(box, 'color_combobox')
        assert isinstance(box.color_combobox, QW.QComboBox)

        # Test if there is a color label
        assert hasattr(box, 'color_label')
        assert isinstance(box.color_label, QW.QLabel)

    # Test the color picker
    def test_colorpicker(self, qtbot, monkeypatch, box):
        # Obtain a random hex color
        color = QG.QColor(hex(randint(256**3)).replace('0x', '#'))

        # Monkey patch the QColorDialog.getColor function
        monkeypatch.setattr(QW.QColorDialog, 'getColor',
                            lambda *args, **kwargs: color)

        # Use the colorpicker
        with qtbot.waitSignal(box.modified):
            qtbot.mouseClick(box.color_label, QC.Qt.LeftButton)

    # Test setting the color
    def test_set_color(self, qtbot, box):
        # Generate random color
        color = hex(randint(256**3)).replace('0x', '')

        # Remove the current value in the box
        qtbot.keyClick(box.color_combobox, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.color_combobox, QC.Qt.Key_Delete)

        # Try setting the color
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.color_combobox, color)

        # Check that this is now the color that is set
        assert (get_box_value(box) == "#%s" % (color))

    # Test getting an incorrect color
    def test_get_invalid_color(self, qtbot, box):
        # Obtain what currently the default color is
        assert hasattr(box, 'default_color')
        def_color = str(box.default_color)

        # Remove the current value in the box
        qtbot.keyClick(box.color_combobox, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.color_combobox, QC.Qt.Key_Delete)

        # Try setting the color to something invalid
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.color_combobox, "test")

        # Check if the colorbox now returns the default color
        assert (get_box_value(box) == def_color)

    # Test setting an invalid color label
    def test_set_invalid_color_label(self, box):
        # Try setting the color label to something invalid
        assert box.set_color_label("1.5")


# Pytest for DefaultBox
class TestDefaultBox(object):
    # Test a default box
    @pytest.fixture(scope='function')
    def box(self, qtbot):
        # Create a DefaultBox instance
        box = DefaultBox()
        qtbot.addWidget(box)

        # Return box
        return(box)

    # Test if the proper boxes are available
    def test_boxes(self, box):
        # Test if the type_box is present
        assert hasattr(box, 'type_box')
        assert isinstance(box.type_box, QW.QComboBox)

        # Test if the value_box is present
        assert hasattr(box, 'value_box')
        assert isinstance(box.value_box, QW.QWidget)

    # Test bool box
    def test_bool(self, qtbot, box):
        # Request a bool box
        set_box_value(box.type_box, 'bool')

        # Check that the value box is now a check box
        assert isinstance(box.value_box, QW.QCheckBox)

        # Try to toggle the value box
        with qtbot.waitSignal(box.modified):
            box.value_box.click()

        # Check that the value is now True
        assert get_box_value(box)

    # Test float box
    def test_float(self, qtbot, box):
        # Request a float box
        set_box_value(box.type_box, 'float')

        # Check that the value box is now a double spinbox
        assert isinstance(box.value_box, QW.QDoubleSpinBox)

        # Remove the current value in the box
        qtbot.keyClick(box.value_box, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.value_box, QC.Qt.Key_Delete)

        # Try to set the value of the spinbox
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.value_box, '13.13')

        # Check that the value is now 13.13
        assert (get_box_value(box) == 13.13)

    # Test integer box
    def test_int(self, qtbot, box):
        # Request an integer box
        set_box_value(box.type_box, 'int')

        # Check that the value box is now a spinbox
        assert isinstance(box.value_box, QW.QSpinBox)

        # Remove the current value in the box
        qtbot.keyClick(box.value_box, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.value_box, QC.Qt.Key_Delete)

        # Try to set the value of the spinbox
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.value_box, '100')

        # Check that the value is now 100
        assert (get_box_value(box) == 100)

    # Test string box
    def test_str(self, qtbot, box):
        # Request a string box
        set_box_value(box.type_box, 'str')

        # Check that the value box is now a line edit
        assert isinstance(box.value_box, QW.QLineEdit)

        # Remove the current value in the box
        qtbot.keyClick(box.value_box, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.value_box, QC.Qt.Key_Delete)

        # Try to set the value of the line edit
        text = "This iS a Te$t! ;)"
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.value_box, text)

        # Check that the value is correct
        assert (get_box_value(box) == text)

    # Test if setting the box value works properly
    def test_set_box_value(self, box):
        # Set the box value to a bool and check
        set_box_value(box, True)
        assert (get_box_value(box.type_box) == 'bool')

        # Set the box value to a float and check
        set_box_value(box, 13.13)
        assert (get_box_value(box.type_box) == 'float')

        # Set the box value to an integer and check
        set_box_value(box, 100)
        assert (get_box_value(box.type_box) == 'int')

        # Set the box value to a string and check
        set_box_value(box, "More Te3t1ng! -.-")
        assert (get_box_value(box.type_box) == 'str')


# Pytest for FigSizeBox
class TestFigSizeBox(object):
    # Test a figsize box
    @pytest.fixture(scope='function')
    def box(self, qtbot):
        # Create a FigSizeBox instance
        box = FigSizeBox()
        qtbot.addWidget(box)

        # Return box
        return(box)

    # Test if figsize box comes with two spinboxes
    def test_size_boxes(self, box):
        # Test presence of width box
        assert hasattr(box, 'width_box')
        assert isinstance(box.width_box, QW.QDoubleSpinBox)

        # Test presence of height box
        assert hasattr(box, 'height_box')
        assert isinstance(box.height_box, QW.QDoubleSpinBox)

    # Test setting the width
    def test_set_width(self, qtbot, box):
        # Remove the current value in the box
        qtbot.keyClick(box.width_box, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.width_box, QC.Qt.Key_Delete)

        # Try to set the value of the spinbox
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.width_box, '13.13')

        # Check that the value is now 13.13
        assert (get_box_value(box.width_box) == 13.13)

    # Test setting the height
    def test_set_height(self, qtbot, box):
        # Remove the current value in the box
        qtbot.keyClick(box.height_box, QC.Qt.Key_A, QC.Qt.ControlModifier)
        qtbot.keyClick(box.height_box, QC.Qt.Key_Delete)

        # Try to set the value of the spinbox
        with qtbot.waitSignal(box.modified):
            qtbot.keyClicks(box.height_box, '42.10')

        # Check that the value is now 42.10
        assert (get_box_value(box.height_box) == 42.10)

    # Test setting both simultaneously
    def test_set_sizes(self, box):
        # Set the value of the figsize box
        set_box_value(box, (13.13, 42.10))

        # Check that this is now the current value of the box
        assert (get_box_value(box) == (13.13, 42.10))
