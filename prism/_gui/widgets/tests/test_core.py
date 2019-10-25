# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from mpi4pyd import MPI
from PyQt5 import QtWidgets as QW
import pytest

# PRISM imports
from prism._gui.widgets.core import (
     BaseBox, get_box_value, get_modified_box_signal, set_box_value)


# Skip this entire module for any rank that is not the controller
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.rank,
                                reason="Worker ranks cannot test this")


# %% GLOBALS
DIR_PATH = path.abspath(path.dirname(__file__))     # Path to tests directory


# %% CUSTOM CLASSES
class CustomBox(BaseBox):
    pass


class ProperBox(BaseBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0

    def get_box_value(self):
        return(self.value)

    def set_box_value(self, value):
        self.modified.emit()
        self.value = value


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for BaseBox base class
class TestBaseBox(object):
    # Test a BaseBox subclass box
    @pytest.fixture(scope='function')
    def box(self, qtbot):
        # Create a CustomBox instance
        box = CustomBox()
        qtbot.addWidget(box)

        # Return box
        return(box)

    # Test if initializing and using BaseBox directly fails
    def test_BaseBox_obj(self, qtbot):
        # Initialize BaseBox
        base_box = BaseBox()
        qtbot.addWidget(base_box)

        # Try calling the get_box_value method
        with pytest.raises(NotImplementedError):
            base_box.get_box_value()

        # Try calling the set_box_value method
        with pytest.raises(NotImplementedError):
            base_box.set_box_value('test')

    # Test if a QWidget can have a BaseBox box as its parent
    def test_BaseBox_parent(self, qtbot, box):
        # Create a layout for the CustomBox
        layout = QW.QHBoxLayout(box)

        # Create a checkbox child
        checkbox = QW.QCheckBox()
        layout.addWidget(checkbox)

        # Check if modifying checkbox automatically triggers modified signal
        with qtbot.waitSignal(box.modified):
            set_box_value(checkbox, True)

        # Check if attempting to set the signal again ignores it
        box.connect_box(checkbox)

    # Test if an unrelated box can connect its signal to a BaseBox
    def test_connect_modified_signal(self, qtbot, box):
        # Create a checkbox
        checkbox = QW.QCheckBox()
        qtbot.addWidget(checkbox)

        # Connect signals
        box.connect_box(checkbox)

        # Check if modifying checkbox automatically triggers modified signal
        with qtbot.waitSignal(box.modified):
            set_box_value(checkbox, True)


# Pytest for utility functions
class Test_functions(object):
    # Test if a spinbox can have its value set/returned properly
    def test_spinbox(self, qtbot):
        # Create a spinbox
        box = QW.QSpinBox()
        qtbot.addWidget(box)

        # Obtain modified signal of this box
        signal = get_modified_box_signal(box)

        # Set the value of this box
        with qtbot.waitSignal(signal):
            set_box_value(box, 50)

        # Check the value of this box
        assert (get_box_value(box) == 50)

    # Test if a button can have its value set/returned properly
    def test_button(self, qtbot):
        # Create a button
        box = QW.QCheckBox()
        qtbot.addWidget(box)

        # Obtain modified signal of this box
        signal = get_modified_box_signal(box)

        # Set the value of this box
        with qtbot.waitSignal(signal):
            set_box_value(box, True)

        # Check the value of this box
        assert get_box_value(box)

    # Test if a combobox can have its value set/returned properly
    def test_combobox(self, qtbot):
        # Create a combobox
        box = QW.QComboBox()
        box.addItems(['a', 'b', 'c'])
        qtbot.addWidget(box)

        # Obtain modified signal of this box
        signal = get_modified_box_signal(box)

        # Set the value of this box
        with qtbot.waitSignal(signal):
            set_box_value(box, 'b')

        # Check the value of this box
        assert (get_box_value(box) == 'b')

    # Test if an editable combobox can have its value set/returned properly
    def test_editable_combobox(self, qtbot):
        # Create a combobox
        box = QW.QComboBox()
        box.setEditable(True)
        box.addItems(['a', 'b', 'c'])
        qtbot.addWidget(box)

        # Obtain modified signal of this box
        signal = get_modified_box_signal(box)

        # Set the value of this box
        with qtbot.waitSignal(signal):
            set_box_value(box, 'd')

        # Check the value of this box
        assert (get_box_value(box) == 'd')

    # Test if a lineedit can have its value set/returned properly
    def test_lineedit(self, qtbot):
        # Create a lineedit
        box = QW.QLineEdit()
        qtbot.addWidget(box)

        # Obtain modified signal of this box
        signal = get_modified_box_signal(box)

        # Set the value of this box
        with qtbot.waitSignal(signal):
            set_box_value(box, "This is a test!")

        # Check the value of this box
        assert (get_box_value(box) == "This is a test!")

    # Test if a proper BaseBox can have its value set/returned properly
    def test_valid_BaseBox(self, qtbot):
        # Create a ProperBox
        box = ProperBox()
        qtbot.addWidget(box)

        # Obtain modified signal of this box
        signal = get_modified_box_signal(box)

        # Set the value of this box
        with qtbot.waitSignal(signal):
            set_box_value(box, 150)

        # Check the value of this box
        assert (get_box_value(box) == 150)

    # Test if an incorrect BaseBox cannot have its value set/returned
    def test_invalid_BaseBox(self, qtbot):
        # Create a CustomBox
        box = CustomBox()
        qtbot.addWidget(box)

        # Try to set the value of this box
        with pytest.raises(NotImplementedError):
            set_box_value(box, 150)

        # Try to get the value of this box
        with pytest.raises(NotImplementedError):
            get_box_value(box)

    # Test if an unsupported box cannot have its value set/returned
    def test_invalid_box(self, qtbot):
        # Create a label
        box = QW.QLabel()
        qtbot.addWidget(box)

        # Try to obtain the modified signal of this box
        with pytest.raises(NotImplementedError):
            get_modified_box_signal(box)

        # Try to set the value of this box
        with pytest.raises(NotImplementedError):
            set_box_value(box, 'test')

        # Try to get the value of this box
        with pytest.raises(NotImplementedError):
            get_box_value(box)
