# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import logging
from os import path
from sys import platform
import warnings

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from prism.__version__ import __version__, compat_version
from prism._internal import (FeatureWarning, PRISM_Logger, RequestError,
                             RequestWarning, check_vals, check_compatibility,
                             get_PRISM_File, get_info, getCLogger, getRLogger,
                             np_array)

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save if this platform is Windows
win32 = platform.startswith('win')


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for FeatureWarning
def test_FeatureWarning():
    with pytest.warns(FeatureWarning):
        warnings.warn("WARNING", FeatureWarning)


# Pytest for PRISM_File class
class Test_PRISM_File(object):
    def test(self, tmpdir):
        # Set _hdf5_file property to something default
        File = get_PRISM_File(path.join(tmpdir.strpath, 'test.hdf5'))

        # Check if opening master file works
        filename = path.join(tmpdir.strpath, 'test.hdf5')
        with File('w') as file:
            assert path.basename(file.filename) == 'test.hdf5'
        assert path.exists(filename)

        # Check if opening emulator system file works
        filename = path.join(tmpdir.strpath, 'test_0.hdf5')
        with File('w', emul_s=0) as file:
            assert path.basename(file.filename) == 'test_0.hdf5'
        assert path.exists(filename)


# Pytest for the check_compatibility function
def test_check_compatibility():
    # Check if current version is compatible (it should)
    check_compatibility(__version__)

    # Check if providing too old or too new version raises an error
    with pytest.raises(RequestError):
        check_compatibility(compat_version[-1])
    with pytest.raises(RequestError):
        check_compatibility('999.999.999')

    # Check if providing a v1.0.x version raises a warning
    with pytest.warns(RequestWarning):
        check_compatibility('1.0.0')


# Pytest for check_val function
class Test_check_val(object):
    # Check if all the type checks work correctly
    def test_type(self):
        # Test for bool
        assert check_vals(0, 'bool', 'bool') == 0
        assert check_vals(1, 'bool', 'bool') == 1
        assert check_vals(False, 'bool', 'bool') == 0
        assert check_vals(True, 'bool', 'bool') == 1
        assert check_vals("0", 'bool', 'bool') == 0
        assert check_vals("1", 'bool', 'bool') == 1
        assert check_vals("False", 'bool', 'bool') == 0
        assert check_vals("True", 'bool', 'bool') == 1
        with pytest.raises(TypeError):
            check_vals(2, 'int', 'bool')

        # Test for str
        assert check_vals('str', 'str', 'str') == 'str'
        assert check_vals(u'str', 'str', 'str') == u'str'
        with pytest.raises(TypeError):
            check_vals(1, 'int', 'str')

        # Test for complex
        assert check_vals(1, 'int', 'complex') == 1.0+0j
        assert check_vals(1., 'float', 'complex') == 1.0+0j
        assert check_vals(1.1, 'float', 'complex') == 1.1+0j
        assert check_vals(1j, 'complex', 'complex') == 0.0+1.0j
        assert check_vals(1.0j, 'complex', 'complex') == 0.0+1.0j
        assert check_vals(1.1-0.5j, 'complex', 'complex') == 1.1-0.5j
        with pytest.raises(TypeError):
            check_vals('1', 'str', 'complex')

        # Test for float
        assert check_vals(1, 'int', 'float') == 1.0
        assert check_vals(1., 'float', 'float') == 1.0
        assert check_vals(1.1, 'float', 'float') == 1.1
        with pytest.raises(TypeError):
            check_vals(1j, 'complex', 'float')

        # Test for int
        assert check_vals(1, 'int', 'int') == 1
        assert check_vals(int(1.1), 'int', 'int') == 1
        with pytest.raises(TypeError):
            check_vals(1.1, 'float', 'int')

    # Check if all the value checks work correctly
    def test_value(self):
        # Check for negative value
        assert check_vals(-1, 'neg', 'neg') == -1
        with pytest.raises(ValueError):
            check_vals(0, 'zero', 'neg')
        with pytest.raises(ValueError):
            check_vals(1, 'pos', 'neg')

        # Check for non-negative value
        assert check_vals(0, 'zero', 'nneg') == 0
        assert check_vals(1, 'pos', 'nneg') == 1
        with pytest.raises(ValueError):
            check_vals(-1, 'neg', 'nneg')

        # Check for positive value
        assert check_vals(1, 'pos', 'pos') == 1
        with pytest.raises(ValueError):
            check_vals(0, 'zero', 'pos')
        with pytest.raises(ValueError):
            check_vals(-1, 'neg', 'pos')

        # Check for non-positive value
        assert check_vals(0, 'zero', 'npos') == 0
        assert check_vals(-1, 'neg', 'npos') == -1
        with pytest.raises(ValueError):
            check_vals(1, 'pos', 'npos')

        # Check for non-zero value
        assert check_vals(-1, 'neg', 'nzero') == -1
        assert check_vals(1, 'pos', 'nzero') == 1
        with pytest.raises(ValueError):
            check_vals(0, 'zero', 'nzero')

        # Check for normalized value
        assert check_vals(1, 'normal', 'normal') == 1
        assert check_vals(0, 'normal', 'normal') == 0
        assert check_vals(0.5, 'normal', 'normal') == 0.5
        assert check_vals(-0.5, 'normal', 'normal') == -0.5
        assert check_vals(-1, 'normal', 'normal') == -1
        with pytest.raises(ValueError):
            check_vals(2, 'normal', 'normal')
        with pytest.raises(ValueError):
            check_vals(-2, 'normal', 'normal')

        # Check for infinite value
        with pytest.raises(ValueError):
            check_vals(np.infty, 'infty')
        with pytest.raises(ValueError):
            check_vals(np.NaN, 'nan')
        with pytest.raises(TypeError):
            check_vals('str', 'str')

    # Check if providing array_like as value works correctly
    def test_array_like(self):
        # Check for list
        lst = [1, 2]
        lst2 = check_vals(lst, 'list', 'float')
        assert lst2 == [1.0, 2.0]
        assert type(lst2[0]) == float

        # Check for tuple
        tpl = (1, 2)
        tpl2 = check_vals(tpl, 'tuple', 'float')
        assert tpl2 == (1.0, 2.0)
        assert type(tpl2[0]) == float

        # Check for NumPy array
        array = np.array([1, 2])
        assert (check_vals(array, 'array', 'pos') == [1, 2]).all()

        # Check for NumPy array of floats
        array = np.array([1, 2])
        array2 = check_vals(array, 'array', 'float')
        assert (array2 == [1.0, 2.0]).all()
        assert array2.dtype.name == 'float64'

        # Check for NumPy array of bools
        array = np.array([False, True])
        array2 = check_vals(array, 'array', 'bool')
        assert (array2 == [0, 1]).all()
        assert array2.dtype.name == 'int64' if not win32 else 'int32'

        # Check for NumPy array of strings
        array = np.array(['a', 'b'])
        array2 = check_vals(array, 'array', 'str')
        assert array2.dtype.name == 'str32'

        # Check if providing an empty list raises an error
        with pytest.raises(InputError):
            check_vals([], 'list', 'float')

        # Check if providing a dict or sequenced list raises an error
        with pytest.raises(InputError):
            check_vals({}, 'dict', 'float')
        with pytest.raises(InputError):
            check_vals([1, [2]], 'seq_list', 'float')
        with pytest.raises(InputError):
            check_vals([[1, 2], [3, 4, 5]], 'seq_list', 'float')

    # Check if providing incorrect arguments raises an error
    def test_args(self):
        with pytest.raises(ValueError):
            check_vals(1, 'int', 'invalid')


# Pytest for the get_info function
def test_get_info():
    # Print the output of the get_info() function
    get_info()


# Pytest for the getCLogger function
def test_getCLogger():
    # Check if getCLogger returns a CLogger instance
    assert isinstance(getCLogger('C_TEST'), PRISM_Logger)

    # Check if getCLogger returns the base logger if not specified
    assert getCLogger() == logging.root.manager.loggerDict['prism']

    # Check if LoggerClass is still default Logger
    assert logging.getLoggerClass() == logging.Logger


# Pytest for the getRLogger function
def test_getRLogger():
    # Check if getRLogger returns an RLogger instance
    assert isinstance(getRLogger('R_TEST'), PRISM_Logger)

    # Check if getRLogger returns the base logger if not specified
    assert getRLogger() == logging.root.manager.loggerDict['prism']

    # Check if LoggerClass is still default Logger
    assert logging.getLoggerClass() == logging.Logger


# Pytest for the np_array function
def test_np_array():
    array = np.array([1, 2])
    assert np_array(array) is array
