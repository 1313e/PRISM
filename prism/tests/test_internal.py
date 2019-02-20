# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import logging
from os import path
from sys import platform

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from prism.__version__ import compat_version, prism_version
from prism._internal import (FeatureWarning, PRISM_Logger, RequestError,
                             RequestWarning, docstring_append, docstring_copy,
                             docstring_substitute, check_instance, check_vals,
                             check_compatibility, convert_str_seq, delist,
                             get_PRISM_File, get_info, getCLogger, getRLogger,
                             import_cmaps, np_array, raise_error,
                             raise_warning, rprint)

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save if this platform is Windows
win32 = platform.startswith('win')


# %% PYTEST CLASSES AND FUNCTIONS
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


# Pytest for the custom function decorators
class TestDecorators(object):
    # Create method with no docstring that is appended
    @docstring_append("appended")
    def append_method1(self):
        pass

    # Create method with a docstring that is appended
    @docstring_append("appended")
    def append_method2(self):
        """original """

    # Check if docstring_append works correctly
    def test_docstring_append(self):
        assert self.append_method1.__doc__ == "appended"
        assert self.append_method2.__doc__ == "original appended"

    # Create method with no docstring at all
    def empty_method(self):
        pass

    # Create method that copies an empty docstring
    @docstring_copy(empty_method)
    def copy_method1(self):
        pass

    # Create method that copies a docstring
    @docstring_copy(append_method1)
    def copy_method2(self):
        pass

    # Check if docstring_copy works correctly
    def test_docstring_copy(self):
        assert self.copy_method1.__doc__ is None
        assert self.copy_method1.__doc__ == self.empty_method.__doc__
        assert self.copy_method2.__doc__ == self.append_method1.__doc__

    # Check if providing both args and kwargs raises an error
    with pytest.raises(InputError):
        @docstring_substitute("positional", x="keyword")
        def substitute_method1(self):
            pass

    # Create method using args substitutes
    @docstring_substitute("positional")
    def substitute_method2(self):
        """%s"""

    # Create method using kwargs substitutes
    @docstring_substitute(x="keyword")
    def substitute_method3(self):
        """%(x)s"""

    # Check if providing args to a method with no docstring raises an error
    with pytest.raises(InputError):
        @docstring_substitute("positional")
        def substitute_method4(self):
            pass

    # Check if docstring_substitute works correctly
    def test_docstring_substitute(self):
        assert self.substitute_method2.__doc__ == "positional"
        assert self.substitute_method3.__doc__ == "keyword"


# Pytest for the check_compatibility function
def test_check_compatibility():
    # Check if current version is compatible (it should)
    check_compatibility(prism_version)

    # Check if providing too old or too new version raises an error
    with pytest.raises(RequestError):
        check_compatibility(compat_version[-1])
    with pytest.raises(RequestError):
        check_compatibility('999.999.999')

    # Check if providing a v1.0.x version raises a warning
    with pytest.warns(RequestWarning):
        check_compatibility('1.0.0')


# Pytest for the check_instance function
def test_check_instance():
    # Check if providing a non-class raises an error
    with pytest.raises(InputError):
        check_instance(np.array(1), np.array)

    # Check if providing an incorrect instance raises an error
    with pytest.raises(TypeError):
        check_instance(list(), np.ndarray)

    # Check if providing a proper instance of a class gives 1
    assert check_instance(np.array(1), np.ndarray) == 1


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
            check_vals(2, 'bool', 'bool')

        # Test for str
        assert check_vals('str', 'str', 'str') == 'str'
        assert check_vals(u'str', 'str', 'str') == u'str'
        with pytest.raises(TypeError):
            check_vals(1, 'str', 'str')

        # Test for float
        assert check_vals(1, 'float', 'float') == 1
        assert check_vals(1., 'float', 'float') == 1.
        assert check_vals(1.0, 'float', 'float') == 1.0
        with pytest.raises(TypeError):
            check_vals('1', 'str', 'float')

        # Test for int
        assert check_vals(1, 'int', 'int') == 1
        assert check_vals(int(1.0), 'int', 'int') == 1
        with pytest.raises(TypeError):
            check_vals(1.0, 'float', 'int')

    # Check if all the value checks work correctly
    def test_value(self):
        # Check for negative value
        assert check_vals(-1, 'neg', 'neg') == -1
        with pytest.raises(ValueError):
            check_vals(0, 'neg', 'neg')
        with pytest.raises(ValueError):
            check_vals(1, 'neg', 'neg')

        # Check for non-negative value
        assert check_vals(0, 'nneg', 'nneg') == 0
        assert check_vals(1, 'nneg', 'nneg') == 1
        with pytest.raises(ValueError):
            check_vals(-1, 'nneg', 'nneg')

        # Check for positive value
        assert check_vals(1, 'pos', 'pos') == 1
        with pytest.raises(ValueError):
            check_vals(0, 'pos', 'pos')
        with pytest.raises(ValueError):
            check_vals(-1, 'pos', 'pos')

        # Check for non-positive value
        assert check_vals(0, 'npos', 'npos') == 0
        assert check_vals(-1, 'npos', 'npos') == -1
        with pytest.raises(ValueError):
            check_vals(1, 'npos', 'npos')

        # Check for non-zero value
        assert check_vals(-1, 'nzero', 'nzero') == -1
        assert check_vals(1, 'nzero', 'nzero') == 1
        with pytest.raises(ValueError):
            check_vals(0, 'nzero', 'nzero')

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


# Pytest for the convert_str_seq function
def test_convert_str_seq():
    # Check if string sequence is converted correctly
    assert convert_str_seq('[[]]]1e1,\n8.,A<{7)\\B') == [10., 8.0, 'A', 7, 'B']


# Pytest for the delist function
def test_delist():
    # Check if providing not a list raises an error
    with pytest.raises(TypeError):
        delist(np.array([1]))

    # Check if provided list is delisted correctly
    assert delist([[], (), [np.array(1)], [7], 8]) == [[np.array(1)], [7], 8]


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


# Pytest for the import_cmaps function
def test_import_cmaps():
    # Check if providing a non-existing directory raises an error
    with pytest.raises(OSError):
        import_cmaps('./test')

    # Check if providing a custom directory with invalid cmaps raises an error
    with pytest.raises(InputError):
        import_cmaps(path.join(dirpath, 'data'))


# Pytest for the np_array function
def test_np_array():
    array = np.array([1, 2])
    assert np_array(array) is array


# Pytest for the raise_error function
def test_raise_error():
    # Create a logger and check if an error can be properly raised and logged
    logger = logging.getLogger('TEST')
    with pytest.raises(RequestError):
        raise_error('ERROR', RequestError, logger)


# Pytest for the raise_warning function
def test_raise_warning():
    # Create a logger and check if a warning can be properly raised and logged
    logger = logging.getLogger('TEST')
    with pytest.warns(FeatureWarning):
        raise_warning('WARNING', FeatureWarning, logger)


# Pytest for the rprint function
def test_rprint():
    # Check if rprint works correctly
    rprint('Testing')
