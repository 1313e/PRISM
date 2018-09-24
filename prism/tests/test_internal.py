# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
import logging
from os import path

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from prism._internal import (compat_version, prism_version, CLogger,
                             PRISM_File, RequestError, docstring_append,
                             docstring_copy, docstring_substitute, check_val,
                             check_compatibility, convert_str_seq, delist,
                             getCLogger, import_cmaps, move_logger,
                             raise_error, rprint, start_logger)

# Save the path to this directory
dirpath = path.dirname(__file__)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for CLogger class
class Test_CLogger(object):
    # Initialize logger
    logger = CLogger('TEST')

    # Check if all overridden methods still work properly
    def test_init(self):
        assert isinstance(self.logger, logging.Logger)

    def test_debug(self):
        assert self.logger.debug("Test") is None

    def test_info(self):
        assert self.logger.info("Test") is None

    def test_warning(self):
        assert self.logger.warning("Test") is None

    def test_error(self):
        assert self.logger.error("Test") is None

    def test_exception(self):
        assert self.logger.exception("Test") is None

    def test_critical(self):
        assert self.logger.critical("Test") is None

    def test_log(self):
        assert self.logger.log(logging.DEBUG, "Test") is None
        assert self.logger.log(logging.INFO, "Test") is None
        assert self.logger.log(logging.WARNING, "Test") is None
        assert self.logger.log(logging.ERROR, "Test") is None
        assert self.logger.log(logging.FATAL, "Test") is None
        assert self.logger.log(logging.CRITICAL, "Test") is None


# Pytest for PRISM_File class
class Test_PRISM_File(object):
    def test(self, tmpdir):
        # Set _hdf5_file property to something default
        PRISM_File._hdf5_file = path.join(tmpdir.strpath, 'test.hdf5')

        # Check if opening master file works
        filename = path.join(tmpdir.strpath, 'test.hdf5')
        with PRISM_File('w') as file:
            assert path.basename(file.filename) == 'test.hdf5'
        assert path.exists(filename)

        # Check if opening emulator system file works
        filename = path.join(tmpdir.strpath, 'test_0.hdf5')
        with PRISM_File('w', emul_s=0) as file:
            assert path.basename(file.filename) == 'test_0.hdf5'
        assert path.exists(filename)

        # Check if opening custom defined file works
        filename = path.join(tmpdir.strpath, 'test_test.hdf5')
        with PRISM_File('w', filename=filename) as file:
            assert path.basename(file.filename) == 'test_test.hdf5'
        assert path.exists(filename)


# Pytest for RequestError exception class
def test_RequestError():
    # Check if class is derived from Exception and try to raise it
    assert Exception in RequestError.mro()
    with pytest.raises(RequestError):
        raise RequestError


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
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
        @docstring_substitute("positional")
        def substitute_method4(self):
            pass

    # Check if docstring_substitute works correctly
    def test_docstring_substitute(self):
        assert self.substitute_method2.__doc__ == "positional"
        assert self.substitute_method3.__doc__ == "keyword"


# Pytest for check_val function
class Test_check_val(object):
    # Check if all the type checks work correctly
    def test_type(self):
        # Test for bool
        assert check_val(0, 'bool', 'bool') == 0
        assert check_val(1, 'bool', 'bool') == 1
        assert check_val(False, 'bool', 'bool') == 0
        assert check_val(True, 'bool', 'bool') == 1
        assert check_val("0", 'bool', 'bool') == 0
        assert check_val("1", 'bool', 'bool') == 1
        assert check_val("False", 'bool', 'bool') == 0
        assert check_val("True", 'bool', 'bool') == 1
        with pytest.raises(TypeError):
            check_val(2, 'bool', 'bool')

        # Test for str
        assert check_val('str', 'str', 'str') == 'str'
        assert check_val(u'str', 'str', 'str') == u'str'
        with pytest.raises(TypeError):
            check_val(1, 'str', 'str')

        # Test for float
        assert check_val(1, 'float', 'float') == 1
        assert check_val(1., 'float', 'float') == 1.
        assert check_val(1.0, 'float', 'float') == 1.0

        # Test for int
        assert check_val(1, 'int', 'int') == 1
        assert check_val(int(1.0), 'int', 'int') == 1
        with pytest.raises(TypeError):
            check_val(1.0, 'float', 'int')

    # Check if all the value checks work correctly
    def test_value(self):
        # Check for negative value
        assert check_val(-1, 'neg', 'neg') == -1
        with pytest.raises(ValueError):
            check_val(0, 'neg', 'neg')
        with pytest.raises(ValueError):
            check_val(1, 'neg', 'neg')

        # Check for non-negative value
        assert check_val(0, 'nneg', 'nneg') == 0
        assert check_val(1, 'nneg', 'nneg') == 1
        with pytest.raises(ValueError):
            check_val(-1, 'nneg', 'nneg')

        # Check for positive value
        assert check_val(1, 'pos', 'pos') == 1
        with pytest.raises(ValueError):
            check_val(0, 'pos', 'pos')
        with pytest.raises(ValueError):
            check_val(-1, 'pos', 'pos')

        # Check for non-positive value
        assert check_val(0, 'npos', 'npos') == 0
        assert check_val(-1, 'npos', 'npos') == -1
        with pytest.raises(ValueError):
            check_val(1, 'npos', 'npos')

        # Check for non-zero value
        assert check_val(-1, 'nzero', 'nzero') == -1
        assert check_val(1, 'nzero', 'nzero') == 1
        with pytest.raises(ValueError):
            check_val(0, 'nzero', 'nzero')

        # Check for infinite value
        with pytest.raises(ValueError):
            check_val(np.infty, 'infty')
        with pytest.raises(ValueError):
            check_val(np.NaN, 'nan')
        with pytest.raises(ValueError):
            check_val('str', 'str')

    # Check if providing incorrect arguments raises an error
    def test_args(self):
        with pytest.raises(InputError):
            check_val(1, 'int', 'invalid')


# Pytest for the check_compatibility function
def test_check_compatibility():
    # Check if current version is compatible (it should)
    check_compatibility(prism_version)

    # Check if providing too old or too new version raises an error
    with pytest.raises(RequestError):
        check_compatibility(compat_version[-1])
    with pytest.raises(RequestError):
        check_compatibility('999.999.999')


# Pytest for the convert_str_seq function
def test_convert_str_seq():
    # Check if string sequence is converted correctly
    assert convert_str_seq('[:[]]]1, 8,A<{7)\n') == ['1', '8', 'A', '7', 'n']


# Pytest for the delist function
def test_delist():
    # Check if providing not a list raises an error
    with pytest.raises(TypeError):
        delist(np.array([1]))

    # Check if provided list is delisted correctly
    assert delist([[], [], [np.array(1)], [7], 8]) == [[np.array(1)], [7], 8]


# Pytest for the getCLogger function
def test_getCLogger():
    # Check if getCLogger returns a Logger instance
    assert isinstance(getCLogger('TEST'), logging.Logger)

    # CHeck if getCLogger returns the root logger if not specified
    assert getCLogger() == logging.root


# Pytest for the import_cmaps function
def test_import_cmaps():
    # Check if PRISM's custom cmaps can be loaded imported correctly
    import_cmaps()

    # Check if providing a custom directory with invalid cmaps raises an error
    with pytest.raises(InputError):
        import_cmaps(path.join(dirpath, 'data'))


# Pytest for the move_logger and start_logger functions
def test_movestart_logger(tmpdir):
    # Create a path to a new logfile
    filename = path.join(tmpdir.strpath, 'test.log')

    # Check if created logfile exists and has the correct name
    assert start_logger(filename) == filename
    assert path.exists(filename)

    # Create a move destination
    dest = tmpdir.mkdir('working_dir')

    # Check if logfile can be correctly moved
    move_logger(dest.strpath, filename)

    # Check if a new undefined logfile can be created
    filename_src = start_logger()

    # Check if it can be logged correctly
    logger = logging.getLogger('TEST')
    logger.info('Writing a line')

    # Check if both logfiles can be combined correctly
    move_logger(dest.strpath, filename_src)


# Pytest for the raise_error function
def test_raise_error():
    # Create a logger and check if an error can be properly raised and logged
    logger = logging.getLogger('TEST')
    with pytest.raises(Exception):
        raise_error(Exception, 'ERROR', logger)


# Pytest for the rprint function
def test_rprint():
    # Check if rprint works correctly
    rprint('Testing')
