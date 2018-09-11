# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
import logging
from os import path
import sys

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from ..__version__ import compat_version, prism_version
from .._internal import (CLogger, PRISM_File, RequestError, docstring_append,
                         docstring_copy, docstring_substitute, check_val,
                         check_compatibility, convert_str_seq, delist,
                         getCLogger, import_cmaps, move_logger, raise_error,
                         rprint, start_logger)


# %% PYTEST CLASSES
class Test_CLogger(object):
    logger = CLogger('TEST')

    def test_init(self):
        assert self.logger

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


class Test_PRISM_File(object):
    def test(self, tmpdir):
        PRISM_File._hdf5_file = path.join(tmpdir.strpath, 'test.hdf5')
        filename = path.join(tmpdir.strpath, 'test.hdf5')
        with PRISM_File('w') as file:
            assert path.basename(file.filename) == 'test.hdf5'
        assert path.exists(filename)

        filename = path.join(tmpdir.strpath, 'test_0.hdf5')
        with PRISM_File('w', emul_s=0) as file:
            assert path.basename(file.filename) == 'test_0.hdf5'
        assert path.exists(filename)

        filename = path.join(tmpdir.strpath, 'test_test.hdf5')
        with PRISM_File('w', filename=filename) as file:
            assert path.basename(file.filename) == 'test_test.hdf5'
        assert path.exists(filename)


def test_RequestError():
    assert Exception in RequestError.mro()
    with pytest.raises(RequestError):
        raise RequestError


class TestDecorators(object):
    @docstring_append("appended")
    def append_method1(self):
        pass

    @docstring_append("appended")
    def append_method2(self):
        """original """

    def test_docstring_append(self):
        assert self.append_method1.__doc__ == "appended"
        assert self.append_method2.__doc__ == "original appended"

    def empty_method(self):
        pass

    @docstring_copy(empty_method)
    def copy_method1(self):
        pass

    @docstring_copy(append_method1)
    def copy_method2(self):
        pass

    def test_docstring_copy(self):
        assert self.copy_method1.__doc__ is None
        assert self.copy_method1.__doc__ == self.empty_method.__doc__
        assert self.copy_method2.__doc__ == self.append_method1.__doc__

    with pytest.raises(AssertionError):
        @docstring_substitute("positional", x="keyword")
        def substitute_method1(self):
            pass

    @docstring_substitute("positional")
    def substitute_method2(self):
        """%s"""

    @docstring_substitute(x="keyword")
    def substitute_method3(self):
        """%(x)s"""

    with pytest.raises(AssertionError):
        @docstring_substitute("positional")
        def substitute_method4(self):
            pass

    def test_docstring_substitute(self):
        assert self.substitute_method2.__doc__ == "positional"
        assert self.substitute_method3.__doc__ == "keyword"


class Test_check_val(object):
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
            check_val(b'str', 'str', 'str')
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

    def test_args(self):
        with pytest.raises(InputError):
            check_val(1, 'int', 'invalid')


def test_check_compatibility():
    check_compatibility(prism_version)
    with pytest.raises(RequestError):
        check_compatibility(compat_version[-1])
    with pytest.raises(RequestError):
        check_compatibility('999.999.999')


def test_convert_str_seq():
    assert convert_str_seq('[[]]]1, 8,A<{7)4') == ['1', '8', 'A', '7', '4']


def test_delist():
    with pytest.raises(TypeError):
        delist(np.array([1]))
    assert delist([[], [], [np.array(1)], [7], 8]) == [[np.array(1)], [7], 8]


def test_getCLogger():
    assert isinstance(getCLogger('TEST'), logging.Logger)
    assert getCLogger() == logging.root


def test_import_cmaps():
    import_cmaps()
    with pytest.raises(InputError):
        import_cmaps(path.join(path.dirname(__file__), 'data'))


def test_movestart_logger(tmpdir):
    filename = path.join(tmpdir.strpath, 'test.log')
    assert start_logger(filename) == filename
    assert path.exists(filename)
    dest = tmpdir.mkdir('working_dir')
    move_logger(dest.strpath, filename)
    filename_src = start_logger()
    logger = logging.getLogger('TEST')
    logger.info('Writing a line')
    move_logger(dest.strpath, filename_src)


def test_raise_error():
    logger = logging.getLogger('TEST')
    with pytest.raises(Exception):
        raise_error(Exception, 'ERROR', logger)


def test_rprint():
    rprint('Testing')
