# -*- coding: utf-8 -*-

"""
ModelLink Tests
===============
Contains a collection of functions for testing :class:`~ModelLink` subclasses.


Available functions
===================
:func:`~test_modellink_subclass`
    Test for checking if given `cls` is/are a subclass of the
    :class:`~prism.ModelLink` abstract base class and if their structure is
    correct. Raises an :class:`~AssertionError` if the check fails.

"""

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Import modellink_tests modules
from . import _tests
from ._tests import *

# All declaration
__all__ = []
__all__.extend(_tests.__all__)
