# -*- coding: utf-8 -*-

"""
Tests for PRISM's ModelLink class

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import inspect

# All declaration
__all__ = ['test_modellink_subclass']


# %% TESTS
# Function that tests if ModelLink subclasses have the correct structure
def test_modellink_subclass(cls=None):
    """
    Test for checking if given `cls` is/are a subclass of the
    :class:`~prism.ModelLink` abstract base class and if their structure is
    correct. Raises an :class:`~AssertionError` if the check fails.

    Optional
    --------
    cls : class, list of classes or None. Default: None
        If class or list of classes, perform the test on all elements given.
        If *None*, perform the test on all classes declared in the
        :attr:`~prism.modellink.__all__` attribute of the
        :mod:`~prism.modellink` module.

    """

    # Try to import the modellink module
    try:
        import prism.modellink as modellink
        from prism import ModelLink
    except ImportError:
        raise

    # Create empty list of classes to test
    cls_lst = []

    # If cls was not given, test all ModelLink subclasses in PRISM
    if cls is None:
        for cls_name in modellink.__all__:
            if cls_name not in ('ModelLink', 'tests'):
                cls_lst.append(getattr(modellink, cls_name))

    # If cls is given, add it to the list
    else:
        # If a list of classes was given, add them all
        if isinstance(cls, list):
            cls_lst.extend(cls)

        # If a single class was given, add it to the list
        else:
            cls_lst.append(cls)

    # Test various details for all classes in the list
    for cls in cls_lst:
        # Check if the class is actually a class
        assert inspect.isclass(cls)

        # Check if the class is a subclass of ModelLink
        assert issubclass(cls, ModelLink)

        # Check if the two abstract methods have been overridden
        assert cls.call_model.im_func is not ModelLink.call_model.im_func
        assert cls.get_md_var.im_func is not ModelLink.get_md_var.im_func

        # Check if the _create_properties()-method has not been overridden
        assert(cls._create_properties.im_func is
               ModelLink._create_properties.im_func)


# %% EXECUTION
# If this file is executed, perform the tests
if __name__ == '__main__':
    test_modellink_subclass()
