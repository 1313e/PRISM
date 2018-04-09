# -*- coding: utf-8 -*-

"""
PRISM Internal
==============
Contains a collection of support classes/functions/lists for the PRISM package.


Available classes
-----------------
:class:`~RequestError`
    Generic exception raised for invalid action requests in the PRISM pipeline.


Available functions
-------------------
:func:`~check_compatibility`
    Checks if the provided `emul_version` is compatible with the current
    version of PRISM.
    Raises a :class:`~RequestError` if *False* and indicates which version of
    PRISM still supports the provided `emul_version`.

:func:`~check_float`
    Checks if provided argument `name` of `value` is a float.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*.

:func:`~check_int`
    Checks if provided argument `name` of `value` is an integer.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*.

:func:`~check_neg_float`
    Checks if provided argument `name` of `value` is a negative float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_neg_int`
    Checks if provided argument `name` of `value` is a negative integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_nneg_float`
    Checks if provided argument `name` of `value` is a non-negative float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_nneg_int`
    Checks if provided argument `name` of `value` is a non-negative integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_npos_float`
    Checks if provided argument `name` of `value` is a non-positive float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_npos_int`
    Checks if provided argument `name` of `value` is a non-positive integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_pos_float`
    Checks if provided argument `name` of `value` is a positive float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_pos_int`
    Checks if provided argument `name` of `value` is a positive integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

:func:`~check_str`
    Checks if provided argument `name` of `value` is a string.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*.

:func:`~docstring_append`
    Custom decorator that allows a given string `addendum` to be appended to
    the docstring of the target function, separated by a given string `join`.

:func:`~docstring_copy`
    Custom decorator that allows the docstring of a function `source` to be
    copied to the target function.

:func:`~docstring_substitute`
    Custom decorator that allows either given positional arguments `args` or
    keyword arguments `kwargs` to be substituted into the docstring of the
    target function.

:func:`~move_logger`
    Moves the logging file `filename` from the current working directory to the
    given `working_dir`, and then restarts it again.

:func:`~start_logger`
    Creates a logging file called `filename` in the current working directory,
    opened with `mode` and starts the logger.


Defined lists
-------------
seq_char_list
    List defining characters that need to be removed from a sequence given in
    the PRISM parameter file, before being split up into individual elements.

"""


# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
import os
from os import path
import shutil
import sys

# Package imports
from e13tools.core import _compare_versions
import logging
import logging.config
import numpy as np

# PRISM imports
from .__version__ import compat_version, version as prism_version

# All declaration
__all__ = ['RequestError', 'check_compatibility', 'check_float', 'check_int',
           'check_neg_float', 'check_neg_int', 'check_nneg_float',
           'check_nneg_int', 'check_npos_float', 'check_npos_int',
           'check_pos_float', 'check_pos_int', 'check_str', 'docstring_append',
           'docstring_copy', 'docstring_substitute', 'move_logger',
           'start_logger', 'seq_char_list']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% CLASS DEFINITIONS
# Define Exception class for when a requested action is not possible
class RequestError(Exception):
    """
    Generic exception raised for invalid action requests in the PRISM pipeline.

    General purpose exception class, raised whenever a requested action cannot
    be executed due to it not being allowed or possible in the current state of
    the :obj:`~Pipeline` instance.

    """

    pass


# %% FUNCTION DEFINITIONS
# Function for checking if emulator system is compatible with PRISM version
def check_compatibility(emul_version):
    """
    Checks if the provided `emul_version` is compatible with the current
    version of PRISM.
    Raises a :class:`~RequestError` if *False* and indicates which version of
    PRISM still supports the provided `emul_version`.

    """

    # Do some logging
    logger = logging.getLogger('INIT')
    logger.info("Performing compatibility check.")

    # Loop over all compatibility versions
    for version in compat_version:
        # If a compat_version is the same or newer than the emul_version
        # then it is incompatible
        if _compare_versions(version, emul_version):
            logger.error("The provided emulator system is incompatible with "
                         "the current version of PRISM (v%s). The last "
                         "compatible version of PRISM is v%s."
                         % (prism_version, version))
            raise RequestError("The provided emulator system is incompatible "
                               "with the current version of PRISM (v%s). The"
                               " last compatible version of PRISM is v%s."
                               % (prism_version, version))

    # Check if emul_version is not newer than prism_version
    if not _compare_versions(prism_version, emul_version):
        logger.error("The provided emulator system was constructed with a "
                     "version later than the current version of PRISM (v%s). "
                     "Use PRISM v%s or later to use this emulator system."
                     % (prism_version, emul_version))
        raise RequestError("The provided emulator system was constructed with "
                           "a version later than the current version of PRISM "
                           "(v%s). Use PRISM v%s or later to use this emulator"
                           " system."
                           % (prism_version, emul_version))
    else:
        logger.info("Compatibility check was successful.")


# Function for checking if a float has been provided
def check_float(value, name):
    """
    Checks if provided argument `name` of `value` is a float.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*.

    """

    # Check if float is provided and return if so
    if isinstance(value, (int, float, np.integer, np.floating)):
        return(value)
    else:
        raise TypeError("Input argument '%s' is not of type 'float'!" % (name))


# Function for checking if an int has been provided
def check_int(value, name):
    """
    Checks if provided argument `name` of `value` is an integer.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*.

    """

    # Check if int is provided and return if so
    if isinstance(value, (int, np.integer)):
        return(value)
    else:
        raise TypeError("Input argument '%s' is not of type 'int'!" % (name))


# Function for checking if a provided float is negative
def check_neg_float(value, name):
    """
    Checks if provided argument `name` of `value` is a negative float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if float is provided
    value = check_float(value, name)

    # Check if float is negative
    if(value < 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not negative!" % (name))


# Function for checking if a provided int is negative
def check_neg_int(value, name):
    """
    Checks if provided argument `name` of `value` is a negative integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if int is provided
    value = check_int(value, name)

    # Check if int is negative
    if(value < 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not negative!" % (name))


# Function for checking if a provided float is non-negative
def check_nneg_float(value, name):
    """
    Checks if provided argument `name` of `value` is a non-negative float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-negative
    if not(value < 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not non-negative!" % (name))


# Function for checking if a provided int is non-negative
def check_nneg_int(value, name):
    """
    Checks if provided argument `name` of `value` is a non-negative integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-negative
    if not(value < 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not non-negative!" % (name))


# Function for checking if a provided float is non-positive
def check_npos_float(value, name):
    """
    Checks if provided argument `name` of `value` is a non-positive float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-positive
    if not(value > 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not non-positive!" % (name))


# Function for checking if a provided int is non-positive
def check_npos_int(value, name):
    """
    Checks if provided argument `name` of `value` is a non-positive integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-positive
    if not(value > 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not non-positive!" % (name))


# Function for checking if a provided float is positive
def check_pos_float(value, name):
    """
    Checks if provided argument `name` of `value` is a positive float.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if float is provided
    value = check_float(value, name)

    # Check if float is positive
    if(value > 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not positive!" % (name))


# Function for checking if a provided int is positive
def check_pos_int(value, name):
    """
    Checks if provided argument `name` of `value` is a positive integer.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*.

    """

    # Check if int is provided
    value = check_int(value, name)

    # Check if int is positive
    if(value > 0):
        return(value)
    else:
        raise ValueError("Input argument '%s' is not positive!" % (name))


# Function for checking if a str has been provided
def check_str(value, name):
    """
    Checks if provided argument `name` of `value` is a string.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*.

    """

    # Check if str is provided and return if so
    if isinstance(value, (str, np.string_, unicode)):
        return(value)
    else:
        raise TypeError("Input argument '%s' is not of type 'str'!" % (name))


# Define custom decorator for appending docstrings to a function's docstring
def docstring_append(addendum, join=''):
    """
    Custom decorator that allows a given string `addendum` to be appended to
    the docstring of the target function, separated by a given string `join`.

    """

    def do_append(target):
        if target.__doc__:
            target.__doc__ = join.join([target.__doc__, addendum])
        else:
            target.__doc__ = addendum
        return(target)
    return(do_append)


# Define custom decorator for copying docstrings from one function to another
def docstring_copy(source):
    """
    Custom decorator that allows the docstring of a function `source` to be
    copied to the target function.

    """

    def do_copy(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return(target)
    return(do_copy)


# Define custom decorator for substituting strings into a function's docstring
def docstring_substitute(*args, **kwargs):
    """
    Custom decorator that allows either given positional arguments `args` or
    keyword arguments `kwargs` to be substituted into the docstring of the
    target function.

    """

    if len(args) and len(kwargs):
        raise AssertionError("Either only positional or keyword arguments are "
                             "allowed!")
    else:
        params = args or kwargs

    def do_substitution(target):
        if target.__doc__:
            target.__doc__ = target.__doc__ % (params)
        else:
            raise AssertionError("Target has no docstring available for "
                                 "substitutions!")
        return(target)
    return(do_substitution)


# Define function that can move the logging file of PRISM and restart logging
def move_logger(working_dir, filename='prism_log.log'):
    """
    Moves the logging file `filename` from the current working directory to the
    given `working_dir`, and then restarts it again.

    Parameters
    ----------
    working_dir : string
        String containing the directory the log-file needs to be moved to.

    Optional
    --------
    filename : string. Default: 'prism_log.log'
        String containing the name of the log-file that needs to be moved.

    """

    # Shut down logging process to allow the log-file to be moved
    logging.shutdown()

    # Get source and destination paths
    source = path.abspath(filename)
    destination = path.join(working_dir, filename)

    # Check if file already exists and either combine files or move the file
    if path.isfile(destination):
        with open(destination, 'a') as dest, open(source, 'r') as src:
            for line in src:
                dest.write(line)
        os.remove(source)
    else:
        shutil.move(source, destination)

    # Restart the logger
    start_logger(filename=destination, mode='a')


# Define function that can start the logging process of PRISM
def start_logger(filename='prism_log.log', mode='w'):
    """
    Creates a logging file called `filename` in the current working directory,
    opened with `mode` and starts the logger.

    Optional
    --------
    filename : string. Default: 'prism_log.log'
        String containing the name of the log-file that is created.
    mode : {'r', 'r+', 'w', 'w-'/'x', 'a'}. Default: 'w'
        String indicating how the log-file needs to be opened.

    """

    # Define logging dict
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': "[%(asctime)s][%(threadName)-10s][%(levelname)-4s] "
                          "%(name)-10s \t%(message)s",
                'datefmt': "%Y-%m-%d %H:%M:%S",
            },
        },
        'handlers': {
            'file': {
                '()': logging.FileHandler,
                'level': 'DEBUG',
                'formatter': 'default',
                'filename': filename,
                'mode': mode,
                'encoding': 'utf-8',
            },
        },
        'root': {
            'handlers': ['file'],
            'level': 'DEBUG',
        },
    }

    # Start the logger from the dict above
    logging.config.dictConfig(LOGGING)


# %% LIST DEFINITIONS
seq_char_list = ['(', ')', '[', ']', ',', "'", '"', '|', '/']
