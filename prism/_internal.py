# -*- coding: utf-8 -*-

"""
PRISM Internal
==============
Contains a collection of support classes/functions for the PRISM package.


Available classes
-----------------
:class:`~RequestError`
    Generic exception raised for invalid action requests in the PRISM pipeline.


Available functions
-------------------
:func:`~check_float`
    Checks if provided argument `name` of `value` is a float.
    Returns `value` if *True* and raises a TypeError if *False*.

:func:`~check_int`
    Checks if provided argument `name` of `value` is an integer.
    Returns `value` if *True* and raises a TypeError if *False*.

:func:`~check_neg_float`
    Checks if provided argument `name` of `value` is a negative float.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_neg_int`
    Checks if provided argument `name` of `value` is a negative integer.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_nneg_float`
    Checks if provided argument `name` of `value` is a non-negative float.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_nneg_int`
    Checks if provided argument `name` of `value` is a non-negative integer.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_npos_float`
    Checks if provided argument `name` of `value` is a non-positive float.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_npos_int`
    Checks if provided argument `name` of `value` is a non-positive integer.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_pos_float`
    Checks if provided argument `name` of `value` is a positive float.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~check_pos_int`
    Checks if provided argument `name` of `value` is a positive integer.
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

:func:`~docstring_copy`
    Custom decorator that allows the docstring of a function `source` to be
    copied.

:func:`~move_logger`
    Moves the logging file `filename` from the current working directory to the
    given `working_dir`, and then restarts it again.

:func:`~start_logger`
    Creates a logging file called `filename` in the current working directory,
    opened with `mode` and starts the logger.

"""


# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
import os
from os import path
import shutil

# Package imports
import logging
import logging.config
import numpy as np

# All declaration
__all__ = ['RequestError', 'check_float', 'check_int', 'check_neg_float',
           'check_neg_int', 'check_nneg_float', 'check_nneg_int',
           'check_npos_float', 'check_npos_int', 'check_pos_float',
           'check_pos_int', 'check_str', 'docstring_copy', 'move_logger',
           'start_logger']


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
# Function for checking if a float has been provided
def check_float(value, name):
    """
    Checks if provided argument `name` of `value` is a float.
    Returns `value` if *True* and raises a TypeError if *False*.

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
    Returns `value` if *True* and raises a TypeError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError or ValueError if *False*.

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
    Returns `value` if *True* and raises a TypeError if *False*.

    """

    # Check if str is provided and return if so
    if isinstance(value, (str, np.string_)):
        return(value)
    else:
        raise TypeError("Input argument '%s' is not of type 'str'!" % (name))


# Define custom decorator for copying docstrings from one function to another
def docstring_copy(source):
    """
    Custom decorator that allows the docstring of a function `source` to be
    copied.

    """

    def do_copy(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return(target)
    return(do_copy)


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
