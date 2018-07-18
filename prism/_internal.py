# -*- coding: utf-8 -*-

"""
PRISM Internal
==============
Contains a collection of support classes/functions/lists for the PRISM package.

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
from tempfile import mkstemp

# Package imports
from e13tools.core import _compare_versions
import logging
import logging.config
import numpy as np

# PRISM imports
from .__version__ import compat_version, prism_version
from ._docstrings import (check_bool_doc, check_fin_doc, check_type_doc,
                          check_val_doc)

# All declaration
__all__ = ['RequestError', 'check_bool', 'check_compatibility', 'check_finite',
           'check_float', 'check_int', 'check_neg_float', 'check_neg_int',
           'check_nneg_float', 'check_nneg_int', 'check_npos_float',
           'check_npos_int', 'check_nzero_float', 'check_nzero_int',
           'check_pos_float', 'check_pos_int', 'check_str', 'convert_str_seq',
           'docstring_append', 'docstring_copy', 'docstring_substitute',
           'move_logger', 'start_logger', 'aux_char_list']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str

# Setup logger
logger = logging.getLogger('CHECK')


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


# %% DECORATOR DEFINITIONS
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


# %% FUNCTION DEFINITIONS
# Function for checking if a bool has been provided
@docstring_append(check_bool_doc)
def check_bool(value, name):
    # Check if bool is provided and return if so
    if(str(value).lower() in ('false', '0')):
        return(0)
    elif(str(value).lower() in ('true', '1')):
        return(1)
    else:
        logger.error("Input argument '%s' is not of type 'bool'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'bool'!" % (name))


# Function for checking if emulator system is compatible with PRISM version
def check_compatibility(emul_version):
    """
    Checks if the provided `emul_version` is compatible with the current
    version of PRISM.
    Raises a :class:`~RequestError` if *False* and indicates which version of
    PRISM still supports the provided `emul_version`.

    """

    # Do some logging
    logger.info("Performing version compatibility check.")

    # Loop over all compatibility versions
    for version in compat_version:
        # If a compat_version is the same or newer than the emul_version
        # then it is incompatible
        if _compare_versions(version, emul_version):
            logger.error("The provided emulator system is incompatible with "
                         "the current version of PRISM (v%s). The last "
                         "compatible version is v%s."
                         % (prism_version, version))
            raise RequestError("The provided emulator system is incompatible "
                               "with the current version of PRISM (v%s). The"
                               " last compatible version is v%s."
                               % (prism_version, version))

    # Check if emul_version is not newer than prism_version
    if not _compare_versions(prism_version, emul_version):
        logger.error("The provided emulator system was constructed with a "
                     "version later than the current version of PRISM (v%s). "
                     "Use v%s or later to use this emulator system."
                     % (prism_version, emul_version))
        raise RequestError("The provided emulator system was constructed with "
                           "a version later than the current version of PRISM "
                           "(v%s). Use v%s or later to use this emulator"
                           " system."
                           % (prism_version, emul_version))
    else:
        logger.info("Version compatibility check was successful.")


# Function for checking if a finite value has been provided
@docstring_append(check_fin_doc)
def check_finite(value, name):
    # Check if finite value is provided and return if so
    if np.isfinite(value):
        return(value)
    else:
        logger.error("Input argument '%s' is not finite!" % (name))
        raise ValueError("Input argument '%s' is not finite!" % (name))


# Function for checking if a float has been provided
@docstring_append(check_type_doc % ("a float"))
def check_float(value, name):
    # Check if finite value is provided
    value = check_finite(value, name)

    # Check if float is provided and return if so
    if isinstance(value, (int, float, np.integer, np.floating)):
        return(value)
    else:
        logger.error("Input argument '%s' is not of type 'float'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'float'!" % (name))


# Function for checking if an int has been provided
@docstring_append(check_type_doc % ("an integer"))
def check_int(value, name):
    # Check if finite value is provided
    value = check_finite(value, name)

    # Check if int is provided and return if so
    if isinstance(value, (int, np.integer)):
        return(value)
    else:
        logger.error("Input argument '%s' is not of type 'int'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'int'!" % (name))


# Function for checking if a provided float is negative
@docstring_append(check_val_doc % ("a negative float"))
def check_neg_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is negative
    if(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not negative!" % (name))
        raise ValueError("Input argument '%s' is not negative!" % (name))


# Function for checking if a provided int is negative
@docstring_append(check_val_doc % ("a negative integer"))
def check_neg_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is negative
    if(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not negative!" % (name))
        raise ValueError("Input argument '%s' is not negative!" % (name))


# Function for checking if a provided float is non-negative
@docstring_append(check_val_doc % ("a non-negative float"))
def check_nneg_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-negative
    if not(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-negative!" % (name))
        raise ValueError("Input argument '%s' is not non-negative!" % (name))


# Function for checking if a provided int is non-negative
@docstring_append(check_val_doc % ("a non-negative integer"))
def check_nneg_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-negative
    if not(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-negative!" % (name))
        raise ValueError("Input argument '%s' is not non-negative!" % (name))


# Function for checking if a provided float is non-positive
@docstring_append(check_val_doc % ("a non-positive float"))
def check_npos_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-positive
    if not(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-positive!" % (name))
        raise ValueError("Input argument '%s' is not non-positive!" % (name))


# Function for checking if a provided int is non-positive
@docstring_append(check_val_doc % ("a non-positive integer"))
def check_npos_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-positive
    if not(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-positive!" % (name))
        raise ValueError("Input argument '%s' is not non-positive!" % (name))


# Function for checking if a provided float is non-zero
@docstring_append(check_val_doc % ("a non-zero float"))
def check_nzero_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-zero
    if not(value == 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-zero!" % (name))
        raise ValueError("Input argument '%s' is not non-zero!" % (name))


# Function for checking if a provided int is non-zero
@docstring_append(check_val_doc % ("a non-zero integer"))
def check_nzero_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-zero
    if not(value == 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-zero!" % (name))
        raise ValueError("Input argument '%s' is not non-zero!" % (name))


# Function for checking if a provided float is positive
@docstring_append(check_val_doc % ("a positive float"))
def check_pos_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is positive
    if(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not positive!" % (name))
        raise ValueError("Input argument '%s' is not positive!" % (name))


# Function for checking if a provided int is positive
@docstring_append(check_val_doc % ("a positive integer"))
def check_pos_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is positive
    if(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not positive!" % (name))
        raise ValueError("Input argument '%s' is not positive!" % (name))


# Function for checking if a str has been provided
@docstring_append(check_type_doc % ("a string"))
def check_str(value, name):
    # Check if str is provided and return if so
    if isinstance(value, (str, np.string_, unicode)):
        return(value)
    else:
        logger.error("Input argument '%s' is not of type 'str'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'str'!" % (name))


# Function for converting a string sequence to a sequence of elements
def convert_str_seq(seq):
    """
    Converts a provided sequence to a string, removes all auxiliary characters
    from it and splits it up into individual elements.

    """

    # Convert sequence to a string
    seq = str(seq)

    # Remove all unwanted characters from the string
    for char in aux_char_list:
        seq = seq.replace(char, ' ')

    # Split sequence up into elements
    seq = seq.split()

    # Return it
    return(seq)


# Define function that can move the logging file of PRISM and restart logging
def move_logger(working_dir, filename):
    """
    Moves the logging file `filename` from the current working directory to the
    given `working_dir`, and then restarts it again.

    Parameters
    ----------
    working_dir : str
        String containing the directory the log-file needs to be moved to.
    filename : str
        String containing the name of the log-file that needs to be moved.

    """

    # Shut down logging process to allow the log-file to be moved
    logging.shutdown()

    # Get source and destination paths
    source = path.abspath(filename)
    destination = path.join(working_dir, 'prism_log.log')

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
def start_logger(filename=None, mode='w'):
    """
    Opens a logging file called `filename` in the current working directory,
    opened with `mode` and starts the logger.

    Optional
    --------
    filename : str or None. Default: None
        String containing the name of the log-file that is opened.
        If *None*, a new log-file will be created.
    mode : {'r', 'r+', 'w', 'w-'/'x', 'a'}. Default: 'w'
        String indicating how the log-file needs to be opened.

    """

    # If filename is not defined, make a new one
    if filename is None:
        fd, filename = mkstemp('.log', 'prism_log_', '.')
        os.close(fd)

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

    # Return log-file name
    return(filename)


# %% LIST DEFINITIONS
aux_char_list = ['(', ')', '[', ']', ',', "'", '"', '|', '/', '{', '}', '<',
                 '>', '´', '¨', '`']
