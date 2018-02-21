# -*- coding: utf-8 -*-

"""
PRISM Internal
==============

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function,\
                       with_statement

# Built-in imports
import os
from os import path
import shutil

# Package imports
import logging
import logging.config

# All declaration
__all__ = ['RequestError', 'docstring_copy', 'move_logger', 'start_logger']


# %% CLASS DEFINITIONS
# Define Exception class for when a requested action is not possible
class RequestError(Exception):
    """
    Generic exception raised for invalid action requests in the
    :class:`~Pipeline` class.

    General purpose exception class, raised whenever a requested action cannot
    be executed due to it not being allowed or possible in the current state of
    the :obj:`~Pipeline` instance.

    """

    pass


# %% FUNCTION DEFINITIONS
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
