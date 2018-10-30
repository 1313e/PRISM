# -*- coding: utf-8 -*-

"""
Internal
========
Contains a collection of support classes/functions/lists for the *PRISM*
package.

"""


# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
from copy import copy
import logging
import logging.config
import os
from os import path
import shutil
import sys
from tempfile import mkstemp

# Package imports
from e13tools import InputError, compare_versions
import h5py
from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap as LSC
try:
    from mpi4py import MPI
except ImportError:
    import mpi_dummy as MPI
import numpy as np

# PRISM imports
from .__version__ import compat_version, prism_version

# All declaration
__all__ = ['RequestError', 'aux_char_list', 'check_compatibility',
           'check_instance', 'check_vals', 'convert_str_seq', 'delist',
           'docstring_append', 'docstring_copy', 'docstring_substitute',
           'exec_code_anal', 'getCLogger', 'get_PRISM_File', 'getRLogger',
           'import_cmaps', 'move_logger', 'raise_error', 'rprint',
           'start_logger']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str

# Determine MPI size and ranks
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()


# %% CLASS DEFINITIONS
# Make a custom Filter class that only allows the controller to log messages
class CFilter(logging.Filter):
    """
    Custom :class:`~logging.Filter` class that only allows the controller rank
    to log messages to the logfile. Calls from worker ranks are ignored.

    """

    def __init__(self, rank):
        self.is_controller = 1 if not rank else 0

    def filter(self, record):
        return(self.is_controller)


# Define custom Logger class that uses the CFilter filter
class CLogger(logging.Logger):
    """
    Custom :class:`~logging.Logger` class that uses the :class:`~CFilter`.

    """

    # Initialize Logger, adding the CFilter
    def __init__(self, *args, **kwargs):
        super(CLogger, self).__init__(*args, **kwargs)
        self.addFilter(CFilter(rank))


# Make a custom Filter class that logs the rank of the process that calls it
class RFilter(logging.Filter):
    """
    Custom :class:`~logging.Filter` class that prepends the rank of the MPI
    process that calls it to the logging message.

    """

    def __init__(self, rank):
        self.prefix = "Rank %i:" % (rank)

    def filter(self, record):
        record.msg = " ".join([self.prefix, record.msg])
        return(1)


# Define custom Logger class that uses the RFilter filter
class RLogger(logging.Logger):
    """
    Custom :class:`~logging.Logger` class that uses the :class:`~RFilter` if
    the size of the intra-communicator is more than 1.

    """

    # Initialize Logger, adding the RFilter if size > 1
    def __init__(self, *args, **kwargs):
        super(RLogger, self).__init__(*args, **kwargs)
        if(MPI.__package__ == 'mpi4py' and size > 1):
            self.addFilter(RFilter(rank))


# Define Exception class for when a requested action is not possible
class RequestError(Exception):
    """
    Generic exception raised for invalid action requests in the *PRISM*
    pipeline.

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
        raise InputError("Either only positional or keyword arguments are "
                         "allowed!")
    else:
        params = args or kwargs

    def do_substitution(target):
        if target.__doc__:
            target.__doc__ = target.__doc__ % (params)
        else:
            raise InputError("Target has no docstring available for "
                             "substitutions!")
        return(target)
    return(do_substitution)


# %% FUNCTION DEFINITIONS
# Function for checking if emulator system is compatible with PRISM version
def check_compatibility(emul_version):
    """
    Checks if the provided `emul_version` is compatible with the current
    version of *PRISM*.
    Raises a :class:`~RequestError` if *False* and indicates which version of
    *PRISM* still supports the provided `emul_version`.

    """

    # Do some logging
    logger = getRLogger('CHECK')
    logger.info("Performing version compatibility check.")

    # Loop over all compatibility versions
    for version in compat_version:
        # If a compat_version is the same or newer than the emul_version
        # then it is incompatible
        if compare_versions(version, emul_version):
            err_msg = ("The provided emulator is incompatible with the current"
                       " version of PRISM (v%s). The last compatible version "
                       "is v%s." % (prism_version, version))
            raise_error(RequestError, err_msg, logger)

    # Check if emul_version is not newer than prism_version
    if not compare_versions(prism_version, emul_version):
        err_msg = ("The provided emulator was constructed with a version later"
                   " than the current version of PRISM (v%s). Use v%s or later"
                   " to use this emulator." % (prism_version, emul_version))
        raise_error(RequestError, err_msg, logger)
    else:
        logger.info("Version compatibility check was successful.")


# This function checks if a given instance was initialized properly
def check_instance(instance, cls):
    """
    Checks if provided `instance` has been initialized from a proper `cls`
    (sub)class. Raises a :class:`~TypeError` if `instance` is not an instance
    of `cls`.

    Parameters
    ----------
    instance : :obj:`~object`
        Class instance that needs to be checked.
    cls : class
        The class which `instance` needs to be properly initialized from.

    Returns
    -------
    result : bool
        Bool indicating whether or not the provided `instance` was initialized
        from a proper `cls` (sub)class.

    """

    # Check if instance was initialized from a cls (sub)class
    if not isinstance(instance, cls):
        raise TypeError("Input argument 'instance' must be an instance of the "
                        "%s class!" % (cls.__name__))

    # Retrieve a list of all cls properties
    class_props = [prop for prop in dir(cls) if
                   isinstance(getattr(cls, prop), property)]

    # Check if all cls properties can be called in instance
    for prop in class_props:
        try:
            getattr(instance, prop)
        except AttributeError:
            return(0)
    else:
        return(1)


# This function checks if the input values meet all given criteria
def check_vals(values, name, *args):
    """
    Checks if all values in provided input argument `values` with `name` meet
    all criteria given in `args`. If no criteria are given, it is checked if
    all values are finite.
    Returns `values` (0 or 1 in case of bool) if *True* and raises a
    :class:`~ValueError` or :class:`~TypeError` if *False*.

    Parameters
    ----------
    values : array_like of {int, float, str, bool}
        The values to be checked against all given criteria in `args`.
    name : str
        The name of the input argument, which is used in the error message if
        a criterion is not met.
    args : tuple of {'bool', 'float', 'int', 'neg', 'nneg', 'npos', 'nzero', \
        'pos', 'str'}
        Sequence of strings determining the criteria that `values` must meet.
        If `args` is empty, it is checked if `values` are finite.

    Returns
    -------
    return_values : array_like of {int, float, str}
        If `args` contained 'bool', returns 0 or 1. Else, returns `values`.

    Notes
    -----
    If `values` is array_like, every element is replaced by its checked values
    (0s or 1s in case of bools, or ints converted to floats in case of floats).
    Because of this, a copy will be made of `values`. If this is not possible,
    `values` is adjusted in place.

    """

    # Define logger
    logger = getRLogger('CHECK')

    # Convert args to a list
    args = list(args)

    # Check ndim of values and iterate over values if ndim > 0
    if np.ndim(values):
        # If values is a NumPy array, make empty copy and upcast if necessary
        if isinstance(values, np.ndarray):
            if 'bool' in args or 'int' in args:
                values_copy = np.empty_like(values, dtype=int)
            elif 'float' in args:
                values_copy = np.empty_like(values, dtype=float)
            elif 'str' in args:
                values_copy = np.empty_like(values, dtype=str)
            else:
                values_copy = np.empty_like(values)

        # If not a NumPy array, make a normal copy
        else:
            # Check if values has the copy()-method and use it if so
            try:
                values_copy = values.copy()
            # Else, use the built-in copy() method
            except AttributeError:
                values_copy = copy(values)

        # Iterate over first dimension of values
        for idx, value in enumerate(values):
            # Check value
            values_copy[idx] = check_vals(value, '%s[%i]' % (name, idx), *args)

        # Return values
        return(values_copy)

    # If ndim == 0, set value to values
    else:
        value = values

    # Check for bool
    if 'bool' in args:
        # Check if bool is provided and return if so
        if(str(value).lower() in ('false', '0')):
            return(0)
        elif(str(value).lower() in ('true', '1')):
            return(1)
        else:
            err_msg = "Input argument %r is not of type 'bool'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for string
    elif 'str' in args:
        # Check if str is provided and return if so
        if isinstance(value, (str, np.string_, unicode)):
            return(value)
        else:
            err_msg = "Input argument %r is not of type 'str'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for float
    elif 'float' in args:
        # Check if float is provided and return if so
        if isinstance(value, (int, float, np.integer, np.floating)):
            # Remove 'float' from args and check it again
            args.remove('float')
            value = check_vals(value, name, *args)
            return(float(value))
        else:
            err_msg = "Input argument %r is not of type 'float'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for integer
    elif 'int' in args:
        # Check if int is provided and return if so
        if isinstance(value, (int, np.integer)):
            # Remove 'int' from args and check it again
            args.remove('int')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not of type 'int'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for negative value
    elif 'neg' in args:
        # Check if value is negative and return if so
        if(value < 0):
            # Remove 'neg' from args and check it again
            args.remove('neg')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not negative!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-negative value
    elif 'nneg' in args:
        # Check if value is non-negative and return if so
        if not(value < 0):
            # Remove 'nneg' from args and check it again
            args.remove('nneg')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not non-negative!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-positive value
    elif 'npos' in args:
        # Check if value is non-positive and return if so
        if not(value > 0):
            # Remove 'npos' from args and check it again
            args.remove('npos')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not non-positive!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-zero value
    elif 'nzero' in args:
        # Check if value is non-zero and return if so
        if not(value == 0):
            # Remove 'nzero' from args and check it again
            args.remove('nzero')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not non-zero!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for positive value
    elif 'pos' in args:
        # Check if value is positive and return if so
        if(value > 0):
            # Remove 'pos' from args and check it again
            args.remove('pos')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not positive!" % (name)
            raise_error(ValueError, err_msg, logger)

    # If no criteria are given, it must be a finite value
    elif not len(args):
        # Check if finite value is provided and return if so
        try:
            if np.isfinite(value):
                return(value)
        except Exception:
            pass
        err_msg = "Input argument %r is not finite!" % (name)
        raise_error(ValueError, err_msg, logger)

    # If none of the criteria is found, the criteria are invalid
    else:
        err_msg = "Input argument 'args' is invalid!"
        raise_error(InputError, err_msg, logger)


# Function for converting a string sequence to a sequence of elements
def convert_str_seq(seq):
    """
    Converts a provided sequence to a string, removes all auxiliary characters
    from it, splits it up into individual elements and converts all elements
    back to integers, floats and/or strings.

    """

    # Convert sequence to a string
    seq = str(seq)

    # Remove all unwanted characters from the string
    for char in aux_char_list:
        seq = seq.replace(char, ' ')

    # Split sequence up into elements
    seq = seq.split()

    # Loop over all elements in seq
    for i, val in enumerate(seq):
        # Try to convert to int or float
        try:
            # If string contains a dot, check if it is a float
            if '.' in val:
                seq[i] = float(val)
            # If string contains no dot, check if it is an int
            else:
                seq[i] = int(val)
        # If it cannot be converted to int or float, save as string
        except ValueError:
            seq[i] = val

    # Return it
    return(seq)


# Function that returns a copy of a list with all empty lists removed
def delist(list_obj):
    """
    Returns a copy of `list_obj` with all empty lists removed.

    Parameters
    ----------
    list_obj : list
        A list object that requires its empty list elements to be removed.

    Returns
    -------
    delisted_copy : list
        Copy of `list_obj` with all empty lists removed.

    """

    # Check if list_obj is a list
    if(type(list_obj) != list):
        raise TypeError("Input argument 'list_obj' is not of type 'list'!")

    # Make a copy of itself
    delisted_copy = list(list_obj)

    # Remove all empty lists from this copy
    off_dex = len(delisted_copy)-1
    for i, element in enumerate(reversed(delisted_copy)):
        if(isinstance(element, list) and element == []):
            delisted_copy.pop(off_dex-i)

    # Return the copy
    return(delisted_copy)


# Define custom getLogger function that calls the custom CLogger instead
def getCLogger(name=None):
    """
    Create a :class:`~CLogger` logger instance with `name` and return it.

    """

    # Temporarily set the default class to CLogger and return an instance of it
    logging.setLoggerClass(CLogger)
    logger = logging.getLogger(name)
    logging.setLoggerClass(logging.Logger)
    return(logger)


# Define class factory that returns a specialized h5py.File class
def get_PRISM_File(prism_hdf5_file):
    """
    Returns a class definition ``PRISM_File(mode, emul_s=None, **kwargs)``.

    This class definition is a specialized version of the :class:`~h5py.File`
    class with the filename automatically set to `prism_hdf5_file` and added
    logging to the constructor and destructor methods.

    Parameters
    ----------
    prism_hdf5_file : str
        Absolute path to the master HDF5-file that is used in a
        :obj:`~prism.pipeline.Pipeline` instance.

    Returns
    -------
    PRISM_File : class
        Definition of the class ``PRISM_File(mode, emul_s=None, **kwargs)``.

    """

    # Override h5py's File.__init__() and __exit__() methods
    class PRISM_File(h5py.File):
        """
        Custom :class:`~h5py.File` class that automatically knows where all
        *PRISM* HDF5-files are located when created by the
        :func:`~get_PRISM_File` class factory. Additionally, certain keyword
        arguments have default values and the opening/closing of an HDF5-file
        is logged.

        """

        # Override __init__() to include default settings and logging
        def __init__(self, mode, emul_s=None, **kwargs):
            """
            Opens the master HDF5-file `prism_hdf5_file` in `mode` according to
            some set of default parameters.

            Parameters
            ----------
            mode : {'r', 'r+', 'w', 'w-'/'x', 'a'}
                String indicating how the HDF5-file needs to be opened.

            Optional
            --------
            emul_s : int or None. Default: None
                If int, number indicating the requested emulator system file to
                open.
                If *None*, the master HDF5-file itself is opened.
            kwargs : dict. Default: ``{'driver': None, 'libver': 'earliest'}``
                Other keyword arguments that need to be given to the
                :func:`~h5py.File` function.

            """

            # Log that an HDF5-file is being opened
            logger = getRLogger('HDF5-FILE')

            # Set default settings
            hdf5_kwargs = {'driver': None,
                           'libver': 'earliest'}

            # Check emul_s
            if emul_s is None:
                sub_str = ''
            else:
                sub_str = '_%i' % (emul_s)

            # Add sub_str to filename
            parts = path.splitext(prism_hdf5_file)
            filename = ''.join([parts[0], sub_str, parts[1]])

            # Update hdf5_kwargs with provided ones
            hdf5_kwargs.update(kwargs)

            # Log that an HDF5-file is being opened
            logger.info("Opening HDF5-file %r (mode: %r)."
                        % (path.basename(filename), mode))

            # Inheriting File __init__()
            super(PRISM_File, self).__init__(filename, mode, **hdf5_kwargs)

        # Override __exit__() to include logging
        def __exit__(self, *args, **kwargs):
            # Log that an HDF5-file will be closed
            logger = getRLogger('HDF5-FILE')

            # Log about closing the file
            logger.info("Closing HDF5-file %r."
                        % (path.basename(self.filename)))

            # Inheriting File __exit__()
            super(PRISM_File, self).__exit__(*args, **kwargs)

    # Return PRISM_File class definition
    return(PRISM_File)


# Define custom getLogger function that calls the custom RLogger instead
def getRLogger(name=None):
    """
    Create a :class:`~RLogger` logger instance with `name` and return it.

    """

    # Temporarily set the default class to RLogger and return an instance of it
    logging.setLoggerClass(RLogger)
    logger = logging.getLogger(name)
    logging.setLoggerClass(logging.Logger)
    return(logger)


# Function to import all custom colormaps in a directory
def import_cmaps(cmap_dir):
    """
    Reads in custom colormaps from a provided directory `cmap_dir`, transforms
    them into :obj:`~matplotlib.colors.LinearSegmentedColormap` objects and
    registers them in the :mod:`~matplotlib.cm` module. Both the imported
    colormap and its reversed version will be registered.

    This function is called automatically when *PRISM* is imported.

    Parameters
    ----------
    cmap_dir : str
        If str, relative or absolute path to the directory that contains custom
        colormap files.

    Notes
    -----
    All colormap files in `cmap_dir` must have names starting with 'cm_'. The
    resulting colormaps will have the name of their file without the prefix and
    extension.

    """

    # Obtain path to directory with colormaps
    cmap_dir = path.abspath(cmap_dir)

    # Check if provided directory exists
    if not path.exists(cmap_dir):
        raise OSError("Input argument 'cmap_dir' is a non-existing path (%r)!"
                      % (cmap_dir))

    # Obtain the names of all files in cmap_dir
    filenames = next(os.walk(cmap_dir))[2]
    cm_files = []

    # Extract the files with defined colormaps
    for filename in filenames:
        if(filename[:3] == 'cm_'):
            cm_files.append(filename)
    cm_files.sort()

    # Read in all the defined colormaps, transform and register them
    for cm_file in cm_files:
        # Split basename and extension
        base_str, ext_str = path.splitext(cm_file)
        cm_name = base_str[3:]

        # Process colormap files
        try:
            # Obtain absolute path to colormap data file
            cm_file_path = path.join(cmap_dir, cm_file)

            # Read in colormap data
            if ext_str in ('.npy', '.npz'):
                # If file is a NumPy binary file
                colorlist = np.load(cm_file_path).tolist()
            else:
                # If file is anything else
                colorlist = np.genfromtxt(cm_file_path).tolist()

            # Transform colorlist into a Colormap
            cmap = LSC.from_list(cm_name, colorlist, N=len(colorlist))
            cmap_r = LSC.from_list(cm_name+'_r', list(reversed(colorlist)),
                                   N=len(colorlist))

            # Add cmap to matplotlib's cmap list
            register_cmap(cmap=cmap)
            register_cmap(cmap=cmap_r)
        except Exception as error:
            raise InputError("Provided colormap %r is invalid! (%s)"
                             % (cm_name, error))


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


# This function raises a given error after logging the error
def raise_error(err_type, err_msg, logger):
    """
    Raises a given error of type `err_type` with message `err_msg` and logs the
    error using the provided `logger`.

    Parameters
    ----------
    err_type : :class:`~Exception` subclass
        The type of error that needs to be raised.
    err_msg : str
        The message included in the error.
    logger : :obj:`~logging.Logger` object
        The logger to which the error message must be written.

    """

    # Log the error and raise it right after
    logger.error(err_msg)
    raise err_type(err_msg)


# Redefine the print function to include the MPI rank if MPI is used
def rprint(*args, **kwargs):
    """
    Custom :func:`~print` function that prepends the rank of the MPI process
    that calls it to the message if the size of the intra-communicator is more
    than 1.
    Takes the same input arguments as the normal :func:`~print` function.

    """

    # If MPI is used and size > 1, prepend rank to message
    if(MPI.__package__ == 'mpi4py' and size > 1):
        args = list(args)
        args.insert(0, "Rank %i:" % (rank))
    print(*args, **kwargs)


# Define function that can start the logging process of PRISM
# TODO: Make a filter that only allows PRISM log messages to be logged to file
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
                 '>', '´', '¨', '`', '\\', '?', '!', '%', ':', ';', '+', '-',
                 '=', '$', '~', '#', '@', '^', '&', '*']


# %% COMPILED CODE OBJECT DEFINITIONS
# Code tuple for analyzing
_pre_code_anal = compile("", '<string>', 'exec')
_eval_code_anal = compile("", '<string>', 'exec')
_anal_code_anal = compile("", '<string>', 'exec')
_post_code_anal = compile("self.results = sam_set[sam_idx]", '<string>',
                          'exec')
_exit_code_anal = compile("", '<string>', 'exec')

exec_code_anal = (_pre_code_anal, _eval_code_anal, _anal_code_anal,
                  _post_code_anal, _exit_code_anal)
