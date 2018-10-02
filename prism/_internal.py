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
import logging
import logging.config
import os
from os import path
import shutil
import sys
from tempfile import mkstemp

# Package imports
from e13tools.core import InputError, compare_versions
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
__all__ = ['CLogger', 'PRISM_File', 'RLogger', 'RequestError', 'aux_char_list',
           'check_compatibility', 'check_instance', 'check_val',
           'convert_str_seq', 'delist', 'docstring_append', 'docstring_copy',
           'docstring_substitute', 'getCLogger', 'getRLogger', 'import_cmaps',
           'move_logger', 'raise_error', 'rprint', 'start_logger']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str

# Determine MPI ranks
rank = MPI.COMM_WORLD.Get_rank()


# %% CLASS DEFINITIONS
# Define custom Logger class that only logs if the controller calls it
class CLogger(logging.Logger):
    """
    Custom :class:`~logging.Logger` class that only allows the controller rank
    to log messages to the logfile. Calls from worker ranks are ignored.

    """

    # Set the manager of this class to the default one
    manager = logging.Logger.manager

    def __init__(self, *args, **kwargs):
        self.is_controller = 1 if not rank else 0
        super(CLogger, self).__init__(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).exception(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).critical(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_controller:
            super(CLogger, self).log(*args, **kwargs)


# Override h5py's File.__init__() and __exit__() methods
class PRISM_File(h5py.File):
    """
    Custom :class:`~h5py.File` class that automatically knows where all *PRISM*
    HDF5-files are located if it is initialized by a :obj:`~Pipeline` object.
    Additionally, certain keyword arguments have default values and the
    opening/closing of an HDF5-file is logged.

    """

    # Add hdf5_file attribute
    _hdf5_file = None

    # Override __init__() to include default settings and logging
    def __init__(self, mode, emul_s=None, filename=None, **kwargs):
        """
        Opens the master HDF5-file `filename` in `mode` according to some set
        of default parameters.

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
        filename : str. Default: None
            The name/path of the master HDF5-file that needs to be opened in
            `working_dir`. Default is to open the master HDF5-file that was
            provided during class initialization.
        **kwargs : dict. Default: ``{'driver': None, 'libver': 'earliest'}``
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

        # Check filename
        if filename is None:
            filename = self._hdf5_file
        else:
            pass

        # Add sub_str to filename
        filename = filename.replace('.', '%s.' % (sub_str))

        # Update hdf5_kwargs with provided ones
        hdf5_kwargs.update(kwargs)

        # Log that an HDF5-file is being opened
        logger.info("Opening HDF5-file '%s' (mode: '%s')."
                    % (path.basename(filename), mode))

        # Inheriting File __init__()
        super(PRISM_File, self).__init__(filename, mode, **hdf5_kwargs)

    # Override __exit__() to include logging
    def __exit__(self, *args, **kwargs):
        # Log that an HDF5-file will be closed
        logger = getRLogger('HDF5-FILE')

        # Log about closing the file
        logger.info("Closing HDF5-file '%s'." % (path.basename(self.filename)))

        # Inheriting File __exit__()
        super(PRISM_File, self).__exit__(*args, **kwargs)


# Define custom Logger class that logs the rank of process that calls it
class RLogger(logging.Logger):
    """
    Custom :class:`~logging.Logger` class that prepends the rank of the MPI
    process that calls it to the logging message.

    """

    # Set the manager of this class to the default one
    manager = logging.Logger.manager

    def __init__(self, *args, **kwargs):
        self.prefix = "Rank %i:" % (rank)
        super(RLogger, self).__init__(*args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        msg = " ".join([self.prefix, msg])
        super(RLogger, self).log(level, msg, *args, **kwargs)


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


# This function checks if the input value meets all given criteria
def check_val(value, name, *args):
    """
    Checks if provided input argument `name` of `value` meets all criteria
    given in `args`. If no criteria are given, it is checked if `value` is
    finite.
    Returns `value` (0 or 1 in case of bool) if *True* and raises a
    :class:`~ValueError` or :class:`~TypeError` if *False*.

    Parameters
    ----------
    value : int, float, str, bool
        The value to be checked against all given criteria in `args`.
    name : str
        The name of the input argument, which is used in the error message if
        a criterion is not met.
    args : {'bool', 'float', 'int', 'neg', 'nneg', 'npos', 'nzero', 'pos',\
           'str'}
        Sequence of strings determining the criteria that `value` must meet. If
        `args` is empty, it is checked if `value` is finite.

    Returns
    -------
    return_value : int, float, str
        If `args` contained 'bool', returns 0 or 1. Else, returns `value`.

    """

    # Define logger
    logger = getRLogger('CHECK')

    # Convert args to a list
    args = list(args)

    # Check for bool
    if 'bool' in args:
        # Check if bool is provided and return if so
        if(str(value).lower() in ('false', '0')):
            return(0)
        elif(str(value).lower() in ('true', '1')):
            return(1)
        else:
            err_msg = "Input argument '%s' is not of type 'bool'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for string
    elif 'str' in args:
        # Check if str is provided and return if so
        if isinstance(value, (str, np.string_, unicode)):
            return(value)
        else:
            err_msg = "Input argument '%s' is not of type 'str'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for floats
    elif 'float' in args:
        # Remove 'float' from args and check it again
        args.remove('float')
        value = check_val(value, name, *args)

        # Check if float is provided and return if so
        if isinstance(value, (int, float, np.integer, np.floating)):
            return(value)
        else:
            err_msg = "Input argument '%s' is not of type 'float'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for integers
    elif 'int' in args:
        # Remove 'int' from args and check it again
        args.remove('int')
        value = check_val(value, name, *args)

        # Check if int is provided and return if so
        if isinstance(value, (int, np.integer)):
            return(value)
        else:
            err_msg = "Input argument '%s' is not of type 'int'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for negative values
    elif 'neg' in args:
        # Remove 'neg' from args and check it again
        args.remove('neg')
        value = check_val(value, name, *args)

        # Check if value is negative and return if so
        if(value < 0):
            return(value)
        else:
            err_msg = "Input argument '%s' is not negative!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-negative values
    elif 'nneg' in args:
        # Remove 'nneg' from args and check it again
        args.remove('nneg')
        value = check_val(value, name, *args)

        # Check if value is non-negative and return if so
        if not(value < 0):
            return(value)
        else:
            err_msg = "Input argument '%s' is not non-negative!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-positive values
    elif 'npos' in args:
        # Remove 'npos' from args and check it again
        args.remove('npos')
        value = check_val(value, name, *args)

        # Check if value is non-positive and return if so
        if not(value > 0):
            return(value)
        else:
            err_msg = "Input argument '%s' is not non-positive!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-zero values
    elif 'nzero' in args:
        # Remove 'nzero' from args and check it again
        args.remove('nzero')
        value = check_val(value, name, *args)

        # Check if value is non-zero and return if so
        if not(value == 0):
            return(value)
        else:
            err_msg = "Input argument '%s' is not non-zero!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for positive values
    elif 'pos' in args:
        # Remove 'pos' from args and check it again
        args.remove('pos')
        value = check_val(value, name, *args)

        # Check if value is positive and return if so
        if(value > 0):
            return(value)
        else:
            err_msg = "Input argument '%s' is not positive!" % (name)
            raise_error(ValueError, err_msg, logger)

    # If no criteria are given, it must be a finite value
    elif not len(args):
        # Check if finite value is provided and return if so
        try:
            if np.isfinite(value):
                return(value)
        except Exception:
            pass
        err_msg = "Input argument '%s' is not finite!" % (name)
        raise_error(ValueError, err_msg, logger)

    # If none of the criteria is found, the criteria are invalid
    else:
        err_msg = "Input argument 'args' is invalid!"
        raise_error(InputError, err_msg, logger)


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
    if name:
        CLogger.manager.loggerClass = CLogger
        logger = CLogger.manager.getLogger(name)
        CLogger.manager.loggerClass = None
        return(logger)
    else:
        return CLogger.root


# Define custom getLogger function that calls the custom RLogger instead
def getRLogger(name=None):
    if name:
        RLogger.manager.loggerClass = RLogger
        logger = RLogger.manager.getLogger(name)
        RLogger.manager.loggerClass = None
        return(logger)
    else:
        return RLogger.root


# Function to import all custom colormaps in a directory
def import_cmaps(cmap_dir=None):
    """
    Reads in custom colormaps from a provided directory `cmap_dir`, transforms
    them into :obj:`~matplotlib.colors.LinearSegmentedColormap` objects and
    registers them in the :mod:`~matplotlib.cm` module. Both the imported
    colormap and its reversed version will be registered.

    Optional
    --------
    cmap_dir : str or None. Default: None
        If str, relative or absolute path to the directory that contains custom
        colormap files. If *None*, read in colormap files from *PRISM*'s 'data'
        directory.

    Notes
    -----
    All colormap files in `cmap_dir` must have names starting with 'cm_'. The
    resulting colormaps will have the name of their file without the prefix.

    """

    # Obtain path to directory with colormaps
    if cmap_dir is None:
        cmap_dir = path.join(path.dirname(__file__), 'data')
    else:
        cmap_dir = path.abspath(cmap_dir)

    # Obtain the names of all PRISM data files
    filenames = next(os.walk(cmap_dir))[2]
    cm_files = list(filenames)

    # Extract the files with defined colormaps
    for filename in filenames:
        if(filename[0:3] != 'cm_'):
            cm_files.remove(filename)
    cm_files.sort()

    # Read in all the defined colormaps, transform and register them
    for cm_file in cm_files:
        # Determine the index of the extension
        ext_idx = cm_file.rfind('.')

        # Extract name of colormap
        if(ext_idx == -1):
            cm_name = cm_file[3:]
        else:
            cm_name = cm_file[3:ext_idx]

        # Process colormap files
        try:
            # Obtain absolute path to colormap data file
            cm_file_path = path.join(cmap_dir, cm_file)

            # Read in colormap data
            if(ext_idx != -1 and cm_file[ext_idx:] in ('.npy', '.npz')):
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
            raise InputError("Provided colormap '%s' is invalid! (%s)"
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
def rprint(string):
    if(MPI.__package__ == 'mpi4py'):
        print("Rank %s: %s" % (rank, string))
    else:
        print(string)


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
                 '>', '´', '¨', '`', '\\', '?', '!', '%', ':', ';']
