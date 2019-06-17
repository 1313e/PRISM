# -*- coding: utf-8 -*-

"""
Internal
========
Contains a collection of support classes/functions for the *PRISM* package.

"""


# %% IMPORTS
# Built-in imports
import logging
import logging.config
import os
from os import path
import platform
import shutil
from struct import calcsize
from tempfile import mkstemp
from textwrap import dedent

# Package imports
from e13tools import InputError, compare_versions
from e13tools.utils import raise_error, raise_warning
import h5py
from mpi4pyd import MPI
import numpy as np
from pkg_resources import get_distribution

# PRISM imports
from prism.__version__ import __version__, compat_version

# All declaration
__all__ = ['CFilter', 'FeatureWarning', 'PRISM_Logger', 'RFilter',
           'RequestError', 'RequestWarning', 'check_compatibility',
           'check_vals', 'get_PRISM_File', 'get_formatter', 'get_handler',
           'get_info', 'getCLogger', 'getLogger', 'getRLogger', 'move_logger',
           'np_array', 'set_base_logger']

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

    def __init__(self, MPI_rank):
        self.is_controller = 1 if not MPI_rank else 0
        super().__init__('CFilter')

    def filter(self, record):
        return(self.is_controller)


# Define Warning class for when an experimental feature is being used
class FeatureWarning(FutureWarning):
    """
    Generic warning raised for experimental features in *PRISM*.

    General purpose warning class, raised whenever a feature is used that
    should be considered experimental. Its behavior and API are subject to
    change, or the entire feature may be removed without a deprecation period.

    """

    pass


# Define custom Logger class that allows for filters to be easily used
class PRISM_Logger(logging.Logger):
    """
    Special :class:`~logging.Logger` class that allows for special filters to
    be set more easily.

    """

    # Initialize Logger
    def __init__(self, *args, **kwargs):
        # Call super constructor
        super().__init__(*args, **kwargs)

        # Initialize different custom filters
        self.initialize_filters()

    # This function initializes custom filters
    def initialize_filters(self):
        self.PRISM_filters = {
            'CFilter': CFilter(rank),
            'RFilter': RFilter(rank)}

    # This function adds requested filters to Logger
    def set_filters(self, filters):
        # If filters is not None, add all filters to Logger
        if filters is not None:
            for filter in filters:
                self.addFilter(self.PRISM_filters[filter])


# Make a custom Filter class that logs the rank of the process that calls it
class RFilter(logging.Filter):
    """
    Custom :class:`~logging.Filter` class that prepends the rank of the MPI
    process that calls it to the logging message. If the size of the used MPI
    intra-communicator is 1, this filter does nothing.

    """

    def __init__(self, MPI_rank):
        if(MPI.__package__ == 'mpi4py' and size > 1):
            self.prefix = "Rank %i: " % (MPI_rank)
        else:
            self.prefix = ""
        super().__init__('RFilter')

    def filter(self, record):
        record.msg = "".join([self.prefix, record.msg])
        return(1)


# Define Exception class for when a requested action is not possible
class RequestError(Exception):
    """
    Generic exception raised for invalid action requests in the *PRISM*
    pipeline.

    General purpose exception class, raised whenever a requested action cannot
    be executed due to it not being allowed or possible in the current state of
    the :obj:`~prism.Pipeline` instance.

    """

    pass


# Define Warning class for when a (future) requested action may not be useful
class RequestWarning(UserWarning):
    """
    Generic warning raised for (future) action requests in the *PRISM* pipeline
    that may not be useful.

    General purpose warning class, raised whenever a requested action may not
    produce appropriate or expected results due to the current state of the
    :obj:`~prism.Pipeline` instance. It is also raised if an obtained result
    can lead to such an action in the future.

    """

    pass


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
    logger = getCLogger('COMP_CHECK')
    logger.info("Performing version compatibility check.")

    # Loop over all compatibility versions
    for version in compat_version:
        # If a compat_version is the same or newer than the emul_version
        # then it is incompatible
        if compare_versions(version, emul_version):
            err_msg = ("The provided emulator is incompatible with the current"
                       " version of PRISM (v%s). The last compatible version "
                       "is v%s." % (__version__, version))
            raise_error(err_msg, RequestError, logger)

    # Check if emul_version is 1.0.x and raise warning if so
    if not compare_versions(emul_version, '1.1.0'):
        warn_msg = ("The provided emulator was constructed with an "
                    "unmaintained version of PRISM (v%s). Compatibility with "
                    "the current version of PRISM cannot be guaranteed."
                    % (emul_version))
        raise_warning(warn_msg, RequestWarning, logger, 2)

    # Check if emul_version is not newer than prism_version
    if not compare_versions(__version__, emul_version):
        err_msg = ("The provided emulator was constructed with a version later"
                   " than the current version of PRISM (v%s). Use v%s or later"
                   " to use this emulator." % (__version__, emul_version))
        raise_error(err_msg, RequestError, logger)
    else:
        logger.info("Version compatibility check was successful.")


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
    values : array_like of {int, float, complex, str, bool}
        The values to be checked against all given criteria in `args`. It must
        be possible to convert `values` to a :obj:`~numpy.ndarray` object.
    name : str
        The name of the input argument, which is used in the error message if
        a criterion is not met.
    args : tuple of {'bool', 'complex', 'float', 'int', 'neg', 'nneg', \
        'normal', 'npos', 'nzero', 'pos', 'str'}
        Sequence of strings determining the criteria that `values` must meet.
        If `args` is empty, it is checked if `values` are finite.

    Returns
    -------
    return_values : array_like of {int, float, complex, str}
        If `args` contained 'bool', returns 0s or 1s. Else, returns `values`.

    Notes
    -----
    If `values` contains integers, but `args` contains 'float', `return_values`
    will be cast as float.

    """

    # Define logger
    logger = getRLogger('CHECK')

    # Convert args to a list
    args = list(args)

    # Check type of values
    if isinstance(values, tuple):
        arr_type = 'tuple'
    elif isinstance(values, list):
        arr_type = 'list'
    elif isinstance(values, np.ndarray):
        arr_type = 'ndarray'
    elif np.isscalar(values):
        arr_type = 'scalar'
    else:
        err_msg = "Input argument %r is not array_like!" % (name)
        raise_error(err_msg, InputError, logger)

    # Convert values to a NumPy array
    try:
        values = np.asanyarray(values)
    except Exception as error:
        err_msg = ("Input argument %r cannot be converted to a NumPy array! "
                   "(%s)" % (name, error))
        raise_error(err_msg, InputError, logger)
    else:
        # Since NumPy v1.16.0, sequenced lists can be converted to NumPy arrays
        # So, check if the dtype is not np.object_
        if issubclass(values.dtype.type, np.object_):
            err_msg = ("Input argument %r cannot be a sequenced container!"
                       % (name))
            raise_error(err_msg, InputError, logger)

    # Check if values is not empty and raise error if so
    if not values.size:
        err_msg = "Input argument %r is empty!" % (name)
        raise_error(err_msg, InputError, logger)

    # Loop over all criteria
    while len(args):
        # Check for bool
        if 'bool' in args:
            # Convert values to str
            values = np.char.lower(np.asanyarray(values, dtype=str))

            # Check if values available are accepted as bools
            check_list = np.zeros_like(values, dtype=int, subok=False)
            check_list[values == '0'] = 1
            check_list[values == 'false'] = 1
            values[values == 'false'] = '0'
            check_list[values == '1'] = 1
            check_list[values == 'true'] = 1
            values[values == 'true'] = '1'

            # Check if check_list solely contains 1s
            if not check_list.all():
                # If not, raise error
                index = np.unravel_index(np.argmin(check_list), values.shape)
                err_msg = ("Input argument '%s%s' is not of type 'bool'!"
                           % (name, list(index) if values.ndim != 0 else ''))
                raise_error(err_msg, TypeError, logger)
            else:
                # If so, convert values to integers and break the loop
                values = np.asanyarray(values, dtype=int)
                break

        # Check for string
        elif 'str' in args:
            # Check if str is provided and break if so
            if issubclass(values.dtype.type, str):
                break
            else:
                err_msg = "Input argument %r is not of type 'str'!" % (name)
                raise_error(err_msg, TypeError, logger)

        # Check for complex
        elif 'complex' in args:
            # Check if complex is provided and continue if so
            if issubclass(values.dtype.type, (np.integer, np.floating,
                                              np.complexfloating)):
                # Remove 'complex' from args and check it again
                args.remove('complex')
                values = np.asanyarray(values, dtype=complex)
                continue
            else:
                err_msg = ("Input argument %r is not of type 'complex'!"
                           % (name))
                raise_error(err_msg, TypeError, logger)

        # Check for float
        elif 'float' in args:
            # Check if float is provided and continue if so
            if issubclass(values.dtype.type, (np.integer, np.floating)):
                # Remove 'float' from args and check it again
                args.remove('float')
                values = np.asanyarray(values, dtype=float)
                continue
            else:
                err_msg = "Input argument %r is not of type 'float'!" % (name)
                raise_error(err_msg, TypeError, logger)

        # Check for integer
        elif 'int' in args:
            # Check if int is provided and continue if so
            if issubclass(values.dtype.type, np.integer):
                # Remove 'int' from args and check it again
                args.remove('int')
                continue
            else:
                err_msg = "Input argument %r is not of type 'int'!" % (name)
                raise_error(err_msg, TypeError, logger)

        # Check for negative value
        elif 'neg' in args:
            # Check if value is negative and continue if so
            try:
                index = list(np.argwhere(values >= 0)[0])
            except IndexError:
                args.remove('neg')
                continue
            else:
                err_msg = ("Input argument '%s%s' is not negative!"
                           % (name, index if values.ndim != 0 else ''))
                raise_error(err_msg, ValueError, logger)

        # Check for non-negative value
        elif 'nneg' in args:
            # Check if value is non-negative and continue if so
            try:
                index = list(np.argwhere(values < 0)[0])
            except IndexError:
                args.remove('nneg')
                continue
            else:
                err_msg = ("Input argument '%s%s' is not non-negative!"
                           % (name, index if values.ndim != 0 else ''))
                raise_error(err_msg, ValueError, logger)

        # Check for normalized value [-1, 1]
        elif 'normal' in args:
            # Check if value is normal and continue if so
            try:
                index = list(np.argwhere(abs(values) > 1)[0])
            except IndexError:
                args.remove('normal')
                continue
            else:
                err_msg = ("Input argument '%s%s' is not normalized!"
                           % (name, index if values.ndim != 0 else ''))
                raise_error(err_msg, ValueError, logger)

        # Check for non-positive value
        elif 'npos' in args:
            # Check if value is non-positive and continue if so
            try:
                index = list(np.argwhere(values > 0)[0])
            except IndexError:
                args.remove('npos')
                continue
            else:
                err_msg = ("Input argument '%s%s' is not non-positive!"
                           % (name, index if values.ndim != 0 else ''))
                raise_error(err_msg, ValueError, logger)

        # Check for non-zero value
        elif 'nzero' in args:
            # Check if value is non-zero and continue if so
            try:
                index = list(np.argwhere(values == 0)[0])
            except IndexError:
                args.remove('nzero')
                continue
            else:
                err_msg = ("Input argument '%s%s' is not non-zero!"
                           % (name, index if values.ndim != 0 else ''))
                raise_error(err_msg, ValueError, logger)

        # Check for positive value
        elif 'pos' in args:
            # Check if value is positive and continue if so
            try:
                index = list(np.argwhere(values <= 0)[0])
            except IndexError:
                args.remove('pos')
                continue
            else:
                err_msg = ("Input argument '%s%s' is not positive!"
                           % (name, index if values.ndim != 0 else ''))
                raise_error(err_msg, ValueError, logger)

        # If none of the criteria is found, the criteria are invalid
        else:
            err_msg = ("Input argument 'args' contains invalid elements (%s)!"
                       % (args))
            raise_error(err_msg, ValueError, logger)

    # If no criteria are left, it must be a finite value
    else:
        # Check if value is finite and continue if so
        try:
            index = list(np.argwhere(~np.isfinite(values))[0])
        except IndexError:
            pass
        except TypeError:
            err_msg = ("Input argument '%s%s' is not of type 'int' or 'float'!"
                       % (name, index if values.ndim != 0 else ''))
            raise_error(err_msg, TypeError, logger)
        else:
            err_msg = ("Input argument '%s%s' is not finite!"
                       % (name, index if values.ndim != 0 else ''))
            raise_error(err_msg, ValueError, logger)

    # Convert values back to its original type
    if(arr_type == 'tuple'):
        values = tuple(values.tolist())
    elif(arr_type == 'list'):
        values = values.tolist()
    elif(arr_type == 'scalar'):
        values = values.item()

    # Return values
    return(values)


# This function returns a logging.Formatter used for PRISM logging
def get_formatter():
    """
    Returns a :obj:`~logging.Formatter` object containing the default logging
    formatting.

    """

    # Set formatting strings
    fmt = "[%(asctime)s][%(levelname)-4s] %(name)-10s \t%(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Initialize Formatter class and return it
    return(logging.Formatter(fmt, datefmt))


# This function returns a logging.Handler used for PRISM logging
def get_handler(filename):
    """
    Returns a :obj:`~logging.Handler` object containing the default logging
    handling settings.

    """

    # Initialize Handler class
    handler = logging.FileHandler(filename, mode='a', encoding='utf-8')

    # Add name to handler
    handler.set_name('prism_base')

    # Set logLevel to DEBUG
    handler.setLevel('DEBUG')

    # Add formatter to handler
    handler.setFormatter(get_formatter())

    # Return handler
    return(handler)


# Define function that returns a string with all PRISM package information
def get_info():
    """
    Prints a string that gives an overview of all information relevant to the
    PRISM package distribution.

    """

    # Create info list
    info_list = []

    # Add header to info_list
    info_list.append(dedent("""
        Configuration
        -------------"""))

    # Add platform to info_list
    info_list.append("Platform: %s %i-bit"
                     % (platform.system(), calcsize('P')*8))

    # Add python version to info_list
    info_list.append("Python: %s" % (platform.python_version()))

    # Add PRISM version to info_list
    info_list.append("Version: %s" % (__version__))

    # Access PRISM metadata
    prism_dist = get_distribution('prism')

    # Get list of all PRISM requirements
    req_list = [req.name for req in prism_dist.requires()]

    # Sort the requirements list
    req_list.sort()

    # Make requirements header
    info_list.append(dedent("""
        Requirements
        ------------"""))

    # Get distribution version of every requirement of PRISM
    for req in req_list:
        dist = get_distribution(req)
        info_list.append("%s: %s" % (req, dist.version))

    # Combine all strings in info_list to info_str
    info_str = '\n'.join(info_list)

    # Print info_str, stripping any additional whitespaces
    print(info_str.strip())


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
        :obj:`~prism.Pipeline` instance.

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

            # Save emul_s as a property
            self.emul_s = emul_s

            # Set default settings
            hdf5_kwargs = {'driver': None,
                           'libver': 'earliest'}

            # Check emul_s and obtain proper logger
            if self.emul_s is None:
                # Only controller opens master file for writing, so use CLogger
                sub_str = ''
                logger = getCLogger('M-HDF5')
            else:
                sub_str = '_%i' % (self.emul_s)
                logger = getRLogger('S-HDF5')

            # Add sub_str to filename
            parts = path.splitext(prism_hdf5_file)
            filename = ''.join([parts[0], sub_str, parts[1]])

            # Update hdf5_kwargs with provided ones
            hdf5_kwargs.update(kwargs)

            # Log that an HDF5-file is being opened
            if self.emul_s is None:
                logger.info("Opening master HDF5-file (mode: %r)." % (mode))
            else:
                logger.info("Opening system HDF5-file %i (mode: %r)."
                            % (self.emul_s, mode))

            # Inheriting File __init__()
            super().__init__(filename, mode, **hdf5_kwargs)

        # Override __exit__() to include logging
        def __exit__(self, *args):
            # Log that an HDF5-file will be closed
            if self.emul_s is None:
                logger = getCLogger('M-HDF5')
                logger.info("Closing master HDF5-file.")
            else:
                logger = getRLogger('S-HDF5')
                logger.info("Closing system HDF5-file %i." % (self.emul_s))

            # Inheriting File __exit__()
            super().__exit__(*args)

    # Return PRISM_File class definition
    return(PRISM_File)


# Define custom getLogger function that adds the CFilter
def getCLogger(name=None):
    """
    Creates a :obj:`~PRISM_Logger` instance with `name`, adds the
    :class:`~CFilter` to it and returns it.

    """

    # Create PRISM_Logger with a CFilter
    return(getLogger(name, ['CFilter']))


# Define custom getLogger function that automatically names loggers correctly
def getLogger(name=None, filters=None):
    """
    Creates a :obj:`~PRISM_Logger` instance with `name` and adds the provided
    `filters` to it. The returned :obj:`~PRISM_Logger` instance is a child of
    the base :class:`~PRISM_Logger` created with :func:`~set_base_logger`, but
    has its name changed (such that the parent name does not show up in the
    log-file).

    Optional
    --------
    name : str or None. Default: None
        The name of the :obj:`~PRISM_Logger` instance to create.
        If *None*, initialize the base :class:`~PRISM_Logger` instead.
    filters : list of str or None. Default: None
        List of strings naming the filters that must be applied to the created
        :obj:`~PRISM_Logger` instance.
        If *None* or the :obj:`~PRISM_Logger` instance already existed, no
        filters will be applied.

    Returns
    -------
    logger : :obj:`~PRISM_Logger` object
        The created :obj:`~PRISM_Logger` instance.

    """

    # Set Logger name prefix
    prefix = 'prism'

    # Check what the provided name is
    if name is None:
        child_name = prefix
        name = 'PRISM_ROOT'
    else:
        child_name = ".".join([prefix, name])

    # Temporarily set default Logger class to PRISM_Logger and initialize it
    logging.setLoggerClass(PRISM_Logger)
    logger = logging.getLogger(child_name)
    logging.setLoggerClass(logging.Logger)

    # Set name and filters if this logger did not already exist
    if(logger.name != name):
        # Remove prefix from the name of the PRISM_Logger instance
        logger.name = name

        # Set the requested filter(s)
        logger.set_filters(filters)

    # Return it
    return(logger)


# Define custom getLogger function that adds the RFilter
def getRLogger(name=None):
    """
    Creates a :obj:`~PRISM_Logger` instance with `name`, adds the
    :class:`~RFilter` to it and returns it.

    """

    # Create PRISM_Logger with an RFilter
    return(getLogger(name, ['RFilter']))


# Define function that can move the logging file of PRISM and restart logging
def move_logger(working_dir):
    """
    Moves the base :class:`~PRISM_Logger` from the current working directory to
    the given `working_dir`, and then restarts it again.

    Parameters
    ----------
    working_dir : str
        String containing the directory the log-file needs to be moved to.

    """

    # Shut down logging process to allow the log-file to be moved
    logging.shutdown()

    # Get source and destination paths
    source = logging.getLogger('prism').handlers[0].baseFilename
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
    set_base_logger(filename=destination)


# This function automatically does not make a copy of a NumPy array
def np_array(obj, *args, **kwargs):
    """
    Returns ``np.array(obj, *args, copy=False, **kwargs)``.

    """

    # Return NumPy array with copy=False
    copy = kwargs.pop('copy', False)
    return(np.array(obj, *args, copy=copy, **kwargs))


# This function sets the base PRISM logger
# TODO: Make base logger unique to Pipeline instance
# This requires a lot of rewriting and many functions to be moved to Pipeline
def set_base_logger(filename=None):
    """
    Initializes the base :class:`~PRISM_Logger`, from which all other
    :obj:`~PRISM_Logger` instances are derived.

    Optional
    --------
    filename : str or None. Default: None
        String containing the name of the log-file that is opened.
        If *None*, a new log-file will be created.

    """

    # If filename is not defined, make a new one
    if filename is None:
        fd, filename = mkstemp('.log', 'prism_', '.')
        os.close(fd)

    # Initialize base_logger
    base_logger = getLogger()

    # Make sure that base_logger has no handlers
    base_logger.handlers = []

    # Initialize base handler and add it to base_logger
    base_logger.addHandler(get_handler(filename))

    # Set logLevel to the same as the logLevel of the handler
    base_logger.setLevel(base_logger.handlers[0].level)

    # Make sure that the base_logger does not propagate logging messages
    base_logger.propagate = False
