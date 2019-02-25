# -*- coding: utf-8 -*-

"""
Internal
========
Contains a collection of support classes/functions/lists for the *PRISM*
package.

"""


# %% IMPORTS
# Built-in imports
from inspect import isclass
import logging
import logging.config
import os
from os import path
import platform
import shutil
from struct import calcsize
from tempfile import mkstemp
from textwrap import dedent
import warnings

# Package imports
from e13tools import InputError, compare_versions
import h5py
from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap as LSC
import numpy as np
from pkg_resources import get_distribution

# PRISM imports
try:
    from mpi4py import MPI
except ImportError:
    import prism._dummyMPI as MPI
from prism.__version__ import compat_version, prism_version

# All declaration
__all__ = ['CFilter', 'FeatureWarning', 'PRISM_Comm', 'PRISM_Logger',
           'RFilter', 'RequestError', 'RequestWarning', 'aux_char_list',
           'check_compatibility', 'check_instance', 'check_vals',
           'convert_str_seq', 'delist', 'docstring_append', 'docstring_copy',
           'docstring_substitute', 'get_PRISM_File', 'get_formatter',
           'get_handler', 'get_info', 'getCLogger', 'getLogger', 'getRLogger',
           'import_cmaps', 'move_logger', 'np_array', 'raise_error',
           'raise_warning', 'rprint', 'set_base_logger']

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


# Make a custom MPI.Comm class that uses a special broadcast method
class PRISM_Comm(object):
    """
    Custom :class:`~MPI.Intracomm` class that automatically makes use of the
    :class:`~numpy.ndarray` buffers when using communications. Is functionally
    the same as the provided `comm` for everything else.

    Optional
    --------
    comm : :obj:`~MPI.Intracomm` object or None. Default: None
        The MPI intra-communicator to use in this :class:`~PRISM_Comm`
        instance.
        If *None*, use :obj:`MPI.COMM_WORLD` instead.

    """

    def __init__(self, comm=None):
        # If comm is None, use MPI.COMM_WORLD
        if comm is None:
            comm = MPI.COMM_WORLD
        # Else, raise error if provided comm is not an MPI intra-communicator
        elif not isinstance(comm, MPI.Intracomm):
            raise TypeError("Input argument 'comm' must be an instance of the "
                            "MPI.Intracomm class!")

        # Bind provided communicator
        self._comm = comm
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    # Override getattr property to use self._comm attributes if necessary
    def __getattribute__(self, name):
        try:
            return(super().__getattribute__(name))
        except AttributeError:
            return(getattr(self._comm, name))

    # Override __dir__ attribute to use the one from self._comm
    def __dir__(self):
        return(dir(self._comm))

    # Specialized bcast function that automatically makes use of NumPy buffers
    def bcast(self, obj, root):
        """
        Special broadcast method that automatically uses the appropriate method
        (:meth:`~MPI.Intracomm.bcast` or :meth:`~MPI.Intracomm.Bcast`)
        depending on the type of the provided `obj`.

        Parameters
        ----------
        obj : :obj:`~numpy.ndarray` or object
            The object to broadcast to all MPI ranks.
            If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Bcast`.
            If not, use :meth:`~MPI.Intracomm.bcast` instead.
        root : int
            The MPI rank that broadcasts `obj`.

        Returns
        -------
        obj : object
            The broadcasted `obj`.

        """

        # Sender
        if(self._rank == root):
            # Check if provided object is a NumPy array
            if isinstance(obj, np.ndarray):
                # If so, send shape and dtype of the NumPy array
                self._comm.bcast(['NumPy ndarray', [obj.shape, obj.dtype]],
                                 root=root)

                # Then send the NumPy array as a buffer object
                self._comm.Bcast(obj, root=root)

            # If not, send obj the normal way
            else:
                # Try to send object
                try:
                    self._comm.bcast([obj.__class__.__name__, obj], root=root)
                # If this fails, raise error about byte size
                except OverflowError:
                    raise InputError("Input argument `obj` has a byte size "
                                     "that cannot be stored in a 32-bit int "
                                     "(%i > %i)!"
                                     % (obj.__sizeof__(), 2**31-1))

        # Receivers wait for instructions
        else:
            # Receive object
            obj_type, obj = self._comm.bcast(obj, root=root)

            # If obj_type is NumPy ndarray, obj contains shape and dtype
            if(obj_type == 'NumPy ndarray'):
                # Create empty NumPy array with given shape and dtype
                obj = np.empty(*obj)

                # Receive NumPy array
                self._comm.Bcast(obj, root=root)

        # Return obj
        return(obj)

    # Specialized gather function that automatically makes use of NumPy buffers
    def gather(self, obj, root):
        """
        Special gather method that automatically uses the appropriate method
        (:meth:`~MPI.Intracomm.gather` or :meth:`~MPI.Intracomm.Gatherv`)
        depending on the type of the provided `obj`.

        Parameters
        ----------
        obj : :obj:`~numpy.ndarray` or object
            The object to gather from all MPI ranks.
            If :obj:`~numpy.ndarray`, use :meth:`~MPI.Intracomm.Gatherv`.
            If not, use :meth:`~MPI.Intracomm.gather` instead.
        root : int
            The MPI rank that gathers `obj`.

        Returns
        -------
        obj : list or None
            If MPI rank is `root`, returns a list of the gathered objects.
            Else, returns *None*.

        Warnings
        --------
        If some but not all MPI ranks use a NumPy array, this method will hang
        indefinitely.
        When gathering NumPy arrays, all arrays must have the same number of
        dimensions and the same shape, except for one axis.

        """

        # Check if provided object is a NumPy array
        if isinstance(obj, np.ndarray):
            # If so, gather the shapes of obj on the receiver
            shapes = self._comm.gather(obj.shape, root=root)

            # If obj has an empty dimension anywhere, replace it with a dummy
            if not np.all(obj.shape):
                obj = np.empty([1]*obj.ndim)

            # Receiver sets up a buffer array and receives NumPy array
            if(self._rank == root):
                # Obtain the required shape of the buffer array
                buff_shape = (self._size, np.product(np.max(shapes, axis=0)))

                # Create buffer array
                buff = np.empty(buff_shape)

                # Gather all NumPy arrays
                self._comm.Gatherv(obj.ravel(), buff, root=root)

                # Make an empty list holding individual arrays
                arr_list = []

                # Loop over gathered buff and transform back to single arrays
                for array, shape in zip(buff, shapes):
                    array_temp = array[:np.product(shape)]
                    arr_list.append(array_temp.reshape(shape))

                # Replace buff by arr_list
                buff = arr_list

            # Senders send the array
            else:
                # Senders set up dummy buffer
                buff = None

                # Send array
                self._comm.Gatherv(obj.ravel(), buff, root=root)

        # If not, gather obj the normal way
        else:
            # Try to send the obj
            try:
                buff = self._comm.gather(obj, root=root)
            # If this fails, raise error about byte size
            except SystemError:
                raise InputError("Input argument 'obj' is too large!")

        # Return buff
        return(buff)


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
    logger = getCLogger('COMP_CHECK')
    logger.info("Performing version compatibility check.")

    # Loop over all compatibility versions
    for version in compat_version:
        # If a compat_version is the same or newer than the emul_version
        # then it is incompatible
        if compare_versions(version, emul_version):
            err_msg = ("The provided emulator is incompatible with the current"
                       " version of PRISM (v%s). The last compatible version "
                       "is v%s." % (prism_version, version))
            raise_error(err_msg, RequestError, logger)

    # Check if emul_version is 1.0.x and raise warning if so
    if not compare_versions(emul_version, '1.1.0'):
        warn_msg = ("The provided emulator was constructed with an "
                    "unmaintained version of PRISM (v%s). Compatibility with "
                    "the current version of PRISM cannot be guaranteed."
                    % (emul_version))
        raise_warning(warn_msg, RequestWarning, logger, 2)

    # Check if emul_version is not newer than prism_version
    if not compare_versions(prism_version, emul_version):
        err_msg = ("The provided emulator was constructed with a version later"
                   " than the current version of PRISM (v%s). Use v%s or later"
                   " to use this emulator." % (prism_version, emul_version))
        raise_error(err_msg, RequestError, logger)
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
    instance : object
        Class instance that needs to be checked.
    cls : class
        The class which `instance` needs to be properly initialized from.

    Returns
    -------
    result : bool
        Bool indicating whether or not the provided `instance` was initialized
        from a proper `cls` (sub)class.

    """

    # Check if cls is a class
    if not isclass(cls):
        raise InputError("Input argument 'cls' must be a class!")

    # Check if instance was initialized from a cls (sub)class
    if not isinstance(instance, cls):
        raise TypeError("Input argument 'instance' must be an instance of the "
                        "%s.%s class!" % (cls.__module__, cls.__name__))

    # Retrieve a list of all cls attributes
    class_attrs = dir(cls)

    # Check if all cls attributes can be called in instance
    for attr in class_attrs:
        try:
            getattr(instance, attr)
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
        The values to be checked against all given criteria in `args`. It must
        be possible to convert `values` to a :obj:`~numpy.ndarray` object.
    name : str
        The name of the input argument, which is used in the error message if
        a criterion is not met.
    args : tuple of {'bool', 'float', 'int', 'neg', 'nneg', 'normal', 'npos', \
        'nzero', 'pos', 'str'}
        Sequence of strings determining the criteria that `values` must meet.
        If `args` is empty, it is checked if `values` are finite.

    Returns
    -------
    return_values : array_like of {int, float, str}
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


# Function for converting a string sequence to a sequence of elements
def convert_str_seq(seq):
    """
    Converts a provided sequence to a string, removes all auxiliary characters
    from it, splits it up into individual elements and converts all elements
    back to integers, floats and/or strings.

    The auxiliary characters are given by the :obj:`~aux_char_list` list. One
    can add, change and remove characters from the list if required.

    Parameters
    ----------
    seq : str or array_like
        The sequence that needs to be converted to individual elements.
        If array_like, `seq` is first converted to a string.

    Returns
    -------
    new_seq : list
        A list with all individual elements converted to integers, floats
        and/or strings.

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
            # If string contains an E or e, check if it is a float
            if 'e' in val.lower():
                seq[i] = float(val)
            # If string contains a dot, check if it is a float
            elif '.' in val:
                seq[i] = float(val)
            # If string contains no dot, E or e, check if it is an int
            else:
                seq[i] = int(val)
        # If it cannot be converted to int or float, save as string
        except ValueError:
            seq[i] = val

    # Return it
    return(seq)


# List of auxiliary characters to be used in convert_str_seq()
aux_char_list = ['(', ')', '[', ']', ',', "'", '"', '|', '/', '{', '}', '<',
                 '>', '´', '¨', '`', '\\', '?', '!', '%', ';', '=', '$', '~',
                 '#', '@', '^', '&', '*']


# Function that returns a copy of a list with all empty lists/tuples removed
def delist(list_obj):
    """
    Returns a copy of `list_obj` with all empty lists and tuples removed.

    Parameters
    ----------
    list_obj : list
        A list object that requires its empty list/tuple elements to be
        removed.

    Returns
    -------
    delisted_copy : list
        Copy of `list_obj` with all empty lists/tuples removed.

    """

    # Check if list_obj is a list
    if(type(list_obj) != list):
        raise TypeError("Input argument 'list_obj' is not of type 'list'!")

    # Make a copy of itself
    delisted_copy = list(list_obj)

    # Remove all empty lists/tuples from this copy
    off_dex = len(delisted_copy)-1
    for i, element in enumerate(reversed(delisted_copy)):
        # Remove empty lists
        if(isinstance(element, list) and element == []):
            delisted_copy.pop(off_dex-i)
        # Remove empty tuples
        elif(isinstance(element, tuple) and element == ()):
            delisted_copy.pop(off_dex-i)

    # Return the copy
    return(delisted_copy)


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

    # Access PRISM metadata
    prism_dist = get_distribution('prism')

    # Add PRISM version to info_list
    info_list.append("Version: %s" % (prism_dist.version))

    # Get list of all PRISM requirements
    req_list = [req.name for req in prism_dist.requires()]

    # If imported MPI is mpi4py, add it to the list as well
    if(MPI.__package__ == 'mpi4py'):
        req_list.append('mpi4py')

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
        If *None*, no filters will be applied.

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
    else:
        child_name = ".".join([prefix, name])

    # Temporarily set default Logger class to PRISM_Logger and initialize it
    logging.setLoggerClass(PRISM_Logger)
    logger = logging.getLogger(child_name)
    logging.setLoggerClass(logging.Logger)

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
        Relative or absolute path to the directory that contains custom
        colormap files. A colormap file can be a NumPy binary file ('.npy' or
        '.npz') or any text file.

    Notes
    -----
    All colormap files in `cmap_dir` must have names starting with 'cm\\_'. The
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
    return(np.array(obj, *args, copy=False, **kwargs))


# This function raises a given error after logging the error
def raise_error(err_msg, err_type=Exception, logger=None):
    """
    Raises a given error `err_msg` of type `err_type` and logs the error using
    the provided `logger`.

    Parameters
    ----------
    err_msg : str
        The message included in the error.

    Optional
    --------
    err_type : :class:`Exception` subclass. Default: :class:`Exception`
        The type of error that needs to be raised.
    logger : :obj:`~logging.Logger` object or None. Default: None
        The logger to which the error message must be written.
        If *None*, the :obj:`~logging.RootLogger` logger is used instead.

    """

    # Log the error and raise it right after
    logger = logging.root if logger is None else logger
    logger.error(err_msg)
    raise err_type(err_msg)


# This function raises a given warning after logging the warning
def raise_warning(warn_msg, warn_type=UserWarning, logger=None, stacklevel=1):
    """
    Raises a given warning `warn_msg` of type `warn_type` and logs the warning
    using the provided `logger`.

    Parameters
    ----------
    warn_msg : str
        The message included in the warning.

    Optional
    --------
    warn_type : :class:`Warning` subclass. Default: :class:`UserWarning`
        The type of warning that needs to be raised.
    logger : :obj:`~logging.Logger` object or None. Default: None
        The logger to which the warning message must be written.
        If *None*, the :obj:`~logging.RootLogger` logger is used instead.
    stacklevel : int. Default: 1
        The stack level of the warning message at the location of this function
        call. The actual used stack level is increased by one.

    """

    # Log the warning and raise it right after
    logger = logging.root if logger is None else logger
    logger.warning(warn_msg)
    warnings.warn(warn_msg, warn_type, stacklevel=stacklevel+1)


# Redefine the print function to include the MPI rank if MPI is used
def rprint(*args, **kwargs):
    """
    Custom :func:`~print` function that prepends the rank of the MPI process
    that calls it to the message if the size of the intra-communicator is more
    than 1.
    Takes the same input arguments as the normal :func:`~print` function.

    """

    # If MPI is used and size > 1, prepend rank to message
    if(MPI.__name__ == 'mpi4py.MPI' and size > 1):
        args = list(args)
        args.insert(0, "Rank %i:" % (rank))
    print(*args, **kwargs)


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

    # Set logLevel to DEBUG
    base_logger.setLevel('DEBUG')
