# -*- coding: utf-8 -*-

"""
ModelLink
=========
Provides the definition of the :class:`~ModelLink` abstract base class.

"""


# %% IMPORTS
# Built-in imports
import abc
from inspect import _empty, currentframe, getframeinfo, isclass, signature
from inspect import _VAR_KEYWORD, _VAR_POSITIONAL
from os import path
import warnings

# Package imports
from e13tools import InputError, ShapeError
import h5py
import hickle
import numpy as np
from numpy.random import rand
from sortedcontainers import SortedDict as sdict, SortedSet as sset

# PRISM imports
from prism._docstrings import std_emul_i_doc
from prism._internal import (FeatureWarning, PRISM_Comm, RequestWarning,
                             check_instance, check_vals, convert_str_seq,
                             docstring_substitute, getCLogger, np_array,
                             raise_error)

# All declaration
__all__ = ['ModelLink', 'test_subclass']


# %% MODELLINK CLASS DEFINITION
# TODO: Allow for inter-process methods?
# Like, having a method that is called before/after construction.
class ModelLink(object, metaclass=abc.ABCMeta):
    """
    Provides an abstract base class definition that allows the
    :class:`~prism.Pipeline` class to be linked to any model/test
    object of choice. Every model wrapper used in the
    :class:`~prism.Pipeline` class must be an instance of the
    :class:`~ModelLink` class.

    Description
    -----------
    The :class:`~ModelLink` class is an abstract base class, which forms the
    base for wrapping a model and allowing *PRISM* to use it effectively.
    Because it is mandatory for every model to be wrapped in a user-made
    :class:`~ModelLink` subclass, several tools are provided to the user to
    make this as versatile as possible.

    The :class:`~ModelLink` class uses three properties that define the way the
    subclass will be used by *PRISM*: :attr:`~name`, :attr:`~call_type` and
    :attr:`~MPI_call`. The first defines what the name of the subclass is,
    which is used by *PRISM* to identify the subclass with and check if one did
    not use a different subclass by accident. The other two are flags that
    determine how the :meth:`~call_model` method should be used. These three
    properties can be set anywhere during the initialization of the
    :class:`~ModelLink` subclass, or are set to a default value if they are not
    modified.

    The model parameters and comparison data can be set in two different ways.
    They can be hard-coded into the :class:`~ModelLink` subclass by altering
    the :meth:`~get_default_model_parameters` and
    :meth:`~get_default_model_data` methods or set by providing them during
    class initialization. A combination of both is also possible. More details
    on this can be found in :meth:`~__init__`.

    The :class:`~ModelLink` class has two abstract methods that must be
    overridden before the subclass can be initialized.
    The :meth:`~call_model` method is the most important method, as it provides
    *PRISM* with a way of calling the model wrapped in the :class:`~ModelLink`
    subclass. The :meth:`~get_md_var` method allows for *PRISM* to calculate
    the model discrepancy variance.

    Note
    ----
    The :meth:`~__init__` method may be extended by the :class:`~ModelLink`
    subclass, but the superclass version must always be called.

    """

    def __init__(self, *, model_parameters=None, model_data=None):
        """
        Initialize an instance of the :class:`~ModelLink` subclass.

        Optional
        --------
        model_parameters, model_data : array_like, dict, str or None.\
            Default: None
            Anything that can be converted to a dict that provides non-default
            model parameters/data information or *None* if only default
            information is used from :meth:`~get_default_model_parameters` or
            :meth:`~get_default_model_data`. For more information on the
            lay-out of these dicts, see ``Notes``.

            If array_like, dict(`model_parameters`/`model_data`) must generate
            a dict with the correct lay-out.
            If dict, the dict itself must have the correct lay-out.
            If str, the string must be the path to a file containing the dict
            keys in the first column and the dict values in the second column,
            which combined generate a dict with the correct lay-out.

        Notes (model_parameters)
        ------------------------
        The model parameters dict requires to have the name of the parameters
        as the keyword, and a 1D list containing the lower bound, the upper
        bound and, if applicable, the estimate of this parameter. It is not
        required to provide an estimate for every parameter. The estimates are
        used to draw illustrative lines when making projection figures.
        An example of a model parameters file can be found in the 'data' folder
        of the *PRISM* package.

        Formatting :
            ``{par_name: [lower_bnd, upper_bnd, par_est]}``

        Notes (model_data)
        ------------------
        The model data dict requires to have the data identifiers
        (:attr:`~data_idx`) as the keyword, and a 1D list containing the data
        value (:attr:`~data_val`); the data errors (:attr:`~data_err`) and the
        data space (:attr:`~data_spc`).

        If the data errors are given with one value, then the data points are
        assumed to have a centered :math:`1\\sigma`-confidence interval. If the
        data errors are given with two values, then the data points are
        assumed to have a :math:`1\\sigma`-confidence interval defined by the
        provided upper and lower errors.

        The data spaces are one of five strings ({'lin', 'log' or 'log_10',
        'ln' or 'log_e'}) indicating in which of the three value spaces
        (linear, log, ln) the data values are. It defaults to 'lin' if it is
        not provided.

        The data identifier is a sequence of ints, floats and strings that is
        unique for every data point. *PRISM* uses it to identify a data point
        with, which is required in some cases (like MPI), while the model
        itself can use it as a description of the operations required to
        extract the data point from the model output. It can be provided as any
        sequence of any length for any data point. If any sequence contains a
        single element, it is replaced by just that element instead of a tuple.

        A simple example of a data identifier is :math:`f(\\text{data_idx}) =
        \\text{data_val}`, where the output of the model is given by
        :math:`f(x)`.

        An example of a model data file can be found in the 'data' folder of
        the *PRISM* package.

        Formatting :
            ``{(data_idx_0, data_idx_1, ..., data_idx_n): [data_val,`` \
            ``data_err, data_spc]}`` \n
            **or** \n
            ``{(data_idx_0, data_idx_1, ..., data_idx_n): [data_val,`` \
            ``upper_data_err, lower_data_err, data_spc]}``

        """

        # Save name of this class if not saved already
        try:
            self._name
        except AttributeError:
            self.name = self.__class__.__name__

        # Set call_type to default ('single') if not modified before
        try:
            self._call_type
        except AttributeError:
            self.call_type = 'single'

        # Set MPI_call to default (False) if not modified before
        try:
            self._MPI_call
        except AttributeError:
            self.MPI_call = False

        # Generate model parameter properties
        self.__set_model_parameters(model_parameters)

        # Generate model data properties
        self.__set_model_data(model_data)

    # Define the representation of a ModelLink object
    def __repr__(self):
        # Obtain representation of model_parameters
        par_repr = []
        for name, rng, est in zip(self._par_name, self._par_rng,
                                  self._par_est):
            if est is None:
                par_repr.append("%r: %r" % (name, rng))
            else:
                par_repr.append("%r: %r" % (name, [*rng, est]))
        par_repr = "model_parameters={%s}" % (", ".join(map(str, par_repr)))

        # Obtain representation of model_data
        data_repr = []
        data_points = []

        # Combine data points together, only adding non-default values
        for val, err, spc in zip(self._data_val, self._data_err,
                                 self._data_spc):
            # Add data value
            data_points.append([val])

            # Add data error, add only one value if error is centered
            if(err[0] == err[1]):
                data_points[-1].append(err[0])
            else:
                data_points[-1].extend(err)

            # Add data space if it is not 'lin'
            if(spc != 'lin'):
                data_points[-1].append(spc)

        # Combine data points with data identifiers
        for idx, point in zip(self._data_idx, data_points):
            data_repr.append("%r: %r" % (idx, point))
        data_repr = "model_data={%s}" % (", ".join(map(str, data_repr)))

        # Obtain non-default representation and add default ones
        str_repr = self.get_str_repr()
        str_repr.extend([par_repr, data_repr])

        # Return representation
        return("%s(%s)" % (self.__class__.__name__, ", ".join(str_repr)))

    # %% CLASS PROPERTIES
    # General
    @property
    def name(self):
        """
        str: Name associated with an instance of this :class:`~ModelLink`
        subclass.
        By default, it is set to the name of this :class:`~ModelLink` subclass.
        Can be manually manipulated to allow for more user control.

        """

        return(self._name)

    @name.setter
    def name(self, name):
        self._name = check_vals(name, 'name', 'str')

    @property
    def single_call(self):
        """
        bool: Whether :meth:`~call_model` can/should be supplied with a single
        evaluation sample. At least one of :attr:`~single_call` and
        :attr:`~multi_call` must be *True*.
        By default, single model calls are requested (True).

        """

        return(bool(self._single_call))

    @property
    def multi_call(self):
        """
        bool: Whether :meth:`~call_model` can/should be supplied with a set of
        evaluation samples. At least one of :attr:`~single_call` and
        :attr:`~multi_call` must be *True*.
        By default, single model calls are requested (False).

        """

        return(bool(self._multi_call))

    @property
    def call_type(self):
        """
        str: String indicating whether :meth:`call_model` should be supplied
        with a single evaluation sample ('single') or a set of samples
        ('multi'), or can be supplied with both ('hybrid').
        By default, single model calls are requested ('single').

        """

        return(self._call_type)

    @call_type.setter
    def call_type(self, call_type):
        # Check if call_type is a string
        call_type = check_vals(call_type, 'call_type', 'str')

        # Set single_call and multi_call appropriately
        if call_type.lower() in ('single', 'one', '1d'):
            self._single_call = 1
            self._multi_call = 0
            self._call_type = 'single'
        elif call_type.lower() in ('multi', 'many', '2d'):
            self._single_call = 0
            self._multi_call = 1
            self._call_type = 'multi'
        elif call_type.lower() in ('hybrid', 'both', 'nd'):
            self._single_call = 1
            self._multi_call = 1
            self._call_type = 'hybrid'
        else:
            raise ValueError("Input argument 'call_type' is invalid (%r)!"
                             % (call_type))

    @property
    def MPI_call(self):
        """
        bool: Whether :meth:`~call_model` can/should be called by all MPI ranks
        simultaneously instead of by the controller.
        By default, only the controller rank calls the model (False).

        """

        return(bool(self._multi_call))

    @MPI_call.setter
    def MPI_call(self, MPI_call):
        self._MPI_call = check_vals(MPI_call, 'MPI_call', 'bool')

    # Model Parameters
    @property
    def n_par(self):
        """
        int: Number of model parameters.

        """

        return(self._n_par)

    @property
    def par_name(self):
        """
        list of str: List with model parameter names.

        """

        return(self._par_name)

    @property
    def par_rng(self):
        """
        :obj:`~numpy.ndarray`: The lower and upper values of the model
        parameters.

        """

        return(self._par_rng)

    @property
    def par_est(self):
        """
        dict of {float, None}: The user-defined estimated values of the model
        parameters. Contains *None* in places where estimates were not
        provided.

        """

        return(sdict(zip(self._par_name, self._par_est)))

    # Model Data
    @property
    def n_data(self):
        """
        int: Number of provided data points.

        """

        return(self._n_data)

    @property
    def data_val(self):
        """
        list of float: The values of provided data points.

        """

        return(self._data_val)

    @property
    def data_err(self):
        """
        list of float: The upper and lower :math:`1\\sigma`-confidence levels
        of provided data points.

        """

        return(self._data_err)

    @property
    def data_spc(self):
        """
        list of str: The types of value space ({'lin', 'log', 'ln'}) of
        provided data points.

        """

        return(self._data_spc)

    @property
    def data_idx(self):
        """
        list of tuples: The user-defined data point identifiers.

        """

        return(self._data_idx)

    # %% GENERAL CLASS METHODS
    # This function returns non-default string representations of input args
    def get_str_repr(self):
        """
        Returns a list of string representations of all additional input
        arguments with which this :class:`~ModelLink` subclass was initialized.

        """

        return([])

    # This function converts values in unit space to parameter space
    def _to_par_space(self, sam_set):
        """
        Converts provided `sam_set` from unit space ([0, 1]) to parameter space
        ([lower_bnd, upper_bnd]).

        """

        return(self._par_rng[:, 0]+sam_set*(self._par_rng[:, 1] -
                                            self._par_rng[:, 0]))

    # This function converts values in parameter space to unit space
    def _to_unit_space(self, sam_set):
        """
        Converts provided `sam_set` from parameter space ([lower_bnd,
        upper_bnd]) to unit space ([0, 1]).

        """

        return((sam_set-self._par_rng[:, 0]) /
               (self._par_rng[:, 1]-self._par_rng[:, 0]))

    # This function converts a sequence of model parameter names/indices
    def _get_model_par_seq(self, par_seq, name):
        """
        Converts a provided sequence `par_seq` of model parameter names and
        indices to a list of indices, removes duplicates and checks if every
        provided name/index is valid.

        Parameters
        ----------
        par_seq : 1D array_like of {int, str}
            A sequence of integers and strings determining which model
            parameters need to be used for a certain operation.
        name : str
            A string stating the name of the variable the result of this method
            will be stored in. Used for error messages.

        Returns
        -------
        par_seq_conv : list of int
            The provided sequence `par_seq` converted to a sorted list of
            model parameter indices.

        """

        # Do some logging
        logger = getCLogger('INIT')
        logger.info("Converting sequence of model parameter names/indices.")

        # Remove all unwanted characters from the string and split it up
        par_seq = convert_str_seq(par_seq)

        # Check elements if they are ints or strings, and if they are valid
        for i, par_idx in enumerate(par_seq):
            try:
                # If par_idx is a string, try to use it as a parameter name
                if isinstance(par_idx, str):
                    par_seq[i] = self._par_name.index(par_idx)
                # If not, try to use it as a parameter index
                else:
                    self._par_name[par_idx]
                    par_seq[i] = par_idx % self._n_par
            # If any operation above fails, raise error
            except Exception as error:
                err_msg = "Input argument %r is invalid! (%s)" % (name, error)
                raise_error(err_msg, InputError, logger)

        # If everything went without exceptions, check if list is not empty and
        # remove duplicates
        if len(par_seq):
            par_seq = list(sset(par_seq))
        else:
            err_msg = "Input argument %r is empty!" % (name)
            raise_error(err_msg, ValueError, logger)

        # Log end
        logger.info("Finished converting sequence of model parameter "
                    "names/indices.")

        # Return it
        return(par_seq)

    # This function checks if a provided mod_set is valid
    def _check_mod_set(self, mod_set, name):
        """
        Checks validity of provided set of model outputs `mod_set` in this
        :obj:`~ModelLink` instance.

        Parameters
        ----------
        mod_set : 1D or 2D array_like
            Model output (set) to validate in this :obj:`~ModelLink` instance.
        name : str
            The name of the model output (set), which is used in the error
            message if the validation fails.

        Returns
        -------
        mod_set : 1D or 2D :obj:`~numpy.ndarray` object
            The provided `mod_set` if the validation was successful.

        """

        # Make logger
        logger = getCLogger('CHECK')
        logger.info("Validating provided set of model outputs %r." % (name))

        # Make sure that mod_set is a NumPy array
        mod_set = np_array(mod_set)

        # Raise error if mod_set is not 1D or 2D
        if not(mod_set.ndim == 1 or mod_set.ndim == 2):
            err_msg = ("Input argument %r is not one-dimensional or "
                       "two-dimensional!" % (name))
            raise_error(err_msg, ShapeError, logger)

        # Raise error if mod_set does not have n_data data values
        if not(mod_set.shape[-1] == self._n_data):
            err_msg = ("Input argument %r has incorrect number of data values "
                       "(%i != %i)!"
                       % (name, mod_set.shape[-1], self._n_data))
            raise_error(err_msg, ShapeError, logger)

        # Check if mod_set solely consists out of floats
        mod_set = check_vals(mod_set, name, 'float')

        # Log again and return mod_set
        logger.info("Finished validating provided set of model outputs %r."
                    % (name))
        return(mod_set)

    # This function checks if a provided sam_set is valid
    def _check_sam_set(self, sam_set, name):
        """
        Checks validity of provided set of model parameter samples `sam_set` in
        this :obj:`~ModelLink` instance.

        Parameters
        ----------
        sam_set : 1D or 2D array_like
            Parameter/sample set to validate in this :obj:`~ModelLink`
            instance.
        name : str
            The name of the parameter/sample set, which is used in the error
            message if the validation fails.

        Returns
        -------
        sam_set : 1D or 2D :obj:`~numpy.ndarray` object
            The provided `sam_set` if the validation was successful.

        """

        # Make logger
        logger = getCLogger('CHECK')
        logger.info("Validating provided set of model parameter samples %r."
                    % (name))

        # Make sure that sam_set is a NumPy array
        sam_set = np_array(sam_set)

        # Raise error if sam_set is not 1D or 2D
        if not(sam_set.ndim == 1 or sam_set.ndim == 2):
            err_msg = ("Input argument %r is not one-dimensional or "
                       "two-dimensional!" % (name))
            raise_error(err_msg, ShapeError, logger)

        # Raise error if sam_set does not have n_par parameter values
        if not(sam_set.shape[-1] == self._n_par):
            err_msg = ("Input argument %r has incorrect number of parameters "
                       "(%i != %i)!"
                       % (name, sam_set.shape[-1], self._n_par))
            raise_error(err_msg, ShapeError, logger)

        # Check if sam_set solely consists out of floats
        sam_set = check_vals(sam_set, name, 'float')

        # Check if all samples are within parameter space
        for i, par_set in enumerate(self._to_unit_space(sam_set)):
            # If not, raise an error
            if not ((par_set >= 0)*(par_set <= 1)).all():
                err_msg = ("Input argument %r contains a sample outside of "
                           "parameter space at index %i!" % (name, i))
                raise_error(err_msg, ValueError, logger)

        # Log again and return sam_set
        logger.info("Finished validating provided set of model parameter "
                    "samples %r." % (name))
        return(sam_set)

    # This function returns the path to a backup file
    # TODO: Should backup file be saved in emulator working directory of PRISM?
    def _get_backup_path(self, emul_i):
        """
        Returns the absolute path to a backup file made by this
        :obj:`~ModelLink` instance, using the provided `emul_i`.

        """

        # Determine the name of the backup hdf5-file
        filename = "backup_%i_%s.hdf5" % (emul_i, self._name)

        # Determine the path of the backup hdf5-file
        filepath = path.join(path.abspath('.'), filename)

        # Return it
        return(filepath)

    # This function makes a backup of args/kwargs to be used during call_model
    def _make_backup(self, *args, **kwargs):
        """
        WARNING: This is an advanced utility method and probably will not work
        unless used properly. Use with caution!

        Creates an HDF5-file backup of the provided `args` and `kwargs` when
        called by the :meth:`~call_model` method or any of its inner functions.
        Additionally, the backup will contain the `emul_i`, `par_set` and
        `data_idx` values that were passed to the :meth:`~call_model` method.
        The backup can be restored using the :meth:`~_read_backup` method.

        If it is detected that this method is used incorrectly, a
        :class:`~prism._internal.RequestWarning` is raised (and the method
        returns) rather than a :class:`~prism._internal.RequestError`, in order
        to not disrupt the call to :meth:`~call_model`.

        Parameters
        ----------
        args : tuple
            All positional arguments that must be stored in the backup file.
        kwargs : dict
            All keyword arguments that must be stored in the backup file.

        Notes
        -----
        If an HDF5-file already exists with the same name as the to be created
        backup, it will be replaced. However, *PRISM* itself will never remove
        any backup files (e.g., reconstructing an iteration).

        The saved `emul_i`, `par_set` and `data_idx` are the values these
        variables have locally in the :meth:`~call_model` method at the point
        this method is called. Because of this, making any changes to them may
        cause problems and is therefore heavily discouraged. If changes are
        necessary, it is advised to assign them to a different variable first.

        """

        # Raise warning about this feature being experimental
        warn_msg = ("The 'call_model' backup system is still experimental and "
                    "it may see significant changes or be (re)moved in the "
                    "future!")
        warnings.warn(warn_msg, FeatureWarning, stacklevel=2)

        # Check if any args or kwargs have been provided
        if not args and not kwargs:
            # If not, issue a warning about that and return
            warn_msg = ("No positional or keyword arguments have been "
                        "provided. Backup creation will be skipped!")
            warnings.warn(warn_msg, RequestWarning, stacklevel=2)
            return

        # Initialize the caller's frame with the current frame
        caller_frame = currentframe()

        # Obtain the call_model frame
        while(caller_frame is not None and
              getframeinfo(caller_frame)[2] != 'call_model'):
            caller_frame = caller_frame.f_back

        # If caller_frame is None, the call_model frame was not found
        if caller_frame is None:
            # Issue a warning about it and return
            warn_msg = ("This method has been called from outside the "
                        "'call_model' method. Backup creation will be "
                        "skipped!")
            warnings.warn(warn_msg, RequestWarning, stacklevel=2)
            return

        # Obtain the locals of the call_model frame
        loc = caller_frame.f_locals

        # Extract local emul_i, par_set and data_idx
        # Unless call_model was called using args, below will extract correctly
        # These one-liners are the equivalent of
        # try:
        #     emul_i = loc['emul_i']
        # except KeyError:
        #     try:
        #         emul_i = loc['kwargs']['emul_i']
        #     except KeyError:
        #         emul_i = None
        emul_i = loc.get('emul_i', loc.get('kwargs', {}).get('emul_i'))
        par_set = loc.get('par_set', loc.get('kwargs', {}).get('par_set'))
        data_idx = loc.get('data_idx', loc.get('kwargs', {}).get('data_idx'))

        # If one of these is None, then it is not correctly locally available
        # This can happen if args are used instead of kwargs for call_model
        # PRISM code always uses kwargs and never causes this problem
        if None in (emul_i, par_set, data_idx):
            warn_msg = ("Required local variables 'emul_i', 'par_set' and "
                        "'data_idx' are not correctly available. Backup "
                        "creation will be skipped!")
            warnings.warn(warn_msg, RequestWarning, stacklevel=2)
            return

        # Obtain path to backup file
        filepath = self._get_backup_path(emul_i)

        # Save emul_i, par_set, data_idx, args and kwargs to hdf5
        with h5py.File(filepath, 'w') as file:
            hickle.dump(emul_i, file, path='/emul_i')
            hickle.dump(dict(par_set), file, path='/par_set')
            hickle.dump(data_idx, file, path='/data_idx')
            hickle.dump(args, file, path='/args')
            hickle.dump(kwargs, file, path='/kwargs')

    # This function reads in a backup made by _make_backup
    # TODO: Allow for absolute path to backup file to be given?
    def _read_backup(self, emul_i):
        """
        Reads in a backup HDF5-file created by the :meth:`~_make_backup`
        method, using the provided `emul_i` and the value of :attr:`~name`.

        Parameters
        ----------
        emul_i : int
            The emulator iteration that was provided to the :meth:`~call_model`
            method when the backup was made.

        Returns
        -------
        data : dict with keys `('emul_i', 'par_set', 'data_idx', 'args',` \
            `'kwargs')`
            A dict containing the data that was provided to the
            :meth:`~_make_backup` method.

        """

        # Raise warning about this feature being experimental
        warn_msg = ("The 'call_model' backup system is still experimental and "
                    "it may see significant changes or be (re)moved in the "
                    "future!")
        warnings.warn(warn_msg, FeatureWarning, stacklevel=2)

        # Check if provided emul_i is an integer
        emul_i = check_vals(emul_i, 'emul_i', 'int', 'nneg')

        # Obtain name of backup file
        filepath = self._get_backup_path(emul_i)

        # Check if filepath exists
        if not path.exists(filepath):
            err_msg = ("Input argument 'emul_i' does not yield an existing "
                       "path to a backup file (%r)!" % (filepath))
            raise OSError(err_msg)

        # Initialize empty data dict
        data = sdict()

        # Read emul_i, par_set, data_idx, args and kwargs from hdf5
        with h5py.File(filepath, 'r') as file:
            data['emul_i'] = hickle.load(file, path='/emul_i')
            data['par_set'] = sdict(hickle.load(file, path='/par_set'))
            data['data_idx'] = hickle.load(file, path='/data_idx')
            data['args'] = hickle.load(file, path='/args')
            data['kwargs'] = hickle.load(file, path='/kwargs')

        # Return data
        return(data)

    @property
    def _default_model_parameters(self):
        """
        dict: The default model parameters to use for every instance of this
        :class:`~ModelLink` subclass.

        """

        return(sdict())

    def get_default_model_parameters(self):
        """
        Returns the default model parameters to use for every instance of this
        :class:`~ModelLink` subclass. By default, returns
        :attr:`~ModelLink._default_model_parameters`.

        """

        return(self._default_model_parameters)

    def __set_model_parameters(self, add_model_parameters):
        """
        Generates the model parameter properties from the default model
        parameters and the additional input argument `add_model_parameters`.

        Parameters
        ----------
        add_model_parameters : array_like, dict, str or None
            Anything that can be converted to a dict that provides non-default
            model parameters information or *None* if only default information
            is used from :meth:`~ModelLink.get_default_model_parameters`.

        Generates
        ---------
        n_par : int
            Number of model parameters.
        par_name : list
            List with model parameter names.
        par_rng : :obj:`~numpy.ndarray` object
            Array containing the lower and upper values of the model
            parameters.
        par_est : list
            List containing user-defined estimated values of the model
            parameters.
            Contains *None* in places where estimates were not provided.

        """

        # Obtain default model parameters
        model_parameters = sdict(self.get_default_model_parameters())

        # If no additional model parameters information is given
        if add_model_parameters is None:
            pass

        # If a parameter file is given
        elif isinstance(add_model_parameters, str):
            # Obtain absolute path to given file
            par_file = path.abspath(add_model_parameters)

            # Read the parameter file in as a string
            pars = np.genfromtxt(par_file, dtype=(str), delimiter=':',
                                 autostrip=True)

            # Make sure that pars is 2D
            pars = np_array(pars, ndmin=2)

            # Combine default parameters with read-in parameters
            model_parameters.update(pars)

        # If a parameter dict is given
        elif isinstance(add_model_parameters, dict):
            model_parameters.update(add_model_parameters)

        # If anything else is given
        else:
            # Check if it can be converted to a dict
            try:
                par_dict = sdict(add_model_parameters)
            except Exception:
                raise TypeError("Input model parameters cannot be converted to"
                                " type 'dict'!")
            else:
                model_parameters.update(par_dict)

        # Save number of model parameters
        n_par = len(model_parameters.keys())
        if(n_par == 1):
            raise InputError("Number of model parameters must be at least 2!")
        else:
            self._n_par = check_vals(n_par, 'n_par', 'pos')

        # Create empty parameter name, ranges and estimate lists/arrays
        self._par_name = []
        self._par_rng = np.zeros([self._n_par, 2])
        self._par_rng[:, 1] = 1
        self._par_est = []

        # Save model parameters as class properties
        for i, (name, values_str) in enumerate(model_parameters.items()):
            # Convert values_str to values
            values = convert_str_seq(values_str)

            # Save parameter name and range
            self._par_name.append(check_vals(name, 'par_name', 'str'))
            self._par_rng[i] = check_vals(values[:2], 'par_rng[%s]' % (name),
                                          'float')

            # Check if a float parameter estimate was provided
            try:
                est = check_vals(values[2], 'par_est[%s]' % (name), 'float')
            # If no estimate was provided, save it as None
            except IndexError:
                self._par_est.append(None)
            # If no float was provided, check if it was None
            except TypeError as error:
                # If it is None, save it as such
                if(values[2].lower() == 'none'):
                    self._par_est.append(None)
                # If it is not None, reraise the previous error
                else:
                    raise error
            # If a float was provided, check if it is within parameter range
            else:
                if(values[0] <= est <= values[1]):
                    self._par_est.append(est)
                else:
                    raise ValueError("Input argument 'par_est[%s]' is outside "
                                     "of defined parameter range!" % (name))

    @property
    def _default_model_data(self):
        """
        dict: The default model data to use for every instance of this
        :class:`~ModelLink` subclass.

        """

        return(dict())

    def get_default_model_data(self):
        """
        Returns the default model data to use for every instance of this
        :class:`~ModelLink` subclass. By default, returns
        :attr:`~ModelLink._default_model_data`.

        """

        return(self._default_model_data)

    def __set_model_data(self, add_model_data):
        """
        Generates the model data properties from the default model data and the
        additional input argument `add_model_data`.

        Parameters
        ---------
        add_model_data : array_like, dict, str or None
            Anything that can be converted to a dict that provides non-default
            model data information or *None* if only default data is used from
            :meth:`~ModelLink.get_default_model_data`.

        Generates
        ---------
        n_data : int
            Number of provided data points.
        data_val : list
            List with values of provided data points.
        data_err : list of lists
            List with upper and lower :math:`1\\sigma`-confidence levels of
            provided data points.
        data_spc : list
            List with types of value space ({'lin', 'log', 'ln'}) of provided
            data points.
        data_idx : list of tuples
            List with user-defined data point identifiers.

        """

        # Obtain default model data
        model_data = dict(self.get_default_model_data())

        # If no additional model data information is given
        if add_model_data is None:
            pass

        # If a data file is given
        elif isinstance(add_model_data, str):
            # Obtain absolute path to given file
            data_file = path.abspath(add_model_data)

            # Read the data file in as a string
            data_points = np.genfromtxt(data_file, dtype=(str),
                                        delimiter=':', autostrip=True)

            # Make sure that data_points is 2D
            data_points = np_array(data_points, ndmin=2)

            # Combine default data with read-in data
            model_data.update(data_points)

        # If a data dict is given
        elif isinstance(add_model_data, dict):
            model_data.update(add_model_data)

        # If anything else is given
        else:
            # Check if it can be converted to a dict
            try:
                data_dict = dict(add_model_data)
            except Exception:
                raise TypeError("Input model data cannot be converted to type "
                                "'dict'!")
            else:
                model_data.update(data_dict)

        # Make an empty model_data dict
        model_data_dict = dict()

        # Loop over all data points in model_data and process data identifiers
        for key, value in model_data.items():
            # Convert key to an actual data_idx
            tmp_idx = convert_str_seq(key)

            # Check if tmp_idx is not empty
            if not len(tmp_idx):
                raise InputError("Model data contains a data point with no "
                                 "identifier!")

            # Convert value to an actual data point
            tmp_point = convert_str_seq(value)

            # Save data_idx with corresponding point to model_data_dict
            model_data_dict[tuple(tmp_idx)] = tmp_point

        # Determine the number of data points
        self._n_data = check_vals(len(model_data_dict), 'n_data', 'pos')

        # Create empty data value, error, space and identifier lists
        self._data_val = []
        self._data_err = []
        self._data_spc = []
        self._data_idx = []

        # Save model data as class properties
        for idx, data in model_data_dict.items():
            # Convert idx to list for error messages
            idx = list(idx)

            # Save data value
            self._data_val.append(check_vals(data[0], 'data_val%s' % (idx),
                                             'float'))

            # Save data error and extract space
            # If length is two, centered error and no data space were given
            if(len(data) == 2):
                self._data_err.append(
                    [check_vals(data[1], 'data_err%s' % (idx), 'float',
                                'pos')]*2)
                spc = 'lin'

            # If length is three, there are two possibilities
            elif(len(data) == 3):
                # If the third column contains a string, it is the data space
                if isinstance(data[2], str):
                    self._data_err.append(
                        [check_vals(data[1], 'data_err%s' % (idx), 'float',
                                    'pos')]*2)
                    spc = data[2]

                # If the third column contains no string, it is error interval
                else:
                    self._data_err.append(
                        check_vals(data[1:3], 'data_err%s' % (idx), 'float',
                                   'pos'))
                    spc = 'lin'

            # If length is four+, error interval and data space were given
            else:
                self._data_err.append(
                    check_vals(data[1:3], 'data_err%s' % (idx), 'float',
                               'pos'))
                spc = data[3]

            # Save data space
            # Check if valid data space has been provided
            spc = str(spc).replace("'", '').replace('"', '')
            if spc.lower() in ('lin', 'linear'):
                self._data_spc.append('lin')
            elif spc.lower() in ('log', 'log10', 'log_10'):
                self._data_spc.append('log10')
            elif spc.lower() in ('ln', 'loge', 'log_e'):
                self._data_spc.append('ln')
            else:
                raise ValueError("Input argument 'data_spc%s' is invalid (%r)!"
                                 % (idx, spc))

            # Save data identifier as tuple or single element
            if(len(idx) == 1):
                self._data_idx.append(idx[0])
            else:
                self._data_idx.append(tuple(idx))

    # %% ABSTRACT USER METHODS
    @abc.abstractmethod
    @docstring_substitute(emul_i=std_emul_i_doc)
    def call_model(self, emul_i, par_set, data_idx):
        """
        Calls the model wrapped in this :class:`~ModelLink` subclass at
        emulator iteration `emul_i` for model parameter values `par_set` and
        returns the data points corresponding to `data_idx`.

        This method is called with solely keyword arguments.

        This is an abstract method and must be overridden by the
        :class:`~ModelLink` subclass.

        Parameters
        ----------
        %(emul_i)s
        par_set : dict of :class:`~numpy.float64`
            Dict containing the values for all model parameters corresponding
            to the requested model realization(s). If model is single-called,
            dict is formatted as ``{par_name: par_val}``. If multi-called, it
            is formatted as ``{par_name: [par_val_1, par_val_2, ...,
            par_val_n]}``.
        data_idx : list of tuples
            List containing the user-defined data point identifiers
            corresponding to the requested data points.

        Returns
        -------
        data_val : 1D or 2D array_like
            Array containing the data values corresponding to the requested
            data points generated by the requested model realization(s). If
            model is multi-called, `data_val` is of shape ``(n_sam, n_data)``.

        """

        # Raise NotImplementedError if only super() was called
        raise NotImplementedError("This method must be user-written in the "
                                  "ModelLink subclass!")

    @abc.abstractmethod
    @docstring_substitute(emul_i=std_emul_i_doc)
    def get_md_var(self, emul_i, par_set, data_idx):
        """
        Calculates the linear model discrepancy variance at a given emulator
        iteration `emul_i` for model parameter values `par_set` and given data
        points `data_idx` for the model wrapped in this :class:`~ModelLink`
        subclass.

        This method is always single-called by one MPI rank with solely keyword
        arguments.

        This is an abstract method and must be overridden by the
        :class:`~ModelLink` subclass.

        Parameters
        ----------
        %(emul_i)s
        par_set : dict of :class:`~numpy.float64`
            Dict containing the values for all model parameters corresponding
            to the requested model realization.
        data_idx : list of tuples
            List containing the user-defined data point identifiers
            corresponding to the requested data points.

        Returns
        -------
        md_var : 1D or 2D array_like
            Array containing the linear model discrepancy variance values
            corresponding to the requested data points. If 1D array_like, data
            is assumed to have a centered one sigma confidence interval. If 2D
            array_like, the values determine the upper and lower variances and
            the array is of shape ``(n_data, 2)``.

        Notes
        -----
        The returned model discrepancy variance values must be of linear form,
        even for those data values that are returned in logarithmic form by the
        :meth:`~call_model` method. If not, the possibility exists that the
        emulation process will not converge properly.

        """

        # Raise NotImplementedError if only super() was called
        raise NotImplementedError("This method must be user-written in the "
                                  "ModelLink subclass!")


# %% UTILITY FUNCTIONS
# This function tests a given ModelLink subclass
# TODO: Are there any more tests that can be done here?
def test_subclass(subclass, *args, **kwargs):
    """
    Tests a provided :class:`~ModelLink` `subclass` by initializing it with the
    given `args` and `kwargs` and checking if all required methods can be
    properly called.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    subclass : :class:`~ModelLink` subclass
        The :class:`~ModelLink` subclass that requires testing.
    args : tuple
        Positional arguments that need to be provided to the constructor of the
        `subclass`.
    kwargs : dict
        Keyword arguments that need to be provided to the constructor of the
        `subclass`.

    Returns
    -------
    modellink_obj : :obj:`~ModelLink` object
        Instance of the provided `subclass` if all tests pass successfully.
        Specific exceptions are raised if a test fails.

    Note
    ----
    Depending on the complexity of the model wrapped in the given `subclass`,
    this function may take a while to execute.

    """

    # Check if provided subclass is a class
    if not isclass(subclass):
        raise InputError("Input argument 'subclass' must be a class!")

    # Check if provided subclass is a subclass of ModelLink
    if not issubclass(subclass, ModelLink):
        raise TypeError("Input argument 'subclass' must be a subclass of the "
                        "ModelLink class!")

    # Try to initialize provided subclass
    try:
        modellink_obj = subclass(*args, **kwargs)
    except Exception as error:
        raise InputError("Input argument 'subclass' cannot be initialized! "
                         "(%s)" % (error))

    # Check if modellink_obj was initialized properly
    if not check_instance(modellink_obj, ModelLink):
        obj_name = modellink_obj.__class__.__name__
        raise InputError("Provided ModelLink subclass %r was not "
                         "initialized properly! Make sure that %r calls "
                         "the super constructor during initialization!"
                         % (obj_name, obj_name))

    # Obtain list of arguments call_model should take
    call_model_args = list(signature(ModelLink.call_model).parameters)
    call_model_args.remove('self')

    # Check if call_model takes the correct arguments
    obj_call_model_args = dict(signature(modellink_obj.call_model).parameters)
    for arg in call_model_args:
        if arg not in obj_call_model_args.keys():
            raise InputError("The 'call_model()'-method in provided ModelLink "
                             "subclass %r does not take required input "
                             "argument %r!" % (modellink_obj._name, arg))
        else:
            obj_call_model_args.pop(arg)

    # Check if call_model takes any other arguments
    for arg, par in obj_call_model_args.items():
        # If this parameter has no default value and is not *args or **kwargs
        if(par.default == _empty and par.kind != _VAR_POSITIONAL and
           par.kind != _VAR_KEYWORD):
            # Raise error
            raise InputError("The 'call_model()'-method in provided ModelLink "
                             "subclass %r takes an unknown non-optional input "
                             "argument %r!" % (modellink_obj._name, arg))

    # Obtain list of arguments get_md_var should take
    get_md_var_args = list(signature(ModelLink.get_md_var).parameters)
    get_md_var_args.remove('self')

    # Check if get_md_var takes the correct arguments
    obj_get_md_var_args = dict(signature(modellink_obj.get_md_var).parameters)
    for arg in get_md_var_args:
        if arg not in obj_get_md_var_args.keys():
            raise InputError("The 'get_md_var()'-method in provided ModelLink "
                             "subclass %r does not take required input "
                             "argument %r!" % (modellink_obj._name, arg))
        else:
            obj_get_md_var_args.pop(arg)

    # Check if get_md_var takes any other arguments
    for arg, par in obj_get_md_var_args.items():
        # If this parameter has no default value and is not *args or **kwargs
        if(par.default == _empty and par.kind != _VAR_POSITIONAL and
           par.kind != _VAR_KEYWORD):
            # Raise an error
            raise InputError("The 'get_md_var()'-method in provided ModelLink "
                             "subclass %r takes an unknown non-optional input "
                             "argument %r!" % (modellink_obj._name, arg))

    # Set MPI intra-communicator
    comm = PRISM_Comm()

    # Obtain random sam_set on controller
    if not comm._rank:
        sam_set = modellink_obj._to_par_space(rand(1, modellink_obj._n_par))
    # Workers get dummy sam_set
    else:
        sam_set = []

    # Broadcast random sam_set to workers
    sam_set = comm.bcast(sam_set, 0)

    # Try to evaluate sam_set in the model
    try:
        # Check who needs to call the model
        if not comm._rank or modellink_obj._MPI_call:
            # Do multi-call
            if modellink_obj._multi_call:
                mod_set = modellink_obj.call_model(
                    emul_i=0,
                    par_set=sdict(zip(modellink_obj._par_name, sam_set.T)),
                    data_idx=modellink_obj._data_idx)

            # Single-call
            else:
                # Initialize mod_set
                mod_set = np.zeros([sam_set.shape[0], modellink_obj._n_data])

                # Loop over all samples in sam_set
                for i, par_set in enumerate(sam_set):
                    mod_set[i] = modellink_obj.call_model(
                        emul_i=0,
                        par_set=sdict(zip(modellink_obj._par_name, par_set)),
                        data_idx=modellink_obj._data_idx)

    # If call_model was not overridden, catch NotImplementedError
    except NotImplementedError:
        raise NotImplementedError("Provided ModelLink subclass %r has no "
                                  "user-written 'call_model()'-method!"
                                  % (modellink_obj._name))

    # If successful, check if obtained mod_set has correct shape
    if not comm._rank:
        mod_set = modellink_obj._check_mod_set(mod_set, 'mod_set')

    # Check if the model discrepancy variance can be obtained
    try:
        md_var = modellink_obj.get_md_var(
            emul_i=0,
            par_set=sdict(zip(modellink_obj._par_name, sam_set[0])),
            data_idx=modellink_obj._data_idx)

    # If get_md_var was not overridden, catch NotImplementedError
    except NotImplementedError:
        warn_msg = ("Provided ModelLink subclass %r has no user-written "
                    "get_md_var()-method! Default model discrepancy variance "
                    "description would be used instead!"
                    % (modellink_obj._name))
        warnings.warn(warn_msg, RequestWarning, stacklevel=2)

    # If successful, check if obtained md_var has correct shape
    else:
        md_var = modellink_obj._check_mod_set(md_var, 'md_var')

    # Return modellink_obj
    return(modellink_obj)
