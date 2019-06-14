# -*- coding: utf-8 -*-

"""
ModelLink
=========
Provides the definition of the :class:`~ModelLink` abstract base class.

"""


# %% IMPORTS
# Built-in imports
import abc
import os
from os import path
from tempfile import mktemp
import warnings

# Package imports
from e13tools import InputError, ShapeError
from e13tools.utils import (convert_str_seq, docstring_substitute,
                            get_outer_frame, raise_error)
import h5py
import hickle
import numpy as np
from sortedcontainers import SortedDict as sdict, SortedSet as sset

# PRISM imports
from prism import __version__
from prism._docstrings import std_emul_i_doc
from prism._internal import (FeatureWarning, RequestWarning, check_vals,
                             getCLogger, np_array)
from prism.modellink.utils import convert_data, convert_parameters

# All declaration
__all__ = ['ModelLink']


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

    Notes
    -----
    The :meth:`~__init__` method may be extended by the :class:`~ModelLink`
    subclass, but the superclass version must always be called.

    If required, one can use the :func:`~prism.modellink.test_subclass`
    function to test a :class:`~ModelLink` subclass on correct functionality.

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
        of the *PRISM* package. If required, one can use the
        :func:`~prism.modellink.convert_parameters` function to validate their
        formatting.

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
        the *PRISM* package. If required, one can use the
        :func:`~prism.modellink.convert_data` function to validate their
        formatting.

        Formatting :
            ``{(data_idx_0, data_idx_1, ..., data_idx_n): [data_val,`` \
            ``data_err, data_spc]}`` \n
            **or** \n
            ``{(data_idx_0, data_idx_1, ..., data_idx_n): [data_val,`` \
            ``upper_data_err, lower_data_err, data_spc]}``

        """

        # Save name of this class if not saved already
        if not hasattr(self, '_name'):
            self.name = self.__class__.__name__

        # Set call_type to default ('single') if not modified before
        if not hasattr(self, '_call_type'):
            self.call_type = 'single'

        # Set MPI_call to default (False) if not modified before
        if not hasattr(self, '_MPI_call'):
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
        # If name is set outside of __init__, save current value
        outer_frame = get_outer_frame(self.__init__)
        if outer_frame is None and not hasattr(self, '_init_name'):
            self._init_name = str(self._name)

        # Save new name
        self._name = check_vals(name, 'name', 'str')

        # If name is set outside of __init__, raise warning
        if outer_frame is None and (self._name != self._init_name):
            warn_msg = ("The 'name' property of this %s instance is being set "
                        "outside its constructor. This may have unexpected "
                        "effects. It is advised to set it back to its original"
                        " value (%r)!"
                        % (self.__class__.__name__, self._init_name))
            warnings.warn(warn_msg, RequestWarning, stacklevel=2)

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
        # If call_type is set outside of __init__, save current value
        outer_frame = get_outer_frame(self.__init__)
        if outer_frame is None and not hasattr(self, '_init_call_type'):
            self._init_call_type = str(self._call_type)

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

        # If call_type is set outside of __init__, raise warning
        if outer_frame is None and (self._call_type != self._init_call_type):
            warn_msg = ("The 'call_type' property of this %s instance is being"
                        " set outside its constructor. This may have "
                        "unexpected effects. It is advised to set it back to "
                        "its original value (%r)!"
                        % (self.__class__.__name__, self._init_call_type))
            warnings.warn(warn_msg, RequestWarning, stacklevel=2)

    @property
    def MPI_call(self):
        """
        bool: Whether :meth:`~call_model` can/should be called by all MPI ranks
        simultaneously instead of by the controller.
        By default, only the controller rank calls the model (False).

        """

        return(bool(self._MPI_call))

    @MPI_call.setter
    def MPI_call(self, MPI_call):
        # If MPI_call is set outside of __init__, save current value
        outer_frame = get_outer_frame(self.__init__)
        if outer_frame is None and not hasattr(self, '_init_MPI_call'):
            self._init_MPI_call = bool(self._MPI_call)

        # Save new MPI_call
        self._MPI_call = check_vals(MPI_call, 'MPI_call', 'bool')

        # If MPI_call is set outside of __init__, raise warning
        if outer_frame is None and (self._MPI_call != self._init_MPI_call):
            warn_msg = ("The 'MPI_call' property of this %s instance is being "
                        "set outside its constructor. This may have unexpected"
                        " effects. It is advised to set it back to its "
                        "original value (%r)!"
                        % (self.__class__.__name__, self._init_MPI_call))
            warnings.warn(warn_msg, RequestWarning, stacklevel=2)

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
                err_msg = ("Input argument %r[%i] is invalid! (%s)"
                           % (name, i, error))
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
        mod_set : 1D or 2D array_like or dict
            Model output (set) to validate in this :obj:`~ModelLink` instance.
        name : str
            The name of the model output (set), which is used in the error
            message if the validation fails.

        Returns
        -------
        mod_set : 1D or 2D :obj:`~numpy.ndarray` object
            The provided `mod_set` if the validation was successful. If
            `mod_set` was a dict, it will be converted to a
            :obj:`~numpy.ndarray` object.

        """

        # Make logger
        logger = getCLogger('CHECK')
        logger.info("Validating provided set of model outputs %r." % (name))

        # If mod_set is a dict, convert it to a NumPy array
        if isinstance(mod_set, dict):
            mod_set = np_array([mod_set[idx] for idx in mod_set.keys()]).T

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
        sam_set : 1D or 2D array_like or dict
            Parameter/sample set to validate in this :obj:`~ModelLink`
            instance.
        name : str
            The name of the parameter/sample set, which is used in the error
            message if the validation fails.

        Returns
        -------
        sam_set : 1D or 2D :obj:`~numpy.ndarray` object
            The provided `sam_set` if the validation was successful. If
            `sam_set` was a dict, it will be converted to a
            :obj:`~numpy.ndarray` object.

        """

        # Make logger
        logger = getCLogger('CHECK')
        logger.info("Validating provided set of model parameter samples %r."
                    % (name))

        # If sam_set is a dict, convert it to a NumPy array
        if isinstance(sam_set, dict):
            sam_set = np_array([sam_set[par] for par in self._par_name]).T

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
        unit_sams = self._to_unit_space(np_array(sam_set, ndmin=2))
        for i, par_set in enumerate(unit_sams):
            # If not, raise an error
            if not ((par_set >= 0)*(par_set <= 1)).all():
                err_msg = ("Input argument %r contains a sample outside of "
                           "parameter space at index %i!" % (name, i))
                raise_error(err_msg, ValueError, logger)

        # Log again and return sam_set
        logger.info("Finished validating provided set of model parameter "
                    "samples %r." % (name))
        return(sam_set)

    # This function checks if a provided md_var is valid
    def _check_md_var(self, md_var, name):
        """
        Checks validity of provided set of model discrepancy variances `md_var`
        in this :obj:`~ModelLink` instance.

        Parameters
        ----------
        md_var : 1D or 2D array_like or dict
            Model discrepancy variance set to validate in this
            :obj:`~ModelLink` instance.
        name : str
            The name of the model discrepancy set, which is used in the error
            message if the validation fails.

        Returns
        -------
        md_var : 2D :obj:`~numpy.ndarray` object
            The (converted) provided `md_var` if the validation was successful.
            If `md_var` was a dict, it will be converted to a
            :obj:`~numpy.ndarray` object.

        """

        # Make logger
        logger = getCLogger('CHECK')
        logger.info("Validating provided set of model discrepancy variances "
                    "%r." % (name))

        # If md_var is a dict, convert it to a NumPy array
        if isinstance(md_var, dict):
            md_var = np_array([md_var[idx] for idx in md_var.keys()])

        # Make sure that md_var is a NumPy array
        md_var = np_array(md_var)

        # Raise error if md_var is not 1D or 2D
        if not(md_var.ndim == 1 or md_var.ndim == 2):
            err_msg = ("Input argument %r is not one-dimensional or "
                       "two-dimensional!" % (name))
            raise_error(err_msg, ShapeError, logger)

        # Check if md_var contains n_data values
        if not(md_var.shape[0] == self._n_data):
            err_msg = ("Received array of model discrepancy variances %r has "
                       "incorrect number of data points (%i != %i)!"
                       % (name, md_var.shape[0], self._n_data))
            raise ShapeError(err_msg)

        # Check if single or dual values were given
        if(md_var.ndim == 1):
            md_var = np_array([md_var]*2).T
        elif(md_var.shape[1] == 2):
            pass
        else:
            err_msg = ("Received array of model discrepancy variances %r has "
                       "incorrect number of values (%i != 2)!"
                       % (name, md_var.shape[1]))
            raise ShapeError(err_msg)

        # Check if all values are non-negative floats
        md_var = check_vals(md_var, 'md_var', 'nneg', 'float')

        # Log again and return md_var
        logger.info("Finished validating provided set of model discrepancy "
                    "variances %r." % (name))
        return(md_var)

    # This function returns the path to a backup file
    # TODO: Should backup file be saved in emulator working directory of PRISM?
    def _get_backup_path(self, emul_i, suffix):
        """
        Returns the absolute path to a backup file made by this
        :obj:`~ModelLink` instance, using the provided `emul_i` and `suffix`.

        This method is used by the :meth:`~_make_backup` and
        :meth:`~_read_backup` methods, and should not be called directly.

        Parameters
        ----------
        emul_i : int
            The emulator iteration for which a backup filepath is needed.
        suffix : str or None
            If str, determine path to associated backup file using provided
            `suffix`. If `suffix` is empty, obtain last created backup file.
            If *None*, create a new path to a backup file.

        Returns
        -------
        filepath : str
            Absolute path to requested backup file.

        """

        # Determine the prefix of the backup hdf5-file
        prefix = "backup_%i_%s(" % (emul_i, self._name)

        # If suffix is None, generate new backup filepath
        if suffix is None:
            # Determine the path of the backup hdf5-file
            filepath = path.abspath(mktemp(').hdf5', prefix, '.'))

            # Return determined filepath
            return(filepath)

        # If suffix is a string, determine the path
        elif isinstance(suffix, str):
            # If the string is empty, find the last created backup file
            if(suffix == ''):
                # Make list of all files in current directory
                filenames = next(os.walk('.'))[2]

                # Make empty list of all backup files
                backup_files = []

                # Loop over all filenames
                for filename in filenames:
                    # If the filename has the correct prefix
                    if filename.startswith(prefix):
                        # Obtain full path to the file
                        filepath = path.abspath(path.join('.', filename))

                        # Obtain creation time and append to backup_files
                        ctime = path.getctime(filepath)
                        backup_files.append([filepath, ctime])

                # Sort backup_files list on creation time
                backup_files.sort(key=lambda x: x[1], reverse=True)

                # If backup_files is not empty, return last one created
                if len(backup_files):
                    return(backup_files[0][0])
                # Else, raise error
                else:
                    err_msg = ("No backup files can be found in the current "
                               "directory for input argument 'emul_i'!")
                    raise OSError(err_msg)

            # If the string is not empty, check if provided suffix is valid
            else:
                # Obtain full filepath
                filepath = path.abspath(path.join(
                        '.', ''.join([prefix, suffix, ').hdf5'])))

                # If filepath exists, return it
                if path.exists(filepath):
                    return(filepath)
                # If not, raise error
                else:
                    err_msg = ("Input argument 'suffix' does not yield an "
                               "existing path to a backup file (%r)!"
                               % (filepath))
                    raise OSError(err_msg)

    # This function makes a backup of args/kwargs to be used during call_model
    def _make_backup(self, *args, **kwargs):
        """
        WARNING: This is an advanced utility method and probably will not work
        unless used properly. Use with caution!

        Creates an HDF5-file backup of the provided `args` and `kwargs` when
        called by the :meth:`~call_model` method or any of its inner functions.
        Additionally, the backup will contain the `emul_i`, `par_set` and
        `data_idx` values that were passed to the :meth:`~call_model` method.
        It also contains the version of *PRISM* that made the backup.
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
        The name of the created backup file contains the value of `emul_i`,
        :attr:`~name` and a random string to avoid replacing an already
        existing backup file.

        The saved `emul_i`, `par_set` and `data_idx` are the values these
        variables have locally in the :meth:`~call_model` method at the point
        this method is called. Because of this, making any changes to them may
        cause problems and is therefore heavily discouraged. If changes are
        necessary, it is advised to copy them to a different variable first.

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

        # Obtain the call_model frame
        caller_frame = get_outer_frame(self.call_model)

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
        filepath = self._get_backup_path(emul_i, None)

        # Save emul_i, par_set, data_idx, args and kwargs to hdf5
        with h5py.File(filepath, 'w') as file:
            file.attrs['emul_i'] = emul_i
            file.attrs['prism_version'] = __version__
            hickle.dump(dict(par_set), file, path='/par_set')
            hickle.dump(data_idx, file, path='/data_idx')
            hickle.dump(args, file, path='/args')
            hickle.dump(kwargs, file, path='/kwargs')

    # This function reads in a backup made by _make_backup
    # TODO: Allow for absolute path to backup file to be given?
    def _read_backup(self, emul_i, *, suffix=None):
        """
        Reads in a backup HDF5-file created by the :meth:`~_make_backup`
        method, using the provided `emul_i` and the value of :attr:`~name`.

        Parameters
        ----------
        emul_i : int
            The emulator iteration that was provided to the :meth:`~call_model`
            method when the backup was made.

        Optional
        --------
        suffix : str or None. Default: None
            The suffix of the backup file (everything between parentheses) that
            needs to be read. If *None*, the last created backup will be read.

        Returns
        -------
        filename : str
            The absolute path to the backup file that has been read.
        data : dict with keys `('emul_i', 'prism_version', 'par_set',` \
            `'data_idx', 'args', 'kwargs')`
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

        # Check if provided suffix is None or a string
        if suffix is None:
            suffix = ''
        else:
            suffix = check_vals(suffix, 'suffix', 'str')

        # Obtain name of backup file
        filepath = self._get_backup_path(emul_i, suffix)

        # Initialize empty data dict
        data = sdict()

        # Read emul_i, par_set, data_idx, args and kwargs from hdf5
        with h5py.File(filepath, 'r') as file:
            data['emul_i'] = file.attrs['emul_i']
            data['prism_version'] = file.attrs['prism_version']
            data['par_set'] = sdict(hickle.load(file, path='/par_set'))
            data['data_idx'] = hickle.load(file, path='/data_idx')
            data['args'] = hickle.load(file, path='/args')
            data['kwargs'] = hickle.load(file, path='/kwargs')

        # Return data
        return(filepath, data)

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
        model_parameters =\
            convert_parameters(self.get_default_model_parameters())

        # If additional model parameters information is given, add it
        if add_model_parameters is not None:
            model_parameters.update(convert_parameters(add_model_parameters))

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
        for i, (name, (*rng, est)) in enumerate(model_parameters.items()):
            # Save parameter name, range and est
            self._par_name.append(name)
            self._par_rng[i] = rng
            self._par_est.append(est)

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
        model_data = convert_data(self.get_default_model_data())

        # If additional model data information is given, add it
        if add_model_data is not None:
            model_data.update(convert_data(add_model_data))

        # Determine the number of data points
        self._n_data = check_vals(len(model_data), 'n_data', 'pos')

        # Create empty data value, error, space and identifier lists
        self._data_val = []
        self._data_err = []
        self._data_spc = []
        self._data_idx = []

        # Save model data as class properties
        for idx, (val, *err, spc) in model_data.items():
            # Save data value, errors, space and identifier
            self._data_val.append(val)
            self._data_err.append(err)
            self._data_spc.append(spc)
            self._data_idx.append(idx)

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
            dict item is formatted as ``{par_name: par_val}``. If multi-called,
            it is formatted as ``{par_name: [par_val_1, par_val_2, ...,
            par_val_n]}``.
        data_idx : list of tuples
            List containing the user-defined data point identifiers
            corresponding to the requested data points.

        Returns
        -------
        data_val : 1D or 2D array_like or dict
            Array containing the data values corresponding to the requested
            data points generated by the requested model realization(s). If
            model is multi-called, `data_val` is of shape ``(n_sam, n_data)``.
            If dict, it has the identifiers in `data_idx` as its keys with
            either scalars or 1D array_likes as its values.

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
            If dict, it has the identifiers in `data_idx` as its keys with
            either scalars or 1D array_likes of length 2 as its values.

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
