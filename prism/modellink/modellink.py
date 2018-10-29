# -*- coding: utf-8 -*-

"""
ModelLink
=========
Provides the definition of the :class:`~ModelLink` abstract base class.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import abc
from os import path
import sys
import warnings

# Package imports
from e13tools import InputError
import numpy as np
from six import with_metaclass
from sortedcontainers import SortedDict, SortedSet

# PRISM imports
from .._docstrings import std_emul_i_doc
from .._internal import (check_vals, convert_str_seq, docstring_substitute,
                         getCLogger, raise_error)

# All declaration
__all__ = ['ModelLink']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% MODELLINK CLASS DEFINITION
# TODO: Allow for inter-process methods?
# Like, having a method that is called before/after construction.
class ModelLink(with_metaclass(abc.ABCMeta, object)):
    """
    Provides an abstract base class definition that allows the
    :class:`~prism.pipeline.Pipeline` class to be linked to any model/test
    object of choice. Every model wrapper used in the
    :class:`~prism.pipeline.Pipeline` class must be an instance of the
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

    Note
    ----
    The :meth:`~__init__` method may be extended by the :class:`~ModelLink`
    subclass, but the superclass version must always be called.

    """

    def __init__(self, model_parameters=None, model_data=None):
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

            If array_like, dict(model_parameters/model_data) must generate a
            dict with the correct lay-out.
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
        provided lower and upper errors.

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
        single element, it is replaced by just that element instead of a list.

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
            ``lower_data_err, upper_data_err, data_spc]}``

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
        self._set_model_parameters(model_parameters)

        # Generate model data properties
        self._set_model_data(model_data)

    # Define the representation of a ModelLink object
    def __repr__(self):
        # Obtain representation of model_parameters
        par_repr = []
        for name, rng, est in zip(self._par_name, self._par_rng,
                                  self._par_est):
            if est is None:
                par_repr.append("%r: %r" % (name, [rng[0], rng[1]]))
            else:
                par_repr.append("%r: %r" % (name, [rng[0], rng[1], est]))
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
            # Make sure that if idx is a list, it is returned as a tuple key
            if isinstance(idx, list):
                idx = tuple(idx)
            data_repr.append("%r: %r" % (idx, point))
        data_repr = "model_data={%s}" % (", ".join(map(str, data_repr)))

        # Obtain non-default representation and add default ones
        str_repr = self._get_str_repr()
        str_repr.extend([par_repr, data_repr])

        # Return representation
        return("%s(%s)" % (self.__class__.__name__, ", ".join(str_repr)))

    def _get_str_repr(self):
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
                if isinstance(par_idx, (str, unicode)):
                    par_seq[i] = self._par_name.index(par_idx)
                # If not, try to use it as a parameter index
                else:
                    self._par_name[par_idx]
                    par_seq[i] = par_idx % self._n_par
            # If any operation above fails, raise error
            except Exception as error:
                err_msg = "Input argument %r is invalid! (%s)" % (name, error)
                raise_error(InputError, err_msg, logger)

        # If everything went without exceptions, check if list is not empty and
        # remove duplicates
        if len(par_seq):
            par_seq = list(SortedSet(par_seq))
        else:
            err_msg = "Input argument %r is empty!" % (name)
            raise_error(ValueError, err_msg, logger)

        # Log end
        logger.info("Finished converting sequence of model parameter "
                    "names/indices.")

        # Return it
        return(par_seq)

    @property
    def _default_model_parameters(self):
        """
        dict: The default model parameters to use for every instance of this
        :class:`~ModelLink` subclass.

        """

        return(SortedDict())

    def get_default_model_parameters(self):
        """
        Returns the default model parameters to use for every instance of this
        :class:`~ModelLink` subclass. By default, returns
        :attr:`~_default_model_parameters`.

        """

        return(self._default_model_parameters)

    def _set_model_parameters(self, add_model_parameters):
        """
        Generates the model parameter properties from the default model
        parameters and the additional input argument `add_model_parameters`.

        Parameters
        ----------
        add_model_parameters : array_like, dict, str or None
            Anything that can be converted to a dict that provides non-default
            model parameters information or *None* if only default information
            is used from :meth:`~get_default_model_parameters`.

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
        model_parameters = SortedDict(self.get_default_model_parameters())

        # If no additional model parameters information is given
        if add_model_parameters is None:
            pass

        # If a parameter file is given
        elif isinstance(add_model_parameters, (str, unicode)):
            # Obtain absolute path to given file
            par_file = path.abspath(add_model_parameters)

            # Read the parameter file in as a string
            pars = np.genfromtxt(par_file, dtype=(str), delimiter=':',
                                 autostrip=True)

            # Make sure that pars is 2D
            pars = np.array(pars, ndmin=2)

            # Combine default parameters with read-in parameters
            model_parameters.update(pars)

        # If a parameter dict is given
        elif isinstance(add_model_parameters, dict):
            model_parameters.update(add_model_parameters)

        # If anything else is given
        else:
            # Check if it can be converted to a dict
            try:
                par_dict = SortedDict(add_model_parameters)
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
                self._par_est.append(check_vals(
                    values[2], 'par_est[%s]' % (name), 'float'))
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
        :attr:`~_default_model_data`.

        """

        return(self._default_model_data)

    def _set_model_data(self, add_model_data):
        """
        Generates the model data properties from the default model data and the
        additional input argument `add_model_data`.

        Parameters
        ---------
        add_model_data : array_like, dict, str or None
            Anything that can be converted to a dict that provides non-default
            model data information or *None* if only default data is used from
            :meth:`~get_default_model_data`.

        Generates
        ---------
        n_data : int
            Number of provided data points.
        data_val : list
            List with values of provided data points.
        data_err : list
            List with lower and upper :math:`1\\sigma`-confidence levels of
            provided data points.
        data_spc : list
            List with types of value space ({'lin', 'log', 'ln'}) of provided
            data points.
        data_idx : list
            List with user-defined data point identifiers.

        """

        # Obtain default model data
        model_data = dict(self.get_default_model_data())

        # If no additional model data information is given
        if add_model_data is None:
            pass

        # If a data file is given
        elif isinstance(add_model_data, (str, unicode)):
            # Obtain absolute path to given file
            data_file = path.abspath(add_model_data)

            # Read the data file in as a string
            data_points = np.genfromtxt(data_file, dtype=(str),
                                        delimiter=':', autostrip=True)

            # Make sure that data_points is 2D
            data_points = np.array(data_points, ndmin=2)

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
            # Convert idx from tuple to a list or element
            if(len(idx) == 1):
                idx = idx[0]
            else:
                idx = list(idx)

            # Save data value
            self._data_val.append(check_vals(data[0], 'data_val[%s]' % (idx),
                                             'float'))

            # Save data error and extract space
            # If length is two, centered error and no data space were given
            if(len(data) == 2):
                self._data_err.append(
                    [check_vals(data[1], 'data_err[%s]' % (idx), 'float')]*2)
                spc = 'lin'

            # If length is three, there are two possibilities
            elif(len(data) == 3):
                # If the third column contains a string, it is the data space
                if isinstance(data[2], (str, unicode)):
                    self._data_err.append(
                        [check_vals(data[1], 'data_err[%s]' % (idx),
                                    'float')]*2)
                    spc = data[2]

                # If the third column contains no string, it is error interval
                else:
                    self._data_err.append(
                        check_vals(data[1:3], 'data_err[%s]' % (idx), 'float'))
                    spc = 'lin'

            # If length is four+, error interval and data space were given
            else:
                self._data_err.append(
                    check_vals(data[1:3], 'data_err[%s]' % (idx), 'float'))
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
                raise ValueError("Input argument 'data_spc[%s]' is "
                                 "invalid (%r)!" % (idx, spc))

            # Save data identifier
            self._data_idx.append(idx)

    @abc.abstractmethod
    @docstring_substitute(emul_i=std_emul_i_doc)
    def call_model(self, emul_i, model_parameters, data_idx):
        """
        Calls the model wrapped in this :class:`~ModelLink` subclass at
        emulator iteration `emul_i` for model parameter values
        `model_parameters` and returns the data points corresponding to
        `data_idx`.

        This is an abstract method and must be overridden by the
        :class:`~ModelLink` subclass.

        Parameters
        ----------
        %(emul_i)s
        model_parameters : dict of :class:`~numpy.float64`
            Dict containing the values for all model parameters corresponding
            to the requested model realization(s). If model is single-called,
            dict is formatted as ``{par_name: par_val}``. If multi-called, it
            is formatted as ``{par_name: [par_val_1, par_val_2, ...,
            par_val_n]}``.
        data_idx : list of lists
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
        raise NotImplementedError

    @abc.abstractmethod
    @docstring_substitute(emul_i=std_emul_i_doc)
    def get_md_var(self, emul_i, data_idx):
        """
        Calculates the linear model discrepancy variance at a given emulator
        iteration `emul_i` for given data points `data_idx` for the model
        wrapped in this :class:`~ModelLink` subclass.

        This is an abstract method and must be overridden by the
        :class:`~ModelLink` subclass.

        Parameters
        ----------
        %(emul_i)s
        data_idx : list of lists
            List containing the user-defined data point identifiers
            corresponding to the requested data points.

        Returns
        -------
        md_var : 1D or 2D array_like
            Array containing the linear model discrepancy variance values
            corresponding to the requested data points. If 1D array_like, data
            is assumed to have a centered one sigma confidence interval. If 2D
            array_like, the values determine the lower and upper variances and
            the array is of shape ``(n_data, 2)``.

        Notes
        -----
        The returned model discrepancy variance values must be of linear form,
        even for those data values that are returned in logarithmic form by the
        :meth:`~call_model` method. If not, the possibility exists that the
        emulation process will not converge properly.

        """

        # Raise NotImplementedError if only super() was called
        raise NotImplementedError

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

    @multi_call.setter
    def multi_call(self, multi_call):
        warn_msg = ("Setting property 'multi_call' is deprecated since v0.5.3."
                    "Use the 'call_type' property instead.")
        warnings.warn(warn_msg, stacklevel=2)

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
        list of float or None: The user-defined estimated values of the model
        parameters. Contains *None* in places where estimates were not
        provided.

        """

        return(self._par_est)

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
        list of float: The lower and upper :math:`1\\sigma`-confidence levels
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
        list of lists: The user-defined data point identifiers.

        """

        return(self._data_idx)
