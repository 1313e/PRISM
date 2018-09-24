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

# Package imports
from e13tools import InputError
import numpy as np
from six import with_metaclass
from sortedcontainers import SortedDict

# PRISM imports
from .._docstrings import std_emul_i_doc
from .._internal import check_val, convert_str_seq, docstring_substitute

# All declaration
__all__ = ['ModelLink']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% MODELLINK CLASS DEFINITION
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
    subclass will be used by *PRISM*: :attr:`~name`, :attr:`~multi_call` and
    :attr:`~MPI_call`. The first defines what the name of the subclass is,
    which is used by *PRISM* to identify the subclass with and check if one did
    not use a different subclass by accident. The other two are switches that
    determine how the :meth:`~call_model` method should be used. These three
    properties can be set anywhere during the initialization of the
    :class:`~ModelLink` subclass, or are set to a default value if they are not
    modified.

    The model parameters and comparison data can be set in two different ways.
    They can be hard-coded into the :class:`~ModelLink` subclass by altering
    the :attr:`~_default_model_parameters` and :attr:`~_default_model_data`
    properties or set by providing them during class initialization. A
    combination of both is also possible. More details on this can be found in
    :meth:`~__init__`.

    Note
    ----
    The :meth:`~__init__` method may be overridden by the :class:`~ModelLink`
    subclass, but the inherited version must always be called.

    """

    def __init__(self, model_parameters=None, model_data=None):
        """
        Initialize an instance of the :class:`~ModelLink` subclass.

        Optional
        --------
        model_parameters : array_like, dict, str or None. Default: None
            Anything that can be converted to a dict that provides non-default
            model parameters information or *None* if only default information
            is used from :attr:`~_default_model_parameters`. For more
            information on the lay-out of this dict, see ``Notes``.

            If array_like, dict(model_parameters) must generate a dict with the
            correct lay-out.
            If dict, the dict itself must have the correct lay-out.
            If str, the string must be a file containing the dict keys in the
            first column and the dict values in the remaining columns, which
            combined generate a dict with the correct lay-out.
        model_data : array_like, str or None. Default: None
            Array containing the non-default data the model will be compared
            against, a filename with data that can be converted to it or *None*
            if only default data is used from :attr:`~_default_model_data`. For
            more information on the lay-out of this array, see ``Notes``.

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
            ``{par_name: [lower_bnd, upper_bnd, par_est]}`` \n
            **and/or** \n
            ``{par_name: [lower_bnd, upper_bnd]}``

        Notes (model_data)
        ------------------
        The model data array contains the data values (:attr:`~data_val`) in
        the first column; the data errors (:attr:`~data_err`) in the second
        (and third) column(s); the data spaces (:attr:`~data_spc`) in the
        third (or fourth) column and the data identifiers (:attr:`~data_idx`)
        in the remaining columns.

        If the data errors are given with one column, then the data points are
        assumed to have a centered :math:`1\\sigma`-confidence interval. If the
        data errors are given with two columns, then the data points are
        assumed to have a :math:`1\\sigma`-confidence interval defined by the
        provided lower and upper errors. The data spaces are one of five
        strings ({'lin', 'log' or 'log_10', 'ln' or 'log_e'}) indicating in
        which of the three value spaces (linear, log, ln) the data values are.

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
            ``([data_val, data_err, data_spc, data_idx_0, data_idx_1, ...,`` \
            ``data_idx_n])`` \n
            **or** \n
            ``([data_val, lower_data_err, upper_data_err, data_spc,`` \
            ``data_idx_0, data_idx_1, ..., data_idx_n])``

        """

        # Save name of this class if not saved already
        try:
            self._name
        except AttributeError:
            self.name = self.__class__.__name__

        # Set multi_call to default (False) if not modified before
        try:
            self._multi_call
        except AttributeError:
            self.multi_call = False

        # Set MPI_call to default (False) if not modified before
        try:
            self._MPI_call
        except AttributeError:
            self.MPI_call = False

        # Generate model parameter properties
        self._set_model_parameters(model_parameters)

        # Generate model data properties
        self._set_model_data(model_data)

    @property
    def _default_model_parameters(self):
        """
        dict: The default model parameters to use for every instance of this
        :class:`~ModelLink` subclass.

        """

        return(SortedDict())

    def _set_model_parameters(self, add_model_parameters):
        """
        Generates the model parameter properties from the default model
        parameters and the additional input argument `add_model_parameters`.

        Parameters
        ----------
        add_model_parameters : array_like, dict, str or None
            Anything that can be converted to a dict that provides non-default
            model parameters information or *None* if only default information
            is used from :attr:`~_default_model_parameters`.

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
        model_parameters = SortedDict(self._default_model_parameters)

        # If no additional model parameters information is given
        if add_model_parameters is None:
            pass

        # If a parameter file is given
        elif isinstance(add_model_parameters, (str, unicode)):
            # Obtain absolute path to given file
            par_file = path.abspath(add_model_parameters)

            # Read the parameter file and obtain parameter names, ranges
            # (and estimates if provided)
            par_names = np.genfromtxt(par_file, dtype=str, usecols=0)
            par_values = np.genfromtxt(par_file, dtype=float,
                                       filling_values=np.infty)[:, 1:]

            # Update the model parameters dict
            model_parameters.update(zip(par_names, par_values))

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
            self._n_par = check_val(n_par, 'n_par', 'pos')

        # Create empty parameter name, ranges and estimate lists/arrays
        self._par_name = []
        self._par_rng = np.zeros([self._n_par, 2])
        self._par_rng[:, 1] = 1
        self._par_est = []

        # Save model parameters as class properties
        for i, (name, values) in enumerate(model_parameters.items()):
            self._par_name.append(check_val(name, 'par_name[%s]' % (i), 'str'))
            self._par_rng[i] = (
                check_val(values[0], 'lower_bnd[%s]' % (i), 'float'),
                check_val(values[1], 'upper_bnd[%s]' % (i), 'float'))

            # Check if a parameter estimate was provided
            try:
                values[2]
            # If not, save it as None
            except IndexError:
                self._par_est.append(None)
            # If so, check if it was not replaced
            else:
                # If provided as None or inf, save it as None
                if values[2] in (np.infty, None):
                    self._par_est.append(None)
                # If not, check if value provided is a float and save it
                else:
                    self._par_est.append(
                        check_val(values[2], 'par_est[%s]' % (i), 'float'))

    @property
    def _default_model_data(self):
        """
        list of lists: The default model data to use for every instance of this
        :class:`~ModelLink` subclass.

        """

        return([])

    def _set_model_data(self, add_model_data):
        """
        Generates the model data properties from the default model data and the
        additional input argument `add_model_data`.

        Parameters
        ---------
        add_model_data : array_like, str or None
            Array containing the non-default data the model will be compared
            against, a filename with data that can be converted to it or *None*
            if only default data is used from :attr:`~_default_model_data`.

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

        # Obtain default data
        model_data = self._default_model_data

        # If no additional data information is given
        if add_model_data is None:
            pass

        # If a data file is given
        elif isinstance(add_model_data, (str, unicode)):
            # Obtain absolute path to given file
            data_file = path.abspath(add_model_data)

            # Read the data file and obtain data values and errors
            data_points = np.genfromtxt(data_file, dtype=float,
                                        usecols=(0, 1, 2))

            # Check how many floats were provided in the last column
            n_float = sum(np.isfinite(data_points[:, 2]))

            # Check if the last column solely contains floats or strings
            if(n_float == 0):
                # If only strings were provided, then data_err has one element
                spc_col = 2
            elif(n_float == len(data_points)):
                # If only floats were provided, then data_err has two elements
                spc_col = 3
            else:
                # If a mix of floats and strings was provided, it is invalid
                raise InputError("Input model comparison data has inconsistent"
                                 " number of data error values!")

            # Obtain data spaces
            data_spc = np.genfromtxt(data_file, dtype=str, usecols=(spc_col))

            # Obtain data identifiers
            data_str = np.genfromtxt(data_file, dtype=str, delimiter='\n')

            # Make sure that arrays are 1D/2D (n_data=1)
            data_points = np.array(data_points, ndmin=2)
            data_spc = np.array(data_spc, ndmin=1)
            data_str = np.array(data_str, ndmin=1)

            # If data_err is one column, add that column again
            if(spc_col == 2):
                data_points[:, 2] = data_points[:, 1]

            # Create empty list of data_idx
            data_idx = []

            # Convert all data_str sequences to data_idx
            for str_seq in data_str:
                # Convert str_seq and remove val, err and spc
                tmp_idx = convert_str_seq(str_seq)[spc_col+1:]

                # Check if int, float or str was provided and save it
                for i, idx in enumerate(tmp_idx):
                    # Try to convert to int or float
                    try:
                        # If string contains a dot, check if it is a float
                        if '.' in idx:
                            tmp_idx[i] = float(idx)
                        # If string contains no dot, check if it is an int
                        else:
                            tmp_idx[i] = int(idx)
                    # If cannot be converted to int or float, save as string
                    except ValueError:
                        tmp_idx[i] = idx

                # Add converted data_idx sequence to the data_idx list
                data_idx.append(tmp_idx)

            # Update the model data list
            for (val, lower_err, upper_err), spc, idx in zip(data_points,
                                                             data_spc,
                                                             data_idx):
                model_data.append([val, lower_err, upper_err, spc, idx])

        # If anything else is given, it must be array_like
        else:
            # Check if it can be iterated over
            try:
                iter(add_model_data)
            except TypeError:
                raise TypeError("Input model comparison data is not iterable!")
            else:
                # Update the model data list
                for data in add_model_data:
                    if isinstance(data[2], (str, unicode)):
                        data.insert(2, data[1])
                    model_data.append(data)

        # Save number of model data points
        self._n_data = check_val(len(model_data), 'n_data', 'pos')

        # Create empty data value, error, space and identifier lists
        self._data_val = []
        self._data_err = []
        self._data_spc = []
        self._data_idx = []

        # Save model data as class properties
        for i, data in enumerate(model_data):
            # Save data value and error
            self._data_val.append(check_val(data[0], 'data_val[%s]' % (i),
                                            'float'))
            self._data_err.append(
                [check_val(data[1], 'lower_data_err[%s]' % (i), 'float'),
                 check_val(data[2], 'upper_data_err[%s]' % (i), 'float')])

            # Check if valid data space has been provided and save if so
            spc = str(data[3]).replace("'", '').replace('"', '')
            if spc.lower() in ('lin', 'linear'):
                self._data_spc.append('lin')
            elif spc.lower() in ('log', 'log10', 'log_10'):
                self._data_spc.append('log10')
            elif spc.lower() in ('ln', 'loge', 'log_e'):
                self._data_spc.append('ln')
            else:
                raise ValueError("Input argument 'data_spc[%s]' is invalid! "
                                 "('%s')" % (i, spc))

            # Extract the data_idx from the model_data
            if(len(data) == 5):
                idx = data[4]
            else:
                idx = data[4:]

            # If idx contains a single element, save element instead of list
            # If it has no len(), then it is already a single element
            try:
                if(len(idx) == 1):
                    idx = idx[0]
            except TypeError:
                pass

            # Check if the data identifier is not an empty list
            if(idx == []):
                raise InputError("Data point %s has no identifier!" % (i))

            # Save data identifier
            self._data_idx.append(idx)

        # Check if all provided data identifiers are unique
        for i, idx in enumerate(self._data_idx):
            if(self._data_idx.count(idx) != 1):
                raise InputError("Data point %s does not have a unique "
                                 "identifier!" % (i))

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
            to the requested model realization(s). If :attr:`~multi_call` is
            *False*, dict is formatted as ``{par_name: par_val}``. If *True*,
            it is formatted as ``{par_name: [par_val_1, par_val_2, ...,
            par_val_n]}``.
        data_idx : list of lists
            List containing the user-defined data point identifiers
            corresponding to the requested data points.

        Returns
        -------
        data_val : 1D or 2D array_like
            Array containing the data values corresponding to the requested
            data points generated by the requested model realization(s). If
            :attr:`~multi_call` is *True*, `data_val` is of shape
            ``(n_sam, n_data)``.

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

    @classmethod
    def _check_subinstance(cls, instance):
        """
        Checks if provided `instance` has been initialized from a proper
        :class:`~ModelLink` subclass.

        """

        # Check if instance was initialized from a ModelLink subclass
        if not isinstance(instance, cls):
            raise TypeError

        # Retrieve a list of all ModelLink properties
        modellink_props = [prop for prop in dir(cls) if
                           isinstance(getattr(cls, prop), property)]

        # Check if call_model can be called
        try:
            instance.call_model(0, 0, 0)
        except NotImplementedError:
            return(0)
        except TypeError:
            pass

        # Check if all ModelLink properties can be called in instance
        for prop in modellink_props:
            try:
                getattr(instance, prop)
            except AttributeError:
                return(0)
        else:
            return(1)

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
        self._name = check_val(name, 'name', 'str')

    @property
    def multi_call(self):
        """
        bool: Whether :meth:`~call_model` can/should be supplied with a set of
        evaluation samples instead of a single sample.
        By default, single model calls are requested (False).

        """

        return(bool(self._multi_call))

    @multi_call.setter
    def multi_call(self, multi_call):
        self._multi_call = check_val(multi_call, 'multi_call', 'bool')

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
        self._MPI_call = check_val(MPI_call, 'MPI_call', 'bool')

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
