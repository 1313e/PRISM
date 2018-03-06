# -*- coding: utf-8 -*-

"""
ModelLink
=========
Provides the definition of the :class:`~ModelLink` abstract base class.


Available classes
-----------------
:class:`~ModelLink`
    Provides an abstract base class definition that allows the
    :class:`~Pipeline` class to be linked to any model/test object of choice.
    Every model wrapper class used in the :class:`~Pipeline` class must be an
    instance of the :class:`~ModelLink` class.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import abc
from os import path
from six import with_metaclass

# Package imports
import numpy as np

# PRISM imports
from .._internal import check_float, check_int, check_pos_int, check_str

# All declaration
__all__ = ['ModelLink']


# %% CLASS DEFINITION
class ModelLink(with_metaclass(abc.ABCMeta, object)):
    """
    Provides an abstract base class definition that allows the
    :class:`~Pipeline` class to be linked to any model/test object of choice.
    Every model wrapper class used in the :class:`~Pipeline` class must be an
    instance of the :class:`~ModelLink` class.

    """

    def __init__(self, model_parameters=None, model_data=None):
        """
        Initialize an instance of the :class:`~ModelLink` abstract base class.

        Optional
        --------
        model_parameters : array_like, dict, str or None. Default: None
            Anything that can be converted to a dict that provides non-default
            model parameters information or *None* if only default information
            is used from :meth:`~_default_model_parameters`. For more
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
            if only default data is used from :meth:`~_default_model_data`. For
            more information on the lay-out of this array, see ``Notes``.

        Notes
        -----
        The model parameters dict requires to have the name of the parameters
        as the keyword, and a 1D list containing the lower bound, the upper
        bound and, if applicable, the estimate of this parameter.
        An example of a model parameters file can be found in the 'data' folder
        of the PRISM package.

        The model data array contains the data values in the first column, the
        data errors in the second column and, if applicable, the data index in
        the third column. The data index is a supportive index that can be used
        to determine which model data output needs to be given to the PRISM
        pipeline. The pipeline itself does not require this data index.
        An example of a model data file can be found in the 'data' folder of
        the PRISM package.

        """

        # Obtain default model parameters dict
        self._model_parameters = self._default_model_parameters

        # Check if model parameters have been provided manually
        if model_parameters is not None:
            self.model_parameters = model_parameters

        # Obtain default model data list
        self._model_data = self._default_model_data

        # Check if model data has been provided manually
        if model_data is not None:
            self.model_data = model_data

        # Save the model parameters and data as class properties
        self._create_properties()

    @property
    def _default_model_parameters(self):
        return({})

    @property
    def model_parameters(self):
        return(self._model_parameters)

    @model_parameters.setter
    def model_parameters(self, new_model_parameters):
        """
        Generates the custom model parameter dict from input argument
        `new_model_parameters`.

        Parameters
        ----------
        new_model_parameters : array_like, dict or str

        Generates
        ---------
        model_parameters : dict
            Updated dict containing the names, ranges and estimates of all
            default and non-default model parameters.

        """

        # If a parameter file is given
        if isinstance(new_model_parameters, str):
            par_file = path.abspath(new_model_parameters)

            # Read the parameter-file and obtain parameter names, ranges
            # (and estimates if provided)
            par_names = np.genfromtxt(par_file, dtype=str, usecols=0)
            par_values = np.genfromtxt(par_file, dtype=float,
                                       filling_values=np.infty)[:, 1:]

            # Update the model parameters dict
            self._model_parameters.update(zip(par_names, par_values))

        # If a parameter dict is given
        elif isinstance(new_model_parameters, dict):
            self._model_parameters.update(new_model_parameters)

        # If anything else is given
        else:
            # Check if it can be converted to a dict
            try:
                par_dict = dict(new_model_parameters)
            except Exception:
                raise TypeError("Input cannot be converted to a dict!")
            else:
                self._model_parameters.update(par_dict)

    @property
    def _default_model_data(self):
        return([[], [], []])

    @property
    def model_data(self):
        return(self._model_data)

    # TODO: Allow no data_idx to be provided
    @model_data.setter
    def model_data(self, new_model_data):
        """
        Generates the custom model data list from input argument
        `new_model_data`.

        Parameters
        ---------
        new_model_data : array_like or str

        Generates
        ---------
        model_data : list
            Updated list containing the values, errors and set indices of all
            default and non-default model data.

        """

        # If a data file is given
        if isinstance(new_model_data, str):
            data_file = path.abspath(new_model_data)

            # Read the data-file and obtain data points
            data = np.genfromtxt(data_file, dtype=(float))

            # Make sure that data is a 2D numpy array
            data = np.array(data, ndmin=2)

            # Save data values, errors and set indices to the correct arrays
            data_val = data[:, 0].tolist()
            data_err = data[:, 1].tolist()
            data_idx = data[:, 2].tolist()

            # Update the model data list
            for val, err, idx in zip(data_val, data_err, data_idx):
                self._model_data[0].append(check_float(val, 'data_val'))
                self._model_data[1].append(check_float(err, 'data_err'))
                self._model_data[2].append(check_int(int(idx), 'data_idx'))

        # If anything else is given, it must be array_like
        else:
            try:
                new_model_data = np.array(new_model_data)
            except Exception:
                raise TypeError("Input is not array_like!")
            else:
                # Update the model data list
                for i, data in enumerate(new_model_data):
                    self._model_data[0].append(check_float(data[0],
                                                           'data_val'))
                    self._model_data[1].append(check_float(data[1],
                                                           'data_err'))
                    self._model_data[2].append(check_int(int(data[2]),
                                                         'data_idx'))

    def _create_properties(self):
        # Save model data as class properties
        self._par_dim = check_pos_int(len(self._model_parameters.keys()),
                                      'par_dim')

        # Create empty parameter name, ranges and estimate arrays
        self._par_names = []
        self._par_rng = np.zeros([self._par_dim, 2])
        self._par_rng[:, 1] = 1
        self._par_estimate = []

        # Save parameter ranges
        for i, (name, values) in enumerate(self._model_parameters.items()):
            self._par_names.append(check_str(name, 'par_name'))
            self._par_rng[i] = (check_float(values[0], 'lower_par'),
                                check_float(values[1], 'upper_par'))
            try:
                self._par_estimate.append(
                    check_float(values[2], 'par_estimate') if
                    values[2] != np.infty else None)
            except IndexError:
                self._par_estimate.append(None)

        # Save model data as class properties
        self._n_data = check_pos_int(len(self._model_data[0]), 'n_data')
        self._data_val = self._model_data[0]
        self._data_err = self._model_data[1]
        self._data_idx = self._model_data[2]

    @abc.abstractmethod
    def call_model(self, emul_i, model_parameters, data_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def get_md_var(self, emul_i, data_idx):
        raise NotImplementedError


# %% CLASS PROPERTIES
    # Model Parameters
    @property
    def par_dim(self):
        """
        Number of model parameters.

        """

        return(self._par_dim)

    @property
    def par_names(self):
        """
        List with model parameter names.

        """

        return(self._par_names)

    @property
    def par_rng(self):
        """
        Array containing the lower and upper values of the model parameters.

        """

        return(self._par_rng)

    @property
    def par_estimate(self):
        """
        Array containing user-defined estimated values of the model parameters.
        Contains *None* in places where estimates were not provided.

        """

        return(self._par_estimate)

    # Model Data
    @property
    def n_data(self):
        """
        Number of provided data points.

        """

        return(self._n_data)

    @property
    def data_val(self):
        """
        Array with values of provided data points.

        """

        return(self._data_val)

    @property
    def data_err(self):
        """
        Array with errors of provided data points.

        """

        return(self._data_err)

    @property
    def data_idx(self):
        """
        Array with user-defined data point identifiers.

        """

        return(self._data_idx)
