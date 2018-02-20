# -*- coding: utf-8 -*-

# Emulator model wrapper/link definition, Ellert van der Velden, 2017
# Do not run directly

# %% IMPORTS
from __future__ import absolute_import, division, print_function
from six import with_metaclass

import abc
import numpy as np
from os import path

__all__ = ['ModelLink']


# %% CLASS DEFINITION
class ModelLink(with_metaclass(abc.ABCMeta, object)):
    """
    Provides a base class definition that allows the :class:`~Emulator` class
    to be linked to any model/test object of choice. Every model wrapper class
    used in the :class:`~Emulator` class must be an instance of the
    :class:`~ModelLink` class.

    """

    def __init__(self, model_parameters=None, model_data=None):
        """
        Initialize an instance of the :class:`~ModelLink` base class.

        Optional
        --------
        model_parameters : array_like, dict, str or None. Default: None
            Anything that can be converted to a dict that provides non-default
            model parameters information or *None* if only default information
            is used from :func:`~_default_model_parameters`. For more
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
            if only default data is used from :func:`~_default_model_data`. For
            more information on the lay-out of this array, see ``Notes``.

        Notes
        -----
        TODO: Write the notes

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
                self._model_data[0].append(val)
                self._model_data[1].append(err)
                self._model_data[2].append(int(idx))

        # If anything else is given, it must be array_like
        else:
            try:
                new_model_data = np.array(new_model_data)
            except Exception:
                raise TypeError("Input is not array_like!")
            else:
                # Update the model data list
                for i, data in enumerate(new_model_data):
                    self._model_data[0].append(data[0])
                    self._model_data[1].append(data[1])
                    self._model_data[2].append(int(data[2]))

    def _create_properties(self):
        # Save model data as class properties
        self._par_dim = len(self._model_parameters.keys())

        # Create empty parameter name, ranges and estimate arrays
        self._par_names = []
        self._par_rng = np.zeros([self._par_dim, 2])
        self._par_rng[:, 1] = 1
        self._par_estimate = []

        # Save parameter ranges
        for i, (name, values) in enumerate(self._model_parameters.items()):
            self._par_names.append(name)
            self._par_rng[i] = (values[0], values[1])
            try:
                self._par_estimate.append(values[2] if
                                          values[2] != np.infty else None)
            except IndexError:
                self._par_estimate.append(None)

        # Save model data as class properties
        self._n_data = len(self._model_data[0])
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
