# -*- coding: utf-8 -*-

# Simple Gaussian ModelLink
# Compatible with Python 2.7 and 3.4+


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
import numpy as np

# PRISM imports
from prism.modellink import ModelLink

# All declaration
__all__ = ['GaussianLink3D']


# %% CLASS DEFINITION
class GaussianLink2D(ModelLink):
    """
    :class:`~ModelLink` class wrapper for a simple Gaussian model, used for
    testing the functionality of the *PRISM* pipeline in unittests.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`A\\exp\\left(-\\frac{(x-B)^2}{2}\\right)` to
        obtain the data value.

    """

    def __init__(self, *args, **kwargs):
        # Request single model calls
        self.multi_call = False

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super(GaussianLink2D, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = {'A': [1, 5, 2.5],
                    'B': [1, 3, 2]}
        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        if self._multi_call:
            parsA = model_parameters['A']
            parsB = model_parameters['B']
            n_sam = len(parsA)
            mod_set = np.zeros([n_sam, self._n_data])
            for i, (parA, parB) in enumerate(zip(parsA, parsB)):
                mod_set[i] = parA*np.exp(-1*((data_idx-parB)**2/(2)))
        else:
            par = model_parameters
            mod_set = par['A']*np.exp(-1*((data_idx-par['B'])**2/(2)))

        return(mod_set)

    def get_md_var(self, emul_i, data_idx):
        return(pow(0.1*np.ones(len(data_idx)), 2))


class GaussianLink3D(ModelLink):
    """
    :class:`~ModelLink` class wrapper for a simple Gaussian model, used for
    testing the functionality of the *PRISM* pipeline in unittests.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`A\\exp\\left(-\\frac{(x-B)^2}{2C^2}\\right)` to
        obtain the data value.

    """

    def __init__(self, *args, **kwargs):
        # Request single model calls
        self.multi_call = False

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super(GaussianLink3D, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = {'A': [1, 5, 2.5],
                    'B': [1, 3, 2],
                    'C': [0, 2, 1]}
        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        par = model_parameters
        mod_set = par['A']*np.exp(-1*((data_idx-par['B'])**2/(2*par['C']**2)))

        return(mod_set)

    def get_md_var(self, emul_i, data_idx):
        return(pow(0.1*np.ones(len(data_idx)), 2))
