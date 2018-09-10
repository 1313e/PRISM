# -*- coding: utf-8 -*-

# Simple Gaussian ModelLink
# Compatible with Python 2.7 and 3.4+

"""
GaussianLink
============
Provides the definition of the :class:`~GaussianLink` class.


Available classes
-----------------
:class:`~GaussianLink`
    :class:`~ModelLink` class wrapper for a simple Gaussian model, used for
    testing the functionality of the *PRISM* pipeline in unittests.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
import numpy as np

# PRISM imports
from .modellink import ModelLink

# All declaration
__all__ = ['GaussianLink']


# %% CLASS DEFINITION
class GaussianLink(ModelLink):
    """
    :class:`~ModelLink` class wrapper for a simple Gaussian model, used for
    testing the functionality of the *PRISM* pipeline in unittests.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`\\sum_i A_i\\exp\\left(-\\frac{(x-B_i)^2}{2C_i^2}\\right)` to
        obtain the data value.

    """

    def __init__(self, *args, **kwargs):
        # Request single model calls
        self.multi_call = False

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super(GaussianLink, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = {'A1': [1, 5, 2.5],
                    'B1': [1, 3, 2],
                    'C1': [0, 2, 1],
                    'A2': [1, 5, 2.5],
                    'B2': [2, 4, 3],
                    'C2': [0, 2, 1],
                    'A3': [1, 5, 2.5],
                    'B3': [3, 5, 4],
                    'C3': [0, 2, 1]}
        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        par = model_parameters
        mod_set =\
            par['A1']*np.exp(-1*((data_idx-par['B1'])**2/(2*par['C1']**2))) +\
            par['A2']*np.exp(-1*((data_idx-par['B2'])**2/(2*par['C2']**2))) +\
            par['A3']*np.exp(-1*((data_idx-par['B3'])**2/(2*par['C3']**2)))

        return(mod_set)

    def get_md_var(self, emul_i, data_idx):
        return(pow(0.1*np.ones(len(data_idx)), 2))
#        super(GaussianLink, self).get_md_var(emul_i=emul_i, data_idx=data_idx)
