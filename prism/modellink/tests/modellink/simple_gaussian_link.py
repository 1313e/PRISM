# -*- coding: utf-8 -*-

# Simple Gaussian ModelLink
# Compatible with Python 3.5+


# %% IMPORTS
# Package imports
import numpy as np

# PRISM imports
from prism.modellink import ModelLink

# All declaration
__all__ = ['GaussianLink2D', 'GaussianLink3D']


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
        # Request single or multi model calls
        self.call_type = 'hybrid'

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super(GaussianLink2D, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = {'A': [1, 5, 2.5],
                    'B': [1, 3, 2]}
        return(par_dict)

    @property
    def _default_model_data(self):
        model_data = {1: [1, 0.05],
                      3: [2, 0.05],
                      4: [3, 0.05]}
        return(model_data)

    def call_model(self, emul_i, par_set, data_idx):
        par = par_set
        mod_set = {idx: 0 for idx in data_idx}
        for idx in data_idx:
            mod_set[idx] += par['A']*np.exp(-1*((idx-par['B'])**2/(2)))

        return(mod_set)

    def get_md_var(self, emul_i, par_set, data_idx):
        return({idx: 0.1**2 for idx in data_idx})


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
        # Do not set flags to cover auto-setting them
        # Inheriting ModelLink __init__()
        super(GaussianLink3D, self).__init__(*args, **kwargs)

    def call_model(self, emul_i, par_set, data_idx):
        par = par_set
        mod_set = par['A']*np.exp(-1*((data_idx-par['B'])**2/(2*par['C']**2)))

        return(mod_set)

    def get_md_var(self, *args, **kwargs):
        super(GaussianLink3D, self).get_md_var(*args, **kwargs)
