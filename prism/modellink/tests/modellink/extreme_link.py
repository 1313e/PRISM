# -*- coding: utf-8 -*-

# Simple Extreme Outliers ModelLink
# Compatible with Python 3.5+


# %% IMPORTS
# Package imports
import numpy as np

# PRISM imports
from prism.modellink import ModelLink

# All declaration
__all__ = ['ExtremeLink']


# %% CLASS DEFINITION
class ExtremeLink(ModelLink):
    """
    :class:`~prism.modellink.ModelLink` class wrapper for a simple model with
    extreme outliers, used for testing the functionality of the *PRISM*
    pipeline in tests.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`A*x^2+B`. If :math:`x=0` and :math:`-1\\leq A\\leq 1`, this
        function instead always returns :math:`-100`.

    """

    def __init__(self, *args, **kwargs):
        # Request only multi model calls
        self.call_type = 'multi'

        # Request only controller calls
        self.MPI_call = False

        # Call ModelLink's __init__()
        super().__init__(*args, **kwargs)

    def get_default_model_parameters(self):
        # Define default parameters
        par_dict = {'A': [-10, 10, 2],
                    'B': [0, 10, 5]}
        return(par_dict)

    def get_default_model_data(self):
        # Define default data
        data_dict = {-5: [55, 0.01],
                     -3: [23, 0.01],
                     -1: [7, 0.01],
                     0: [5, 0.01],
                     1: [7, 0.01],
                     3: [23, 0.01],
                     5: [55, 0.01]}
        return(data_dict)

    # Override call_model method
    def call_model(self, emul_i, par_set, data_idx):
        # Extract values for A and B
        A = par_set['A']
        B = par_set['B']

        # Calculate mod_set
        mod_set = []
        for idx in data_idx:
            mod_set.append(A*idx**2+B)

        # Transform modset to NumPy array
        mod_set = np.array(mod_set).T

        # If data_idx was 0 and -1 <= A <= 1, set value to -100
        idx0 = data_idx.index(0)
        A_range = (-1 <= A)*(A <= 1)
        mod_set[A_range, idx0] = -100

        # Return mod_set
        return(mod_set)

    # Override get_md_var method
    def get_md_var(self, emul_i, par_set, data_idx):
        super().get_md_var(emul_i, par_set, data_idx)
