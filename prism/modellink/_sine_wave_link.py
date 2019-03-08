# -*- coding: utf-8 -*-

# Simple Sine Wave ModelLink
# Compatible with Python 3.5+

"""
SineWaveLink
============
Provides the definition of the :class:`~SineWaveLink` class.

"""


# %% IMPORTS
# Package imports
import numpy as np

# PRISM imports
from prism._internal import np_array
from prism.modellink import ModelLink

# All declaration
__all__ = ['SineWaveLink']


# %% CLASS DEFINITION
class SineWaveLink(ModelLink):
    """
    :class:`~prism.modellink.ModelLink` class wrapper for a simple sine wave
    model, used for testing the functionality of the *PRISM* pipeline.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`A+0.1*B*\\sin(C*x+D)` to obtain the data value.

    """

    def __init__(self, *args, **kwargs):
        # Request single or multi model calls
        self.call_type = 'hybrid'

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super().__init__(*args, **kwargs)

    def get_default_model_parameters(self):
        par_dict = {'A': [2, 7, 4],
                    'B': [-1, 12, 3],
                    'C': [0, 10, 5],
                    'D': [1.5, 5, 4.6]}
        return(par_dict)

    def call_model(self, emul_i, par_set, data_idx):
        par = par_set
        mod_set = [0]*len(data_idx)
        for i, idx in enumerate(data_idx):
            mod_set[i] = par['A']+0.1*par['B']*np.sin(par['C']*idx+par['D'])

        return(np_array(mod_set).T)

    def get_md_var(self, emul_i, par_set, data_idx):
        return(pow(0.1*np.ones_like(data_idx), 2))
