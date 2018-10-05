# -*- coding: utf-8 -*-

# Simple Sine Wave ModelLink
# Compatible with Python 2.7 and 3.5+

"""
SineWaveLink
============
Provides the definition of the :class:`~SineWaveLink` class.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
import numpy as np

# PRISM imports
from .modellink import ModelLink

# All declaration
__all__ = ['SineWaveLink']


# %% CLASS DEFINITION
class SineWaveLink(ModelLink):
    """
    :class:`~ModelLink` class wrapper for a simple sine wave model, used for
    testing the functionality of the *PRISM* pipeline in unittests.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`A+0.1*B*\\sin(C*x+D)` to obtain the data value.

    """

    def __init__(self, *args, **kwargs):
        # Request single model calls
        self.multi_call = False

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super(SineWaveLink, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = {'A': [2, 7, 4],
                    'B': [-1, 12, 3],
                    'C': [0, 10, 5],
                    'D': [1.5, 5, 4.6]}
        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        par = model_parameters
        mod_set =\
            par['A']+0.1*par['B']*np.sin(par['C']*np.array(data_idx)+par['D'])

        return(mod_set)

    def get_md_var(self, emul_i, data_idx):
        return(pow(0.1*np.ones_like(data_idx), 2))
