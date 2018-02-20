# -*- coding: utf-8 -*-

# Simple Sine Wave ModelLink
# Compatible with Python 2.7 and 3.4+

# %% IMPORTS
from __future__ import absolute_import, division, print_function

import numpy as np
from modellink import ModelLink

__all__ = ['SineWaveLink']


# %% CLASS DEFINITION
class SineWaveLink(ModelLink):
    def __init__(self, model_data='data_sine_wave.txt'):
        super(SineWaveLink, self).__init__(model_data=model_data)

    @property
    def _default_model_parameters(self):
        par_dict = {'A': [0, 10, 5],
                    'B': [-1, 12, 3],
                    'C': [2, 7, 4],
                    'D': [1.5, 5, 4.6]}
        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        par = model_parameters
        mod_set =\
            par['C']+0.1*par['B']*np.sin(par['A']*np.array(data_idx)+par['D'])

        return(mod_set)

    def get_md_var(self, emul_i, data_idx):
        super(SineWaveLink, self).get_md_var(emul_i=emul_i, data_idx=data_idx)
