# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
import numpy as np
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism.modellink import SineWaveLink, test_subclass as _test_subclass


# %% GLOBALS
DIR_PATH = path.dirname(__file__)           # Path to directory of this file


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for SineWaveLink class
def test_SineWaveLink():
    # Save paths to data file
    model_data = path.join(DIR_PATH, 'data/data_sine_wave.txt')

    # Initialize SineWaveLink class
    modellink_obj = _test_subclass(SineWaveLink, model_data=model_data)
    repr(modellink_obj)

    # Call model
    par_set = [4, 3, 5, 4.6]
    par_dict = sdict(zip(modellink_obj._par_name, np.array(par_set)))
    exp_mod_out = [3.9477019656331063, 4.268437351642151, 4.204589086020441,
                   3.8476310228828132, 3.7089682798878445]
    assert np.allclose(modellink_obj.call_model(
                        1, par_dict, sorted(modellink_obj._data_idx)),
                       exp_mod_out)

    # Retrieve model discrepancy variance
    assert np.allclose(modellink_obj.get_md_var(
                        1, par_dict, modellink_obj._data_idx),
                       [0.01, 0.01, 0.01, 0.01, 0.01])
