# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
import numpy as np
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism.modellink import PolyLink, test_subclass as _test_subclass


# %% GLOBALS
DIR_PATH = path.dirname(__file__)           # Path to directory of this file


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for PolyLink class
def test_PolyLink():
    # Save paths to data file
    model_data = path.join(DIR_PATH, 'data/data_poly.txt')

    # Initialize PolyLink class
    modellink_obj = _test_subclass(PolyLink, 3, model_data=model_data)
    repr(modellink_obj)

    # Check if this instance is has the correct number of polynomial terms
    assert modellink_obj.order == 3

    # Call model
    par_set = [2.5, 2, 1, 2.5]
    par_dict = sdict(zip(modellink_obj._par_name, np.array(par_set)))
    exp_mod_out = [8.0, 85.0, 186.5]
    assert np.allclose(modellink_obj.call_model(
                        1, par_dict, sorted(modellink_obj._data_idx)),
                       exp_mod_out)

    # Retrieve model discrepancy variance
    assert np.allclose(modellink_obj.get_md_var(
                        1, par_dict, modellink_obj._data_idx),
                       [0.01, 0.01, 0.01])
