# -*- coding: utf-8 -*-

# %% IMPORTS
from __future__ import absolute_import, division, print_function

# Built-in imports
from os import path

# Package imports
import numpy as np

# PRISM imports
from prism._internal import check_instance
from prism.modellink import GaussianLink, ModelLink

# Save the path to this directory
dirpath = path.dirname(__file__)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for GaussianLink class
def test_GaussianLink():
    # Save paths to data file
    model_data = path.join(dirpath, 'data/data_gaussian.txt')

    # Initialize GaussianLink class
    model_link = GaussianLink(3, model_data=model_data)
    repr(model_link)

    # Check if this instance is from a proper ModelLink subclass
    assert check_instance(model_link, ModelLink)
    assert model_link.n_gaussians == 3

    # Call model
    par_set = [2.5, 2, 1, 2.5, 3, 1, 2.5, 4, 1]
    par_dict = dict(zip(model_link._par_name, np.array(par_set)))
    exp_mod_out = [4.853169333697371, 4.5858319665035, 4.0377509940191105]
    assert np.isclose(model_link.call_model(1, par_dict,
                                            sorted(model_link._data_idx)),
                      exp_mod_out).all()

    # Retrieve model discrepancy variance
    assert np.isclose(model_link.get_md_var(1, par_set, model_link._data_idx),
                      [0.01, 0.01, 0.01]).all()
