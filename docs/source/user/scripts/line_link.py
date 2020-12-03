# -*- coding: utf-8 -*-

# Package imports
import numpy as np

# PRISM imports
from prism.modellink import ModelLink


# LineLink class definition
class LineLink(ModelLink):
    # Extend class constructor
    def __init__(self, *args, **kwargs):
        # No custom operations or flags required
        pass

        # Call superclass constructor
        super().__init__(*args, **kwargs)

    # Define default model parameters (optional)
    def get_default_model_parameters(self):
        par_dict = {
            # Intercept in [-10, 10], guess of 3
            'A': [-10, 10, 3],
            # Slope in [0, 5], guess of 1.5
            'B': [0, 5, 1.5]}
        return(par_dict)

    # Define default model data (optional)
    def get_default_model_data(self):
        data_dict = {
            # f(1) = 4.5 +- 0.1
            1: [4.5, 0.1],
            # f(2.5) = 6.8 +- 0.1
            2.5: [6.8, 0.1],
            # f(-2) = 0 +- 0.1
            -2: [0, 0.1]}
        return(data_dict)

    # Override call_model abstract method
    def call_model(self, emul_i, par_set, data_idx):
        # Calculate the value on a straight line for requested data_idx
        vals = par_set['A']+np.array(data_idx)*par_set['B']
        return(vals)

    # Override get_md_var abstract method
    def get_md_var(self, emul_i, par_set, data_idx):
        # Calculate the model discrepancy variance
        # For a straight line, this value can be set to a constant
        return(1e-4*np.ones_like(data_idx))
