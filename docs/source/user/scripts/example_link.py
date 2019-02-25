# -*- coding: utf-8 -*-

# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
import numpy as np

# PRISM imports
from prism.modellink import ModelLink


# ExampleLink class definition
class ExampleLink(ModelLink):
    def __init__(self, *args, **kwargs):
        # Perform any custom operations here
        pass

        # Call ModelLink's __init__()
        super().__init__(*args, **kwargs)

    def get_default_model_parameters(self):
        # Define default parameters (not required)
        par_dict = {}
        return(par_dict)

    def get_default_model_data(self):
        # Define default data (not required)
        data_dict = {}
        return(data_dict)

    # Override call_model method
    def call_model(self, emul_i, par_set, data_idx):
        # Perform operations for obtaining the model output
        # Following is provided:
        # 'emul_i': Requested iteration
        # 'par_set': Requested sample(s)
        # 'data_idx': Requested data point(s)
        pass

    # Override get_md_var method
    def get_md_var(self, emul_i, par_set, data_idx):
        # Perform operations for obtaining the model discrepancy variance
        # Following is provided:
        # 'emul_i': Requested iteration
        # 'par_set': Requested sample
        # 'data_idx': Requested data point(s)
        pass
