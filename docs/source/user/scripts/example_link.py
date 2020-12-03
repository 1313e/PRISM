# -*- coding: utf-8 -*-

# Package imports
import numpy as np

# PRISM imports
from prism.modellink import ModelLink


# ExampleLink class definition
class ExampleLink(ModelLink):
    # Extend class constructor
    def __init__(self, *args, **kwargs):
        # Perform any custom operations here
        pass

        # Set ModelLink flags (name, call_type, MPI_call)
        pass

        # Call superclass constructor
        super().__init__(*args, **kwargs)

    # Define default model parameters (optional)
    def get_default_model_parameters(self):
        par_dict = {}
        return(par_dict)

    # Define default model data (optional)
    def get_default_model_data(self):
        data_dict = {}
        return(data_dict)

    # Override call_model abstract method
    def call_model(self, emul_i, par_set, data_idx):
        # Perform operations for obtaining the model output
        # Following is provided:
        # 'emul_i': Requested iteration
        # 'par_set': Requested sample(s) dict
        # 'data_idx': Requested data points
        pass

    # Override get_md_var abstract method
    def get_md_var(self, emul_i, par_set, data_idx):
        # Perform operations for obtaining the model discrepancy variance
        # Following is provided:
        # 'emul_i': Requested iteration
        # 'par_set': Requested sample dict
        # 'data_idx': Requested data points
        pass
