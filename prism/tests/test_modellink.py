# -*- coding: utf-8 -*-

# %% IMPORTS
from __future__ import absolute_import, division, print_function

# Built-in imports
from os import path

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from .modellink.simple_gaussian_link import GaussianLink2D, GaussianLink3D

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save paths to various files
model_data_single = path.join(dirpath, 'data/data_gaussian_single.txt')
model_data_double = path.join(dirpath, 'data/data_gaussian_double.txt')
model_data_types = path.join(dirpath, 'data/data_gaussian_types.txt')
model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')
model_parameters_invalid_est = path.join(dirpath,
                                         'data/parameters_invalid_est.txt')


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for ModelLink class exception handling
class Test_ModelLink_Exceptions(object):
    # Try to create a GaussianLink3D object with not enough model pars
    def test_n_model_par(self):
        with pytest.raises(InputError):
            par_dict = {'A': [1, 5, 2.5]}
            GaussianLink3D(model_parameters=par_dict)

    # Try to create a GaussianLink2D object with invalid model pars
    def test_invalid_model_par(self):
        with pytest.raises(TypeError):
            GaussianLink2D(model_parameters=np.array([1]))

    # Try to create a GaussianLink2D object with invalid parameter estimates
    def test_invalid_model_par_est(self):
        with pytest.raises(TypeError):
            GaussianLink2D(model_parameters=model_parameters_invalid_est)

    # Try to create a GaussianLink3D object with not enough model data
    def test_n_model_data(self):
        with pytest.raises(ValueError):
            GaussianLink3D(model_parameters=model_parameters_3D)

    # Try to create a GaussianLink2D object with invalid model data
    def test_invalid_model_data(self):
        with pytest.raises(TypeError):
            GaussianLink2D(model_data=np.array(1))

    # Try to create a GaussianLink3D object with missing data_idx
    def test_missing_data_idx(self):
        with pytest.raises(InputError):
            model_data = {(): [1, 0.05],
                          3: [2, 0.05],
                          4: [3, 0.05]}
            GaussianLink3D(model_parameters=model_parameters_3D,
                           model_data=model_data)

    # Try to create a GaussianLink3D object with invalid data_spc
    def test_invalid_data_spc(self):
        with pytest.raises(ValueError):
            model_data = {1: [1, 0.05],
                          3: [2, 0.05],
                          4: [3, 0.05, 'A']}
            GaussianLink3D(model_parameters=model_parameters_3D,
                           model_data=model_data)


# Pytest for ModelLink class versatility
class Test_ModelLink_Versatility(object):
    # Create a GaussianLink2D object with externally defined mod_par dict
    # Also use the two options of leaving par_est out
    def test_ext_mod_par_dict(self):
        par_dict = {'A': [1, 5, None],
                    'B': [1, 3]}
        model_link = GaussianLink2D(model_parameters=par_dict)
        assert model_link.par_est == [None, None]
        repr(model_link)

    # Create a GaussianLink2D object with externally defined mod_par list
    def test_ext_mod_par_list(self):
        par_list = [['A', [1, 5, 2.5]],
                    ['B', [1, 3, 2]]]
        model_link = GaussianLink2D(model_parameters=par_list)
        repr(model_link)

    # Create a GaussianLink2D object and test the value conversions
    def test_convert_to_space(self):
        model_link = GaussianLink2D()
        assert np.isclose(model_link._to_par_space([0.2, 0.7]),
                          [1.8, 2.4]).all()
        assert np.isclose(model_link._to_unit_space([1.8, 2.4]),
                          [0.2, 0.7]).all()

    # Create a GaussianLink3D object with externally defined mod_data dict
    def test_ext_mod_data_dict(self):
        data_dict = {(1, 2): [1, 0.05],
                     3: [2, 0.05],
                     4: [3, 0.05]}
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=data_dict)
        repr(model_link)

    # Create a GaussianLink3D object with externally defined mod_data list
    def test_ext_mod_data_list(self):
        data_list = [[(1, 2), [1, 0.05]],
                     [3, [2, 0.05]],
                     [4, [3, 0.05]]]
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=data_list)
        repr(model_link)

    # Create a GaussianLink3D object with mod_data file with double errors
    def test_double_errors(self):
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=model_data_double)
        repr(model_link)

    # Create a GaussianLink3D object with different data_idx and data_spc
    # Also include a data_idx sequence
    def test_data_input_types(self):
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=model_data_types)
        repr(model_link)
