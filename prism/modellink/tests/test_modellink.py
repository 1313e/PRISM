# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from prism._internal import RequestWarning
from prism.modellink import ModelLink, test_subclass as _test_subclass
from prism.modellink.tests.modellink import GaussianLink2D, GaussianLink3D

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save paths to various files
model_data_single = path.join(dirpath, 'data/data_gaussian_single.txt')
model_data_double = path.join(dirpath, 'data/data_gaussian_double.txt')
model_data_types = path.join(dirpath, 'data/data_gaussian_types.txt')
model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')
model_parameters_invalid_est = path.join(dirpath,
                                         'data/parameters_invalid_est.txt')
model_parameters_outside_est = path.join(dirpath,
                                         'data/parameters_outside_est.txt')


# %% CUSTOM CLASSES
# Custom invalid ModelLink class
class InvalidModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        pass


# Custom improper ModelLink class
class ImproperModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        pass

    def call_model(self, *args, **kwargs):
        super().call_model(*args, **kwargs)

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class with no call_model
class NoCallModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        self.MPI_call = True
        super().__init__(*args, **kwargs)

    def call_model(self, *args, **kwargs):
        super().call_model(*args, **kwargs)

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class that does not accept the correct call_model arguments
class WrongCallModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        self.MPI_call = True
        super().__init__(*args, **kwargs)

    def call_model(self, emul_i):
        pass

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class with no get_md_var()
class NoMdVarModelLink(ModelLink):
    def call_model(self, data_idx, *args, **kwargs):
        return([1]*len(data_idx))

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class that does not accept the correct get_md_var arguments
class WrongMdVarModelLink(ModelLink):
    def call_model(self, data_idx, *args, **kwargs):
        return([1]*len(data_idx))

    def get_md_var(self, emul_i):
        pass


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

    # Try to create a GaussianLink2D object with an estimate outside par_rng
    def test_outside_model_par_est(self):
        with pytest.raises(ValueError):
            GaussianLink2D(model_parameters=model_parameters_outside_est)

    # Try to create a GaussianLink2D object using different types of model call
    # Also include an invalid call type
    def test_invalid_call_type(self):
        modellink_obj = GaussianLink2D()
        modellink_obj.call_type = 'single'
        modellink_obj.call_type = 'multi'
        modellink_obj.call_type = 'hybrid'
        with pytest.raises(ValueError):
            modellink_obj.call_type = 'invalid'

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
        assert model_link._par_est == [None, None]
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


# Pytest for test_subclass function
class Test_test_subclass(object):
    # Test not a class
    def test_no_class(self):
        with pytest.raises(InputError):
            _test_subclass(np.array)

    # Test a ModelLink subclass that cannot be initialized
    def test_invalid_ModelLink(self):
        with pytest.raises(InputError):
            _test_subclass(InvalidModelLink)

    # Test not a ModelLink subclass
    def test_no_ModelLink_subclass(self):
        with pytest.raises(TypeError):
            _test_subclass(ValueError)

    # Test an improper ModelLink
    def test_improper_ModelLink(self):
        with pytest.raises(InputError):
            _test_subclass(ImproperModelLink)

    # Test a ModelLink subclass with no custom call_model()-method
    def test_no_call_ModelLink(self):
        with pytest.raises(NotImplementedError):
            _test_subclass(NoCallModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)

    # Test a ModelLink subclass that has an invalid call_model()-method
    def test_invalid_call_ModelLink(self):
        with pytest.raises(TypeError):
            _test_subclass(WrongCallModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)

    # Test a ModelLink subclass with no custom get_md_var()-method
    def test_no_md_var_ModelLink(self):
        with pytest.warns(RequestWarning):
            _test_subclass(NoMdVarModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)

    # Test a ModelLink subclass that has an invalid get_md_var()-method
    def test_invalid_md_var_ModelLink(self):
        with pytest.raises(TypeError):
            _test_subclass(WrongMdVarModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)
