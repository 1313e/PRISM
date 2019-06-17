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
from prism.modellink import ModelLink
from prism.modellink.utils import test_subclass as _test_subclass


# Save the path to this directory
dirpath = path.dirname(__file__)

# Save paths to various files
model_data_single = path.join(dirpath, 'data/data_gaussian_single.txt')
model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')


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

    def call_model(self, emul_i, par_set, data_idx):
        super().call_model(emul_i, par_set, data_idx)

    def get_md_var(self, emul_i, par_set, data_idx):
        super().get_md_var(emul_i, par_set, data_idx)


# Custom ModelLink class that does not accept the correct call_model arguments
class WrongCallModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        self.MPI_call = True
        super().__init__(*args, **kwargs)

    def call_model(self, emul_i):
        pass

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class that accepts too many call_model arguments
class ManyCallModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        self.MPI_call = True
        super().__init__(*args, **kwargs)

    def call_model(self, emul_i, par_set, data_idx, test):
        pass

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class with no get_md_var()
class NoMdVarModelLink(ModelLink):
    def call_model(self, emul_i, par_set, data_idx):
        return([1]*len(data_idx))

    def get_md_var(self, emul_i, par_set, data_idx):
        super().get_md_var(emul_i, par_set, data_idx)


# Custom ModelLink class that does not accept the correct get_md_var arguments
class WrongMdVarModelLink(ModelLink):
    def call_model(self, emul_i, par_set, data_idx):
        return([1]*len(data_idx))

    def get_md_var(self, emul_i):
        pass


# Custom ModelLink class that accepts too many get_md_var arguments
class ManyMdVarModelLink(ModelLink):
    def call_model(self, emul_i, par_set, data_idx):
        return([1]*len(data_idx))

    def get_md_var(self, emul_i, par_set, data_idx, test):
        pass


# %% PYTEST CLASSES AND FUNCTIONS
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
        with pytest.raises(InputError):
            _test_subclass(WrongCallModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)

    # Test a ModelLink subclass that has too many call_model arguments
    def test_too_many_call_ModelLink(self):
        with pytest.raises(InputError):
            _test_subclass(ManyCallModelLink,
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
        with pytest.raises(InputError):
            _test_subclass(WrongMdVarModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)

    # Test a ModelLink subclass that has too many get_md_var() arguments
    def test_too_many_md_var_ModelLink(self):
        with pytest.raises(InputError):
            _test_subclass(ManyMdVarModelLink,
                           model_parameters=model_parameters_3D,
                           model_data=model_data_single)
