# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from e13tools.core import InputError
import numpy as np
import pytest

# PRISM imports
from prism._internal import RequestWarning, MPI
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


# Custom ModelLink class that uses the backup system
class BackupModelLink(ModelLink):
    def call_model(self, data_idx, *args, **kwargs):
        mod_set = [1]*len(data_idx)
        self._make_backup(mod_set, mod_set=mod_set)
        return(mod_set)

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


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


# Pytest for testing the backup system for call_model
@pytest.mark.filterwarnings("ignore::prism._internal.FeatureWarning")
class Test_backup_system(object):
    # Create a universal ModelLink object to use in this class
    @pytest.fixture(scope='class')
    def modellink_obj(self):
        return(BackupModelLink(model_parameters=model_parameters_3D,
                               model_data=model_data_single))

    # Test the backup system the correct way
    def test_default(self, modellink_obj):
        # Set input arguments
        emul_i = 1
        par_set = dict(zip(['A', 'B', 'C'], [1, 1, 1]))
        data_idx = [1, 'A', (1, 2)]

        # Call the model to create the backup on the controller
        if not MPI.COMM_WORLD.Get_rank():
            modellink_obj.call_model(emul_i=emul_i,
                                     par_set=par_set,
                                     data_idx=data_idx)

        # Manually assign mod_set, as workers would not have access to it
        mod_set = [1, 1, 1]

        # Try loading the backup data
        data = modellink_obj._read_backup(emul_i)

        # Check that all data is correct
        assert data['emul_i'] == emul_i
        assert data['par_set'] == par_set
        assert data['data_idx'] == data_idx
        assert data['args'] == (mod_set,)
        assert data['kwargs'] == {'mod_set': mod_set}

    # Test the backup system using no args or kwargs
    def test_no_args_kwargs(self, modellink_obj):
        with pytest.warns(RequestWarning):
            assert modellink_obj._make_backup() is None

    # Test the backup system calling it outside the call_model method
    def test_no_call_model(self, modellink_obj):
        with pytest.warns(RequestWarning):
            assert modellink_obj._make_backup(0) is None

    # Test the backup system calling call_model with args instead of kwargs
    def test_call_model_args(self, modellink_obj):
        # Set input arguments
        emul_i = 2
        par_set = dict(zip(['A', 'B', 'C'], [1, 1, 1]))
        data_idx = [1, 'A', (1, 2)]

        # Try using call_model providing input as args
        with pytest.warns(RequestWarning):
            modellink_obj.call_model(data_idx, emul_i, par_set)

        # Make sure no backup file was created
        with pytest.raises(OSError):
            modellink_obj._read_backup(emul_i)


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
