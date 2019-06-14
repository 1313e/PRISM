# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from e13tools.core import InputError, ShapeError
from mpi4pyd import MPI
import numpy as np
import pytest

# PRISM imports
from prism import __version__
from prism._internal import RequestWarning
from prism.modellink import ModelLink
from prism.modellink.tests.modellink import GaussianLink2D, GaussianLink3D


# Save the path to this directory
dirpath = path.dirname(__file__)

# Save paths to various files
model_data_single = path.join(dirpath, 'data/data_gaussian_single.txt')
model_data_double = path.join(dirpath, 'data/data_gaussian_double.txt')
model_data_types = path.join(dirpath, 'data/data_gaussian_types.txt')
model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')
model_parameters_invalid_rng = path.join(dirpath,
                                         'data/parameters_invalid_rng.txt')
model_parameters_invalid_est = path.join(dirpath,
                                         'data/parameters_invalid_est.txt')
model_parameters_outside_est = path.join(dirpath,
                                         'data/parameters_outside_est.txt')


# %% CUSTOM CLASSES
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

    # Try to create a GaussianLink2D object with invalid parameter ranges
    def test_invalid_model_par_rng(self):
        with pytest.raises(ValueError):
            GaussianLink2D(model_parameters=model_parameters_invalid_rng)

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


# Pytest for ModelLink class checking functions
class Test_ModelLink_Check_Methods(object):
    # Initialize a GaussianLink2D object to use for testing
    @pytest.fixture(scope='class')
    def modellink_obj(self):
        return(GaussianLink2D())

    # Test a valid sam_set
    def test_valid_sam_set(self, modellink_obj):
        modellink_obj._check_sam_set(np.ones([1, modellink_obj._n_par]),
                                     'sam_set')

    # Test a valid mod_set
    def test_valid_mod_set(self, modellink_obj):
        modellink_obj._check_mod_set(np.ones([1, modellink_obj._n_data]),
                                     'mod_set')

    # Test a valid single md_var
    def test_valid_single_md_var(self, modellink_obj):
        modellink_obj._check_md_var(np.ones([modellink_obj._n_data]),
                                    'md_var_single')

    # Test a valid dual md_var
    def test_valid_dual_md_var(self, modellink_obj):
        modellink_obj._check_md_var(np.ones([modellink_obj._n_data, 2]),
                                    'md_var_dual')

    # Test a valid sam_set dict
    def test_valid_sam_set_dict(self, modellink_obj):
        sam_dict = dict(zip(modellink_obj._par_name,
                            np.ones([1, modellink_obj._n_par]).T))
        modellink_obj._check_sam_set(sam_dict, 'sam_dict')

    # Test a valid mod_set dict
    def test_valid_mod_set_dict(self, modellink_obj):
        mod_dict = dict(zip(modellink_obj._data_idx,
                            np.ones([1, modellink_obj._n_data]).T))
        modellink_obj._check_mod_set(mod_dict, 'mod_dict')

    # Test a valid md_var dict
    def test_valid_md_var_dict(self, modellink_obj):
        md_var_dict = dict(zip(modellink_obj._data_idx,
                               np.ones([modellink_obj._n_data])))
        modellink_obj._check_md_var(md_var_dict, 'md_var_dict')

    # Test a 3D sam_set
    def test_3D_sam_set(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_sam_set(np.zeros([1, 1, 1]), 'sam_set')

    # Test a 3D mod_set
    def test_3D_mod_set(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_mod_set(np.zeros([1, 1, 1]), 'mod_set')

    # Test a 3D md_var
    def test_3D_md_var(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_md_var(np.zeros([1, 1, 1]), 'md_var')

    # Test a sam_set with wrong n_par
    def test_sam_set_n_par(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_sam_set(np.ones([1, modellink_obj._n_par+1]),
                                         'sam_set')

    # Test a mod_set with wrong n_data
    def test_mod_set_n_data(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_mod_set(np.ones([1, modellink_obj._n_data+1]),
                                         'mod_set')

    # Test a sam_set outside of par_space
    def test_sam_set_par_space(self, modellink_obj):
        with pytest.raises(ValueError):
            modellink_obj._check_sam_set(np.zeros([1, modellink_obj._n_par]),
                                         'sam_set')

    # Test an md_var with wrong n_Data
    def test_md_var_n_data(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_md_var(np.ones([1]), 'md_var')

    # Test a md_var with three values
    def test_md_var_shape(self, modellink_obj):
        with pytest.raises(ShapeError):
            modellink_obj._check_md_var(np.ones([modellink_obj._n_data, 3]),
                                        'md_var')


# Pytest for ModelLink class versatility
class Test_ModelLink_Versatility(object):
    # Create a GaussianLink2D object with externally defined mod_par dict
    # Also use the two options of leaving par_est out
    def test_ext_mod_par_dict(self):
        par_dict = {'A': [1, 5, None],
                    'B': [1, 3]}
        modellink_obj = GaussianLink2D(model_parameters=par_dict)
        assert modellink_obj._par_est == [None, None]
        repr(modellink_obj)

    # Create a GaussianLink2D object with externally defined mod_par list
    def test_ext_mod_par_list(self):
        par_list = [['A', [1, 5, 2.5]],
                    ['B', [1, 3, 2]]]
        modellink_obj = GaussianLink2D(model_parameters=par_list)
        repr(modellink_obj)

    # Create a GaussianLink2D object and test the value conversions
    def test_convert_to_space(self):
        modellink_obj = GaussianLink2D()
        assert np.isclose(modellink_obj._to_par_space([0.2, 0.7]),
                          [1.8, 2.4]).all()
        assert np.isclose(modellink_obj._to_unit_space([1.8, 2.4]),
                          [0.2, 0.7]).all()

    # Create a GaussianLink3D object with externally defined mod_data dict
    def test_ext_mod_data_dict(self):
        data_dict = {(1, 2): [1, 0.05],
                     3: [2, 0.05],
                     4: [3, 0.05]}
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=data_dict)
        repr(modellink_obj)

    # Create a GaussianLink3D object with externally defined mod_data list
    def test_ext_mod_data_list(self):
        data_list = [[(1, 2), [1, 0.05]],
                     [3, [2, 0.05]],
                     [4, [3, 0.05]]]
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=data_list)
        repr(modellink_obj)

    # Create a GaussianLink3D object with mod_data file with double errors
    def test_double_errors(self):
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data_double)
        repr(modellink_obj)

    # Create a GaussianLink3D object with different data_idx and data_spc
    # Also include a data_idx sequence
    def test_data_input_types(self):
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data_types)
        repr(modellink_obj)

    # Create a GaussianLink2D object and try to change its name
    def test_change_name(self):
        modellink_obj = GaussianLink2D()
        with pytest.warns(RequestWarning):
            modellink_obj.name = 'test'
        modellink_obj.name = modellink_obj._init_name

    # Create a GaussianLink2D object and try to change its call_type
    def test_change_call_type(self):
        modellink_obj = GaussianLink2D()
        with pytest.warns(RequestWarning):
            modellink_obj.call_type = 'single'
            modellink_obj.call_type = 'multi'
            modellink_obj.call_type = 'hybrid'
        modellink_obj.call_type = modellink_obj._init_call_type

    # Create a GaussianLink2D object and try to change its MPI_call
    def test_change_MPI_call(self):
        modellink_obj = GaussianLink2D()
        with pytest.warns(RequestWarning):
            modellink_obj.MPI_call = (modellink_obj.MPI_call+1) % 2
        modellink_obj.MPI_call = modellink_obj._init_MPI_call


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
            mod_set = modellink_obj.call_model(emul_i=emul_i,
                                               par_set=par_set,
                                               data_idx=data_idx)

            # Try loading the backup data
            filename, data = modellink_obj._read_backup(emul_i)

            # Check that all data is correct
            assert data['emul_i'] == emul_i
            assert data['prism_version'] == __version__
            assert data['par_set'] == par_set
            assert data['data_idx'] == data_idx
            assert data['args'] == (mod_set,)
            assert data['kwargs'] == {'mod_set': mod_set}

            # Try loading the backup again, this time using the suffix
            suffix = filename.partition('%s(' % (modellink_obj._name))[2][:-6]
            filename, data = modellink_obj._read_backup(emul_i, suffix=suffix)

            # Check that all data is correct
            assert data['emul_i'] == emul_i
            assert data['prism_version'] == __version__
            assert data['par_set'] == par_set
            assert data['data_idx'] == data_idx
            assert data['args'] == (mod_set,)
            assert data['kwargs'] == {'mod_set': mod_set}

    # Test the backup system while no backup exists
    def test_load_no_backup(self, modellink_obj):
        with pytest.raises(OSError):
            modellink_obj._read_backup(0)

    # Test the backup system with incorrect suffix
    def test_load_incorrect_suffix(self, modellink_obj):
        with pytest.raises(OSError):
            modellink_obj._read_backup(0, suffix='test')

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
