# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import os
from os import path
import shutil

# Package imports
from e13tools import InputError, ShapeError
from e13tools.sampling import lhd
from e13tools.utils import check_instance
import h5py
from mpi4pyd import MPI
import numpy as np
from py.path import local
import pytest
import pytest_mpl
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism import Pipeline
from prism._internal import RequestError, RequestWarning
from prism._projection import Projection
from prism.emulator import Emulator
from prism.modellink import ModelLink
from prism.modellink.tests.modellink import GaussianLink2D, GaussianLink3D


# Save the path to this directory
dirpath = path.dirname(__file__)

# Set the random seed of NumPy
np.random.seed(2)

# Set the current working directory to the temporary directory
local.get_temproot().chdir()

# Ignore any RequestWarnings
pytestmark = pytest.mark.filterwarnings(
    "ignore::prism._internal.RequestWarning")

# Save paths to various files
model_data_single = path.join(dirpath, 'data/data_gaussian_single.txt')
prism_file_default = path.join(dirpath, 'data/prism_default.txt')
prism_file_impl = path.join(dirpath, 'data/prism_impl.txt')
prism_file_n_cross_val = path.join(dirpath, 'data/prism_n_cross_val.txt')
model_parameters_2D = path.join(dirpath, 'data/parameters_gaussian_2D.txt')
model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')


# %% CUSTOM CLASSES
# Custom invalid Emulator class
class InvalidEmulator(Emulator):
    _emul_type = 'invalid'

    def __init__(self, *args, **kwargs):
        pass


# Custom Emulator class
class CustomEmulator(Emulator):
    _emul_type = 'custom'


# Custom improper ModelLink class
class ImproperModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        pass

    def call_model(self, *args, **kwargs):
        super().call_model(*args, **kwargs)

    def get_md_var(self, *args, **kwargs):
        super().get_md_var(*args, **kwargs)


# Custom ModelLink class with double md_var values
class DoubleMdVarModelLink(ModelLink):
    def call_model(self, data_idx, *args, **kwargs):
        par = kwargs['par_set']
        return(np.array(data_idx)*(par['A']+0.00000001*par['B']*par['C']**2))

    def get_md_var(self, data_idx, *args, **kwargs):
        return([[1, 1]]*len(data_idx))


# Custom ModelLink class
class CustomModelLink(ModelLink):
    def call_model(self, data_idx, *args, **kwargs):
        return(np.random.rand(len(data_idx)))

    def get_md_var(self, data_idx, *args, **kwargs):
        return([[1, 1]]*len(data_idx))


# Custom List class that reports wrong length
class InvalidLen2List(list):
    def __len__(self):
        return(2)


# Custom Dict class that returns wrong items
class InvalidDict(dict):
    def __getitem__(self, y):
        super().__getitem__(1)


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for standard Pipeline class (+Emulator, +Projection) for 2D model
class Test_Pipeline_Gaussian2D(object):
    # Test a 2D Gaussian model
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test2D')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Check if representation can be called
    def test_repr(self, pipe):
        pipe2 = eval(repr(pipe))
        assert pipe2._hdf5_file == pipe._hdf5_file

    # Check if proj_res and proj_depth can be called
    def test_proj_par_props(self, pipe):
        assert pipe.proj_res is None
        assert pipe.proj_depth is None

    # Check if first iteration can be constructed
    def test_construct(self, pipe):
        pipe.construct(1, analyze=0)

    # Check if first iteration can be reconstructed unforced
    def test_reconstruct_no_force(self, pipe):
        pipe.construct(1, analyze=0, force=0)

    # Check if first iteration can be projected before analysis
    def test_project_pre_anal(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project(proj_par=(0), figure=False)
            pipe.project(proj_par=(0), figure=True, proj_type='both',
                         show_cuts=True)
            pipe.project(proj_par=(1), figure=True, align='row', smooth=True)

    # Check if first iteration can be projected again (unforced)
    def test_reproject_unforced(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project()

    # Check if pipeline data can be reloaded before analysis
    def test_reload(self, pipe):
        pipe._load_data()

    # Check if first iteration can be analyzed
    def test_analyze(self, pipe):
        pipe.analyze()

    # Check if first iteration can be reprojected (forced)
    def test_reproject_forced(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project(force=True, smooth=True)

    # Check if figure data can be received
    def test_project_fig_data(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project(smooth=True, figure=False)

    # Check if details overview of first iteration can be given
    def test_details(self, pipe):
        pipe.details()

    # Check if first iteration can be reconstructed forced
    def test_reconstruct_force(self, pipe):
        pipe.construct(1, analyze=0, force=1)

    # Check if entire second iteration can be created
    def test_run(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.run(2)

    # Try to access all Pipeline properties
    def test_access_pipe_props(self, pipe):
        check_instance(pipe, Pipeline)
        pipe._comm.__dir__()

    # Try to access all Emulator properties
    def test_access_emul_props(self, pipe):
        check_instance(pipe._emulator, Emulator)

    # Try to reload and reanalyze the entire Pipeline using different impl_cut
    def test_reload_reanalyze_pipeline(self, pipe):
        pipe_reload = Pipeline(pipe._modellink, root_dir=pipe._root_dir,
                               working_dir=pipe._working_dir)
        assert pipe_reload._working_dir == pipe._working_dir
        pipe_reload.analyze(impl_cut=[0.001, 0.001, 0.001])

    # Check if second iteration can be reconstructed
    def test_reconstruct_iteration2(self, pipe):
        pipe.construct(2, analyze=0, force=1)

    # Check if first iteration can be evaluated for a single parameter set
    def test_evaluate_1D(self, pipe):
        pipe.evaluate(pipe._emulator._sam_set[2][0], 1)

    # Check if first iteration can be evaluated with a single parameter dict
    def test_evaluate_dict_1D(self, pipe):
        pipe.evaluate({'A': 2.5, 'B': 2}, 1)

    # Check if first iteration can be evaluated for more than one parameter set
    def test_evaluate_nD(self, pipe):
        pipe.evaluate(pipe._emulator._sam_set[2], 1)

    # Check if first iteration can be evaluated for more than one par_dict
    def test_evaluate_dict_nD(self, pipe):
        pipe.evaluate({'A': [2.5], 'B': [2]}, 1)

    # Check if representation can be called
    def test_repr2(self, pipe):
        pipe2 = eval(repr(pipe))
        assert pipe2._hdf5_file == pipe._hdf5_file

    # Check if the logging system can be disabled
    def test_disable_logging(self, pipe):
        pipe.do_logging = 0
        pipe.do_logging = 0
        pipe.do_logging = 1

    # Check if the workers can start listening
    def test_worker_mode(self, pipe):
        with pipe.worker_mode:
            if pipe._is_controller:
                pipe._make_call(np.array, [1])
                assert pipe._make_call('_emulator._get_emul_i', 1, 0) == 1
                assert pipe._make_call('_evaluate_sam_set', 1,
                                       np.array([[2.5, 2]]),
                                       ("", "", "", "", "")) is None


# Pytest for standard Pipeline class (+Emulator, +Projection) for 3D model
class Test_Pipeline_Gaussian3D(object):
    # Test a 3D Gaussian model
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test3D')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data_single)
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Check if representation can be called
    def test_repr(self, pipe):
        pipe2 = eval(repr(pipe))
        assert pipe2._hdf5_file == pipe._hdf5_file

    # Check if first iteration can be constructed
    def test_construct(self, pipe):
        pipe.construct(1, analyze=0)

    # Check if first iteration can be analyzed
    def test_analyze(self, pipe):
        pipe.analyze()

    # Check if first iteration can be evaluated
    def test_evaluate(self, pipe):
        pipe.evaluate([2.5, 2, 1])

    # Check if first iteration can be projected
    def test_project(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project(1, (0, 1), align='row', smooth=True, proj_type='3D',
                         fig_kwargs={'dpi': 10})
            pipe.project(1, (0, 1), proj_type='3D', fig_kwargs={'dpi': 10},
                         figure=False)
            if pipe._is_controller:
                os.remove(pipe._Projection__get_fig_path((0, 1))[1])
            pipe._comm.Barrier()
            pipe.project(1, (0, 1), align='col', fig_kwargs={'dpi': 10})

    # Check if details overview of first iteration can be given
    def test_details(self, pipe):
        pipe.details()

    # Try to access all Pipeline properties
    def test_access_pipe_props(self, pipe):
        check_instance(pipe, Pipeline)

    # Try to access all Emulator properties
    def test_access_emul_props(self, pipe):
        check_instance(pipe._emulator, Emulator)

    # Check if representation can be called
    def test_repr2(self, pipe):
        pipe2 = eval(repr(pipe))
        assert pipe2._hdf5_file == pipe._hdf5_file


# Pytest for standard Pipeline class for 3D model with a single data point
class Test_Pipeline_Gaussian3D_1_data(object):
    # Test a 3D Gaussian model
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test3D')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data={2: [2, 0.05]})
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Check if representation can be called
    def test_repr(self, pipe):
        pipe2 = eval(repr(pipe))
        assert pipe2._hdf5_file == pipe._hdf5_file

    # Check if first iteration can be constructed
    def test_construct(self, pipe):
        pipe.construct(1, analyze=0)

    # Check if first iteration can be analyzed
    def test_analyze(self, pipe):
        pipe.analyze()

    # Check if first iteration can be evaluated
    def test_evaluate(self, pipe):
        pipe.evaluate([2.5, 2, 1])

    # Check if first iteration can be projected
    def test_project(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project(1, (0, 1), align='row', smooth=True, proj_type='3D',
                         fig_kwargs={'dpi': 10})
            pipe.project(1, (0, 1), proj_type='3D', fig_kwargs={'dpi': 10},
                         figure=False)
            if pipe._is_controller:
                os.remove(pipe._Projection__get_fig_path((0, 1))[1])
            pipe._comm.Barrier()
            pipe.project(1, (0, 1), align='col', fig_kwargs={'dpi': 10})

    # Check if details overview of first iteration can be given
    def test_details(self, pipe):
        pipe.details()

    # Try to access all Pipeline properties
    def test_access_pipe_props(self, pipe):
        check_instance(pipe, Pipeline)

    # Try to access all Emulator properties
    def test_access_emul_props(self, pipe):
        check_instance(pipe._emulator, Emulator)

    # Check if representation can be called
    def test_repr2(self, pipe):
        pipe2 = eval(repr(pipe))
        assert pipe2._hdf5_file == pipe._hdf5_file


# Pytest for Pipeline class exception handling during initialization
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                    reason="Cannot be pytested in MPI")
class Test_Pipeline_Init_Exceptions(object):
    # Create a modellink_obj object used in some test functions
    @pytest.fixture(scope='function')
    def modellink_obj(self):
        return(GaussianLink2D())

    @pytest.fixture(scope='function')
    def root_working_dir(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test_init_exceptions')
        return({'root_dir': path.dirname(tmpdir.strpath),
                'working_dir': path.basename(tmpdir.strpath)})

    # Create a Pipeline object using an invalid Emulator class
    def test_invalid_Emulator(self, root_working_dir, modellink_obj):
        with pytest.raises(InputError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file_default, emul_type=InvalidEmulator)

    # Create a Pipeline object using not an Emulator class
    def test_no_Emulator(self, root_working_dir, modellink_obj):
        with pytest.raises(InputError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file_default, emul_type=Pipeline)

    # Create a Pipeline object using an improper ModelLink object
    def test_improper_ModelLink(self, root_working_dir):
        with pytest.raises(InputError):
            modellink_obj = ImproperModelLink()
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file_default)

    # Create a Pipeline object using not a ModelLink object
    def test_no_ModelLink(self, root_working_dir):
        with pytest.raises(TypeError):
            Pipeline(np.array([1]), **root_working_dir,
                     prism_par=prism_file_default)

    # Create a Pipeline object using alternate values for criterion and
    # pot_active_par. Also include an invalid pot_active_par
    def test_invalid_pot_act_par(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_invalid_pot_act_par.txt')
        with pytest.raises(InputError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file)

    # Create a Pipeline object using alternate values for criterion and
    # pot_active_par. Also include an empty pot_active_par
    def test_empty_pot_act_par(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_empty_pot_act_par.txt')
        with pytest.raises(ValueError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file)

    # Create a Pipeline object using an invalid value for criterion (bool)
    def test_bool_criterion(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_bool_criterion.txt')
        with pytest.raises(TypeError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file)

    # Create a Pipeline object using an invalid string for criterion
    def test_nnormal_criterion(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_invalid_str_criterion.txt')
        with pytest.raises(InputError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file)

    # Create a Pipeline object using an invalid value for pot_active_par (bool)
    def test_bool_pot_act_par(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_bool_pot_act_par.txt')
        with pytest.raises(TypeError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file)

    # Create a Pipeline object using a non_existent PRISM file
    def test_non_existent_prism_file(self, root_working_dir, modellink_obj):
        with pytest.raises(OSError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par='test.txt')

    # Create a Pipeline object using an invalid root dir
    def test_invalid_root_dir(self, tmpdir, modellink_obj):
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(modellink_obj, root_dir=1, working_dir=working_dir,
                     prism_par=prism_file_default)

    # Create a Pipeline object using an invalid working dir
    def test_invalid_working_dir(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(modellink_obj, root_dir=root_dir, working_dir=1.0,
                     prism_par=prism_file_default)

    # Create a Pipeline object using an integer working dir
    def test_int_working_dir(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        with pytest.raises(TypeError):
            Pipeline(modellink_obj, root_dir=root_dir, working_dir=1,
                     prism_par=prism_file_default)

    # Create a Pipeline object using an invalid PRISM file
    def test_invalid_prism_file(self, root_working_dir, modellink_obj):
        with pytest.raises(TypeError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=1)

    # Create a Pipeline object using an invalid prefix
    def test_invalid_prefix(self, root_working_dir, modellink_obj):
        with pytest.raises(TypeError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file_default, prefix=1)

    # Create a Pipeline object using invalid mock data spaces
    def test_invalid_mock_data_spc_predef(self, root_working_dir,
                                          modellink_obj):
        modellink_obj._data_spc = ['A', 'B', 'C']
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file_default)
        with pytest.raises(NotImplementedError):
            pipe._emulator._create_new_emulator()

    # Create a new emulator using invalid mock data spaces
    def test_invalid_mock_data_spc_undef(self, root_working_dir):
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data_single)
        modellink_obj._data_spc = ['A', 'B', 'C']
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file_default)
        with pytest.raises(NotImplementedError):
            pipe._emulator._create_new_emulator()

    # Create a Pipeline object using an empty impl_cut list
    def test_empty_impl_cut(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_empty_impl_cut.txt')
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file)
        with pytest.raises(InputError):
            pipe.construct()

    # Create a Pipeline object using an impl_cut list with only wildcards
    def test_wildcard_impl_cut(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_wildcard_impl_cut.txt')
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file)
        pipe._emulator._n_data_tot.append(modellink_obj._n_data)
        with pytest.raises(ValueError):
            pipe.construct()

    # Create a Pipeline object using an invalid impl_cut list
    def test_invalid_impl_cut(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_invalid_impl_cut.txt')
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file)
        with pytest.raises(ValueError):
            pipe.construct()

    # Create a new emulator using an invalid n_cross_val value
    def test_invalid_n_cross_val(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_invalid_n_cross_val.txt')
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file)
        with pytest.raises(ValueError):
            pipe._emulator._create_new_emulator()

    # Try to load an emulator that was built with a different modellink
    def test_unmatched_ModelLink(self, root_working_dir, modellink_obj):
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file_default)
        pipe.construct(1, analyze=0)
        with pytest.raises(InputError):
            modellink_obj =\
                GaussianLink3D(model_parameters=model_parameters_3D,
                               model_data=model_data_single)
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file_default)

    # Try to use the 'auto' emulation method
    def test_auto_emul_method(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_auto_emul_method.txt')
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file)
        with pytest.raises(NotImplementedError):
            pipe._emulator._create_new_emulator()

    # Try to use an invalid emulation method
    def test_invalid_emul_method(self, root_working_dir, modellink_obj):
        prism_file = path.join(dirpath, 'data/prism_invalid_emul_method.txt')
        pipe = Pipeline(modellink_obj, **root_working_dir,
                        prism_par=prism_file)
        with pytest.raises(ValueError):
            pipe._emulator._create_new_emulator()

    # Try to use an invalid MPI world communicator
    def test_invalid_MPI_comm(self, root_working_dir, modellink_obj):
        with pytest.raises(TypeError):
            Pipeline(modellink_obj, **root_working_dir,
                     prism_par=prism_file_default,
                     comm=object)


# Pytest for Pipeline class user exception handling
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                    reason="Cannot be pytested in MPI")
class Test_Pipeline_User_Exceptions(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test_user_exceptions')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Try using an ext_real_set list with three elements
    def test_three_element_ext_real_set(self, pipe):
        with pytest.raises(ShapeError):
            pipe.construct(1, analyze=0, ext_real_set=[1, 1, 1])

    # Try using an invalid ext_real_set "list"
    def test_invalid_ext_real_set_list(self, pipe):
        with pytest.raises(InputError):
            pipe.construct(1, analyze=0, ext_real_set=InvalidLen2List([1]))

    # Try using an ext_real_set dict with no sam_set
    def test_no_ext_sam_set_dict(self, pipe):
        with pytest.raises(KeyError):
            pipe.construct(1, analyze=0, ext_real_set={'mod_set': 1})

    # Try using an ext_real_set dict with no mod_set
    def test_no_ext_mod_set_dict(self, pipe):
        with pytest.raises(KeyError):
            pipe.construct(1, analyze=0, ext_real_set={'sam_set': 1})

    # Try using an invalid ext_real_set "dict"
    def test_invalid_ext_real_set_dict(self, pipe):
        with pytest.raises(InputError):
            pipe.construct(1, analyze=0,
                           ext_real_set=InvalidDict({'sam_set': 1,
                                                     'mod_set': 1}))

    # Try using an invalid ext_real_set (tuple)
    def test_invalid_ext_real_set(self, pipe):
        with pytest.raises(InputError):
            pipe.construct(1, analyze=0, ext_real_set=(1))

    # Try using an ext_real_set with inconsistent n_sam
    def test_ext_real_set_n_sam(self, pipe):
        with pytest.raises(ShapeError):
            pipe.construct(1, analyze=0, ext_real_set=[
                np.ones([2, pipe._modellink._n_par]),
                np.ones([1, pipe._modellink._n_data])])

    # Try analyzing the emulator with an emul_type other than 'default'
    def test_non_default_emul_type_analyze(self, pipe):
        pipe.construct(1, analyze=0)
        pipe._emulator._emul_type = 'non_default'
        with pytest.raises(NotImplementedError):
            pipe.analyze()

    # Try calling details for an emulator with emul_type other than 'default'
    def test_non_default_emul_type_details(self, pipe):
        with pytest.raises(NotImplementedError):
            pipe.details()
        pipe._emulator._emul_type = 'default'

    # Try evaluating an 3D sam_set
    def test_3D_evaluate(self, pipe):
        with pytest.raises(ShapeError):
            pipe.evaluate([[[2.5, 2]]])

    # Try evaluating a sam_set with wrong number of parameters
    def test_invalid_evaluate(self, pipe):
        with pytest.raises(ShapeError):
            pipe.evaluate([2.5, 2, 1])

    # Try to call project with incorrect proj_type parameter
    def test_invalid_proj_type_val(self, pipe):
        pipe._modellink._n_par = 3
        with pytest.raises(ValueError):
            pipe.project(proj_type='test')
        pipe._modellink._n_par = 2

    # Try to call project with incorrect align parameter
    def test_invalid_align_val(self, pipe):
        with pytest.raises(ValueError):
            pipe.project(align='test')

    # Try to call project with no dict as fig_kwargs
    def test_no_fig_kwargs_dict(self, pipe):
        with pytest.raises(TypeError):
            pipe.project(1, (0, 1), fig_kwargs=())

    # Try to call project with an invalid impl_kwargs dict
    def test_invalid_impl_kwargs_dict(self, pipe):
        with pytest.raises(InputError):
            pipe.project(1, (0, 1), fig_kwargs={'nrows': 1},
                         impl_kwargs_3D={'cmap': 1})

    # Try to call project with an invalid los_kwargs dict
    def test_invalid_los_kwargs_dict(self, pipe):
        with pytest.raises(InputError):
            pipe.project(1, (0, 1), impl_kwargs_2D={'x': 1},
                         impl_kwargs_3D={'x': 1}, los_kwargs_3D={'cmap': 1})

    # Try to load an emulator with invalid emulator iteration groups
    def test_invalid_iteration_groups(self, pipe):
        if pipe._is_controller:
            with h5py.File(pipe._hdf5_file, 'r+') as file:
                file.create_group('test')
        pipe._comm.Barrier()
        with pytest.raises(InputError):
            modellink_obj = GaussianLink2D()
            pipe._emulator._load_emulator(modellink_obj)
        if pipe._is_controller:
            with h5py.File(pipe._hdf5_file, 'r+') as file:
                del file['test']


# Pytest for Pipeline class request exception handling
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                    reason="Cannot be pytested in MPI")
class Test_Pipeline_Request_Exceptions(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe_impl(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test_request_exceptions_impl')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_impl))

    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe_n_cross_val(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test_request_exceptions_n_cross_val')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir,
                        prism_par=prism_file_n_cross_val))

    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe_default(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test_request_exceptions_default')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Try to construct an iteration that cannot be constructed
    def test_invalid_construction(self, pipe_default):
        with pytest.raises(RequestError):
            pipe_default.construct(2)

    # Try to call an iteration that cannot be used
    def test_invalid_call(self, pipe_default):
        with pytest.raises(RequestError):
            pipe_default(2)

    # Try to set the impl_cut while no emulator exists
    def test_set_impl_cut_no_emul(self, pipe_default):
        with pytest.raises(RequestError):
            pipe_default.impl_cut = [1]

    # Try to analyze iteration 1 while no emulator exists
    def test_invalid_analyze(self, pipe_default):
        with pytest.raises(RequestError):
            pipe_default.analyze()

    # Try to analyze iteration 1 while iteration 2 is being constructed
    def test_invalid_analyze2(self, pipe_default):
        pipe_default.construct(1)
        pipe_default.construct(2)
        pipe_default._emulator._ccheck[2].append('active_par')
        pipe_default._emulator._emul_i = 1
        with pytest.raises(RequestError):
            pipe_default.analyze()
        pipe_default._emulator._ccheck[2].remove('active_par')
        pipe_default._emulator._emul_i = 2

    # Try to set the impl_cut while the iteration has been analyzed already
    def test_invalid_set_impl_cut(self, pipe_default):
        with pytest.raises(RequestError):
            pipe_default.impl_cut = [1]

    # Try to call an iteration that does not exist
    def test_invalid_iteration(self, pipe_default):
        with pytest.raises(RequestError):
            pipe_default._emulator._get_emul_i(3, True)

    # Try to make projection figures with no parameters
    def test_no_par_project(self, pipe_default):
        pipe_default._emulator._active_par[1][0] = 1
        with pytest.raises(RequestError):
            pipe_default.project(1, (0))
        pipe_default._emulator._active_par[1][0] = 0

    # Try to reconstruct entire emulator requesting no mock data
    def test_no_mock_reconstruct(self, pipe_default):
        prism_file = path.join(dirpath, 'data/prism_no_mock.txt')
        root_dir = pipe_default._root_dir
        working_dir = pipe_default._working_dir
        modellink_obj = GaussianLink2D()
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        with pytest.raises(RequestError):
            pipe.construct(1, force=True)
        modellink_obj = GaussianLink2D()
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(1, force=True)

    # Try to call an iteration that cannot be finished
    def test_break_call(self, pipe_impl):
        with pytest.raises(RequestError), pytest.warns(RequestWarning):
            pipe_impl(1)

    # Try to construct an iteration with no plausible regions
    def test_impl_construction(self, pipe_impl):
        with pytest.raises(RequestError):
            pipe_impl.construct(2)

    # Try to prepare an iteration that cannot be prepared
    def test_prepare_invalid_iteration(self, pipe_impl):
        with pytest.raises(RequestError):
            pipe_impl._emulator._prepare_new_iteration(3)

    # Try to load an iteration that does not exist
    def test_load_invalid_iteration(self, pipe_impl):
        with pytest.raises(RequestError):
            pipe_impl._emulator._load_data(3)

    # Try to use an emulator with a different emul_type
    def test_unmatched_emul_type(self, pipe_impl):
        pipe_impl._emulator._emul_type = 'test'
        with pytest.raises(RequestError):
            pipe_impl._emulator._retrieve_parameters()
        pipe_impl._emulator._emul_type = 'default'

    # Try to construct an iteration that has less than n_cross_val samples
    def test_n_cross_val_construction(self, pipe_n_cross_val):
        with pytest.warns(RequestWarning):
            pipe_n_cross_val.construct(1)
        with pytest.raises(RequestError):
            pipe_n_cross_val.construct(2)


# Pytest for Pipeline class internal exception handling
class Test_Internal_Exceptions(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        tmpdir = tmpdir_factory.mktemp('test_internal_exceptions')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Try to save data using the wrong keyword for pipeline
    def test_invalid_pipe_save_data_keyword(self, pipe):
        pipe.construct(1, analyze=0)
        if pipe._is_controller:
            with pytest.raises(ValueError):
                pipe._save_data({'test': []})

    # Try to save data using the wrong keyword for emulator
    def test_invalid_emul_save_data_keyword(self, pipe):
        if pipe._is_controller:
            with pytest.raises(ValueError):
                pipe._emulator._save_data(1, None, {'test': []})

    # Try to save data using the wrong keyword for projection
    def test_invalid_proj_save_data_keyword(self, pipe):
        pipe._Projection__prepare_projections(None, None,
                                              los_kwargs_2D={'x': 1},
                                              los_kwargs_3D={'x': 1})
        if pipe._is_controller:
            with pytest.raises(ValueError):
                pipe._Projection__save_data({'test': []})


# Pytest for trying to initialize a lone Projection class
def test_Projection_init():
    with pytest.raises(RequestError):
        Projection()


# Pytest for Pipeline class initialization versatility
class Test_Pipeline_Init_Versatility(object):
    # Create a modellink_obj object used in some test functions
    @pytest.fixture(scope='function')
    def modellink_obj(self):
        return(GaussianLink2D())

    # Create a Pipeline object using a custom Emulator class
    def test_custom_Emulator(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default,
                        emul_type=CustomEmulator)
        repr(pipe)

    # Create a Pipeline object using custom pot_active_par
    def test_custom_pot_act_par(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        prism_file = path.join(dirpath, 'data/prism_custom_pot_act_par.txt')
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=working_dir,
                 prism_par=prism_file)

    # Create a Pipeline object using no defined paths
    def test_default_paths(self, tmpdir, modellink_obj):
        with tmpdir.as_cwd():
            pipe = Pipeline(modellink_obj)
            repr(pipe)

    # Create a Pipeline object using a non_existent root dir
    def test_non_existent_root_dir(self, tmpdir, modellink_obj):
        root_dir = path.join(tmpdir.strpath, 'root')
        Pipeline(modellink_obj, root_dir=root_dir,
                 prism_par=prism_file_default)

    # Create a Pipeline object using a non_existent root dir
    def test_non_existent_working_dir(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = 'working_dir'
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=working_dir,
                 prism_par=prism_file_default)

    # Create a Pipeline object using a custom prefix
    def test_custom_prefix(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = 'working_dir'
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=working_dir,
                 prefix='test_', prism_par=prism_file_default)

    # Create a Pipeline object using a relative path to a PRISM file
    def test_rel_path_PRISM_file(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        shutil.copy(prism_file_default, root_dir)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par='prism_default.txt')
        repr(pipe)

    # Create a Pipeline object using the prism_file input argument
    @pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                        reason="Cannot be pytested in MPI")
    def test_PRISM_par_file(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        shutil.copy(prism_file_default, root_dir)
        with pytest.warns(FutureWarning):
            pipe = Pipeline(modellink_obj, root_dir=root_dir,
                            working_dir=working_dir,
                            prism_file='prism_default.txt')
        repr(pipe)

    # Create a Pipeline object using a PRISM parameters dict
    def test_PRISM_par_dict(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par={'criterion': 1})
        repr(pipe)

    # Create a Pipeline object using a PRISM parameters array_like
    def test_PRISM_par_array_like(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=[['criterion', 1]])
        repr(pipe)

    # Create a Pipeline object requesting a new working dir two times
    def test_new_working_dir(self, tmpdir, modellink_obj):
        root_dir = tmpdir.strpath
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=True,
                 prism_par=prism_file_default)
        Pipeline(modellink_obj, root_dir=root_dir, working_dir='prism_2',
                 prism_par=prism_file_default)
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=True,
                 prism_par=prism_file_default)

    # Create a Pipeline object loading an existing working dir
    def test_load_existing_working_dir(self, tmpdir, modellink_obj):
        root_dir = path.dirname(tmpdir.strpath)
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=True,
                 prism_par=prism_file_default)
        Pipeline(modellink_obj, root_dir=root_dir, working_dir=False,
                 prism_par=prism_file_default)


# Pytest for Pipeline + ModelLink versatility
class Test_Pipeline_ModelLink_Versatility(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='function')
    def pipe2D(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        with pytest.warns(RequestWarning):
            modellink_obj.call_type = 'single'
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='function')
    def pipe3D(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data_single)
        return(Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default))

    # Test if interrupted construction can be continued
    def test_continue_interrupt(self, pipe2D):
        pipe2D.construct(1, analyze=0)
        if pipe2D._is_controller:
            pipe2D._emulator._ccheck[1].append('active_par')
        pipe2D._emulator._emul_i = 0
        pipe2D.construct(1, analyze=0)

    # Test if interrupted construction (at start) can be continued
    def test_continue_interrupt_start(self, pipe2D):
        pipe2D._emulator._create_new_emulator()
        pipe2D.details()
        pipe2D.construct(1, analyze=0)

    # Test details for different switches
    def test_details_switches(self, pipe2D):
        # Construct first iteration
        pipe2D.construct(1, analyze=0)

        # Emulation methods
        pipe2D._emulator._method = 'gaussian'
        pipe2D.details()
        pipe2D._emulator._method = 'regression'
        pipe2D.details()
        pipe2D._emulator._method = 'full'

        # Missing parameter estimates
        temp0 = pipe2D._modellink._par_est[0]
        pipe2D._modellink._par_est[0] = None
        pipe2D.details()
        temp1 = pipe2D._modellink._par_est[1]
        pipe2D._modellink._par_est[1] = None
        pipe2D.details()
        pipe2D._modellink._par_est[0] = temp0
        pipe2D._modellink._par_est[1] = temp1

        # Inactive parameters
        if pipe2D._is_controller:
            pipe2D._emulator._active_par[1][1] = 0
        pipe2D.details()
        if pipe2D._is_controller:
            pipe2D._emulator._active_par[1][1] = 1

    # Test if mock data takes log10 value spaces into account correctly
    def test_mock_data_spaces_log(self, pipe3D):
        np.random.seed(0)
        pipe3D._modellink._data_spc = ['log10', 'log10', 'log10']
        pipe3D._emulator._create_new_emulator()

    # Test if mock data takes ln value spaces into account correctly
    def test_mock_data_spaces_ln(self, pipe3D):
        np.random.seed(0)
        pipe3D._modellink._data_spc = ['ln', 'ln', 'ln']
        pipe3D._emulator._create_new_emulator()

    # Test if an ext_real_set bigger than n_sam_init can be provided
    def test_ext_real_set_large(self, pipe2D):
        # Create ext_real_set larger than n_sam_init
        sam_set = lhd(pipe2D._n_sam_init*2, pipe2D._modellink._n_par,
                      pipe2D._modellink._par_rng, 'center', pipe2D._criterion)
        sam_dict = sdict(zip(pipe2D._modellink._par_name, sam_set.T))
        mod_set = pipe2D._modellink.call_model(
            1, sam_dict, np.array(pipe2D._modellink._data_idx))

        # Try to construct the iteration
        pipe2D.construct(1, analyze=0, ext_real_set=[sam_set, mod_set])

    # Test if an ext_real_set smaller than n_sam_init can be provided
    def test_ext_real_set_small(self, pipe2D):
        # Create ext_real_set smaller than n_sam_init
        sam_set = lhd(pipe2D._n_sam_init//2, pipe2D._modellink._n_par,
                      pipe2D._modellink._par_rng, 'center', pipe2D._criterion)
        sam_dict = sdict(zip(pipe2D._modellink._par_name, sam_set.T))
        mod_set = pipe2D._modellink.call_model(
            1, sam_dict, np.array(pipe2D._modellink._data_idx))

        # Try to construct the iteration
        pipe2D.construct(1, analyze=0, ext_real_set=[sam_set, mod_set])

    # Test if an ext_real_set dict can be provided
    def test_ext_real_set_dict(self, pipe2D):
        # Create ext_real_set dict
        sam_set = lhd(pipe2D._n_sam_init//2, pipe2D._modellink._n_par,
                      pipe2D._modellink._par_rng, 'center', pipe2D._criterion)
        sam_dict = sdict(zip(pipe2D._modellink._par_name, sam_set.T))
        mod_set = pipe2D._modellink.call_model(
            1, sam_dict, np.array(pipe2D._modellink._data_idx))

        # Try to construct the iteration
        pipe2D.construct(1, analyze=0, ext_real_set={
            'sam_set': sam_set, 'mod_set': mod_set})

    # Test if double md_var values can be returned
    def test_double_md_var(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj =\
            DoubleMdVarModelLink(model_parameters=model_parameters_3D,
                                 model_data=model_data_single)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_impl)
        np.random.seed(0)
        pipe.construct(1, analyze=0)


class Test_Pipeline_Emulator_Versatility(object):
    # Test if emulator can be constructed with only regression
    def test_regression_method(self, tmpdir):
        prism_file = path.join(dirpath, 'data/prism_regression_method.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(1)
        pipe._emulator._load_data(1)

    # Test if emulator can be constructed using chosen mock estimates
    def test_chosen_mock(self, tmpdir):
        prism_file = path.join(dirpath, 'data/prism_chosen_mock.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(1)

    # Test if emulator can be constructed with only gaussian
    def test_gaussian_method(self, tmpdir):
        prism_file = path.join(dirpath, 'data/prism_gaussian_method.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(1)

    # Test if emulator can be constructed with no active analysis
    def test_no_active_par_analysis(self, tmpdir):
        prism_file = path.join(dirpath, 'data/prism_no_act_par_anal.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        modellink_obj = GaussianLink2D()
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(1)

    # Test if different data_idx sequences can be loaded properly
    def test_data_idx_seq(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        model_data = {(1, 'A'): [1, 0.05, 'lin'],
                      'A': [2, 0.05, 'lin'],
                      4.: [3, 0.05, 'lin']}
        modellink_obj = CustomModelLink(model_parameters=model_parameters_3D,
                                        model_data=model_data)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file_default)
        pipe._emulator._create_new_emulator()

    # Test if different model_data can be used in different iterations
    def test_change_data(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        prism_file = path.join(dirpath, 'data/prism_no_mock.txt')
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data_single)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(1)

        # Change data for second iteration
        model_data = {2: [1, 0.05, 'lin'],
                      3: [2, 0.05, 'lin'],
                      4: [3, 0.05, 'lin'],
                      5: [3, 0.05, 'lin']}
        modellink_obj = GaussianLink3D(model_parameters=model_parameters_3D,
                                       model_data=model_data)
        pipe = Pipeline(modellink_obj, root_dir=root_dir,
                        working_dir=working_dir, prism_par=prism_file)
        pipe.construct(2, analyze=0)

        # Change a data value
        pipe._modellink._data_val[0] = 0
        pipe._emulator._emul_i = 2
        pipe._emulator._prepare_new_iteration(2)

        # Change a data error
        pipe._modellink._data_err[0] = [0.10, 0.10]
        pipe._emulator._emul_i = 2
        pipe._emulator._prepare_new_iteration(2)

        # Change a data space
        pipe._modellink._data_spc[0] = 'log10'
        pipe._emulator._emul_i = 2
        pipe._emulator._prepare_new_iteration(2)

        # Change a data identifier
        pipe._modellink._data_idx[0] = 1
        pipe._emulator._emul_i = 2
        pipe._emulator._prepare_new_iteration(2)
