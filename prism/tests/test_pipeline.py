# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import logging
import os
from os import path
import shutil
import sys

# Package imports
from e13tools.core import InputError, ShapeError
from e13tools.sampling import lhd
import numpy as np
import pytest
import pytest_mpl

# PRISM imports
from .modellink.simple_gaussian_link import GaussianLink2D, GaussianLink3D
from prism._internal import RequestError
from prism.emulator import Emulator
from prism.modellink import ModelLink
from prism.pipeline import Pipeline

# Save the path to this directory
dirpath = path.dirname(__file__)

# Set the random seed of NumPy
np.random.seed(2)

# Get lists of all Pipeline and Emulator properties
pipe_props = [prop for prop in dir(Pipeline) if
              isinstance(getattr(Pipeline, prop), property)]
emul_props = [prop for prop in dir(Emulator) if
              isinstance(getattr(Emulator, prop), property)]

# Save paths to various files
model_data_single = path.join(dirpath, 'data/data_gaussian_single.txt')
prism_file_default = path.join(dirpath, 'data/prism_default.txt')
prism_file_impl = path.join(dirpath, 'data/prism_impl.txt')
model_parameters_2D = path.join(dirpath, 'data/parameters_gaussian_2D.txt')
model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')


# Start at line 1513 in the Pipeline class

# %% PYTEST CLASSES AND FUNCTIONS
# TODO: See if it is possible to run some methods in parallel
# Custom invalid Emulator class
class InvalidEmulator(Emulator):
    def __init__(self, *args, **kwargs):
        pass


# Custom invalid ModelLink class
class InvalidModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        pass

    def call_model(self, *args, **kwargs):
        super(InvalidModelLink, self).call_model(*args, **kwargs)

    def get_md_var(self, *args, **kwargs):
        super(InvalidModelLink, self).get_md_var(*args, **kwargs)


# Custom ModelLink class with no call_model
class NoCallModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        super(NoCallModelLink, self).__init__(*args, **kwargs)

    def call_model(self, *args, **kwargs):
        super(NoCallModelLink, self).call_model(*args, **kwargs)

    def get_md_var(self, *args, **kwargs):
        super(NoCallModelLink, self).get_md_var(*args, **kwargs)


# Custom ModelLink class with missing attributes
class NoAttrModelLink(ModelLink):
    def __init__(self, *args, **kwargs):
        super(NoAttrModelLink, self).__init__(*args, **kwargs)
        del self._n_data

    def call_model(self, *args, **kwargs):
        return(1)

    def get_md_var(self, *args, **kwargs):
        super(NoAttrModelLink, self).get_md_var(*args, **kwargs)


# Pytest for standard Pipeline class (+Emulator, +Projection) for 2D model
class Test_Pipeline_Gaussian2D(object):
    # Test a 2D Gaussian model
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        # Obtain paths
        tmpdir = tmpdir_factory.mktemp('test2D')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a GaussianLink2D object
        model_link = GaussianLink2D()

        # Create a Pipeline object
        return(Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_default, emul_type='default'))

    # Check if first iteration can be constructed
    def test_construct(self, pipe):
        pipe.construct(1, 0)

    # Check if data can be reloaded before analysis
    def test_reload_pipeline_data_pre_anal(self, pipe):
        pipe._load_data()

    # Check if first iteration can be analyzed
    def test_analyze(self, pipe):
        pipe.analyze()

    # Check if data can be reloaded after analysis
    def test_reload_pipeline_data_post_anal(self, pipe):
        pipe._load_data()

    # Check if first iteration can be evaluated
    def test_evaluate(self, pipe):
        pipe.evaluate([2.5, 2])

    # Check if first iteration can be projected
    def test_project(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project()

    # Check if details overview of first iteration can be given
    def test_details(self, pipe):
        pipe.details()

    # Check if entire second iteration can be created
    def test_run(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.run(2)

    # Try to access all Pipeline properties
    def test_access_pipe_props(self, pipe):
        for prop in pipe_props:
            getattr(pipe, prop)

    # Try to access all Emulator properties
    def test_access_emul_props(self, pipe):
        for prop in emul_props:
            getattr(pipe._emulator, prop)


# Pytest for standard Pipeline class (+Emulator, +Projection) for 3D model
class Test_Pipeline_Gaussian3D(object):
    # Test a 3D Gaussian model
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        # Obtain paths
        tmpdir = tmpdir_factory.mktemp('test3D')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a GaussianLink3D object
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=model_data_single)

        # Create a Pipeline object
        return(Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_default, emul_type='default'))

    # Check if first iteration can be constructed
    def test_construct(self, pipe):
        pipe.construct(1, 0)

    # Check if data can be reloaded before analysis
    def test_reload_pipeline_data_pre_anal(self, pipe):
        pipe._load_data()

    # Check if first iteration can be analyzed
    def test_analyze(self, pipe):
        pipe.analyze()

    # Check if data can be reloaded after analysis
    def test_reload_pipeline_data_post_anal(self, pipe):
        pipe._load_data()

    # Check if first iteration can be evaluated
    def test_evaluate(self, pipe):
        pipe.evaluate([2.5, 2, 1])

    # Check if first iteration can be projected
    def test_project(self, pipe):
        with pytest_mpl.plugin.switch_backend('Agg'):
            pipe.project(proj_par=(0, 1))

    # Check if details overview of first iteration can be given
    def test_details(self, pipe):
        pipe.details()

    # Try to access all Pipeline properties
    def test_access_pipe_props(self, pipe):
        for prop in pipe_props:
            getattr(pipe, prop)

    # Try to access all Emulator properties
    def test_access_emul_props(self, pipe):
        for prop in emul_props:
            getattr(pipe._emulator, prop)


# Pytest for Pipeline class exception handling during initialization
class Test_Pipeline_Init_Exceptions(object):
    # Create a model_link object used in some test functions
    @pytest.fixture(scope='function')
    def model_link(self):
        return(GaussianLink2D())

    # Create a Pipeline object using an invalid Emulator class
    def test_invalid_Emulator(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type=InvalidEmulator)

    # Create a Pipeline object using not an Emulator class
    def test_no_Emulator(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(RequestError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type=Pipeline)

    # Create a Pipeline object using an invalid ModelLink object
    def test_invalid_ModelLink(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            model_link = InvalidModelLink()
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using a ModelLink object with no call_model
    def test_no_call_ModelLink(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            model_link =\
                NoCallModelLink(model_parameters=model_parameters_3D,
                                model_data=model_data_single)
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using a ModelLink object with missing attr
    def test_no_attr_ModelLink(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            model_link =\
                NoAttrModelLink(model_parameters=model_parameters_3D,
                                model_data=model_data_single)
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using not a ModelLink object
    def test_no_ModelLink(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(TypeError):
            Pipeline(np.array([1]), root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using alternate values for criterion and
    # pot_active_par. Also include an invalid pot_active_par
    def test_invalid_pot_act_par(self, tmpdir, model_link):
        prism_file = path.join(dirpath, 'data/prism_invalid_pot_act_par.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file, emul_type='default')

    # Create a Pipeline object using alternate values for criterion and
    # pot_active_par. Also include an empty pot_active_par
    def test_empty_pot_act_par(self, tmpdir, model_link):
        prism_file = path.join(dirpath, 'data/prism_empty_pot_act_par.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(ValueError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file, emul_type='default')

    # Create a Pipeline object using an invalid value for criterion (bool)
    def test_bool_criterion(self, tmpdir, model_link):
        prism_file = path.join(dirpath, 'data/prism_bool_criterion.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(TypeError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file, emul_type='default')

    # Create a Pipeline object using an invalid value for pot_active_par (bool)
    def test_bool_pot_act_par(self, tmpdir, model_link):
        prism_file = path.join(dirpath, 'data/prism_bool_pot_act_par.txt')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(TypeError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file, emul_type='default')

    # Create a Pipeline object using a non_existent PRISM file
    def test_non_existent_prism_file(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(OSError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file='test.txt', emul_type='default')

    # Create a Pipeline object using an invalid root dir
    def test_invalid_root_dir(self, tmpdir, model_link):
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(model_link, root_dir=1, working_dir=working_dir,
                     prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using an invalid working dir
    def test_invalid_working_dir(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(model_link, root_dir=root_dir, working_dir=1.0,
                     prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using an invalid PRISM file
    def test_invalid_prism_file(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(InputError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=1, emul_type='default')

    # Create a Pipeline object using an invalid prefix
    def test_invalid_prefix(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(TypeError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, prefix=1,
                     emul_type='default')

    # Create a Pipeline object using an invalid hdf5_file path
    def test_invalid_hdf5_file_name(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(TypeError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, hdf5_file=1,
                     emul_type='default')

    # Create a Pipeline object using an invalid hdf5_file extension
    def test_invalid_hdf5_file_extension(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        with pytest.raises(ValueError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=prism_file_default, hdf5_file='test.txt',
                     emul_type='default')

    # Create a Pipeline object using invalid mock data spaces
    def test_invalid_mock_data_spaces_predefined(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        model_link._data_spc = ['A', 'B', 'C']
        pipe = Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_default, emul_type='default')
        with pytest.raises(NotImplementedError):
            pipe._emulator._create_new_emulator()

    # Create a Pipeline object using invalid mock data spaces
    def test_invalid_mock_data_spaces_undefined(self, tmpdir):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=model_data_single)
        model_link._data_spc = ['A', 'B', 'C']
        pipe = Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_default, emul_type='default')
        with pytest.raises(NotImplementedError):
            pipe._emulator._create_new_emulator()


# Pytest for Pipeline class request exception handling
class Test_Pipeline_Request_Exceptions(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        # Obtain paths
        tmpdir = tmpdir_factory.mktemp('test_exceptions')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a ModelLink object for testing the Pipeline
        model_link = GaussianLink2D()

        # Create a Pipeline object
        return(Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_impl, emul_type='default'))

    # Try to construct an iteration that cannot be constructed
    def test_invalid_construction(self, pipe):
        with pytest.raises(RequestError):
            pipe.construct(2)

    # Try to call an iteration that cannot be used
    def test_invalid_call(self, pipe):
        with pytest.raises(RequestError):
            pipe(2)

    # Try to call an iteration that cannot be finished
    def test_break_call(self, pipe):
        with pytest.raises(RequestError):
            pipe(1)

    # Try to construct an iteration with no plausible regions
    def test_impl_construction(self, pipe):
        with pytest.raises(RequestError):
            pipe.construct(2)


# Pytest for Pipeline class internal exception handling
class Test_Pipeline_Internal_Exceptions(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='class')
    def pipe(self, tmpdir_factory):
        # Obtain paths
        tmpdir = tmpdir_factory.mktemp('test_exceptions')
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a ModelLink object for testing the Pipeline
        model_link = GaussianLink2D()

        # Create a Pipeline object
        return(Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_impl, emul_type='default'))

    # Try to save data using the wrong keyword
    def test_invalid_save_data_keyword(self, pipe):
        pipe.construct(1, 0)
        with pytest.raises(ValueError):
            pipe._save_data({'test': []})


# Pytest for Pipeline class initialization versatility
class Test_Pipeline_Init_Versatility(object):
    # Create a model_link object used in some test functions
    @pytest.fixture(scope='function')
    def model_link(self):
        return(GaussianLink2D())

    # Create a Pipeline object using a custom Emulator class
    def test_custom_Emulator(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                 prism_file=prism_file_default, emul_type=Emulator)

    # Create a Pipeline object using custom pot_active_par
    def test_custom_pot_act_par(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        prism_file = path.join(dirpath, 'data/prism_custom_pot_act_par.txt')
        Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                 prism_file=prism_file, emul_type='default')

    # Create a Pipeline object using no defined paths
    def test_default_paths(self, model_link):
        pipe = Pipeline(model_link, emul_type='default')
        logging.shutdown()
        os.remove(path.join(pipe._working_dir, 'prism_log.log'))
        os.rmdir(pipe._working_dir)

    # Create a Pipeline object using a non_existent root dir
    def test_non_existent_root_dir(self, tmpdir, model_link):
        root_dir = path.join(tmpdir.strpath, 'root')
        Pipeline(model_link, root_dir=root_dir, prism_file=prism_file_default,
                 emul_type='default')

    # Create a Pipeline object using a non_existent root dir
    def test_non_existent_working_dir(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = 'working_dir'
        Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                 prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object using a relative path to a PRISM file
    def test_rel_path_PRISM_file(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)
        shutil.copy(prism_file_default, root_dir)
        Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                 prism_file='prism_default.txt', emul_type='default')

    # Create a Pipeline object requesting a new working dir three times
    def test_new_working_dir(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        Pipeline(model_link, root_dir=root_dir, working_dir=1,
                 prism_file=prism_file_default, emul_type='default')
        Pipeline(model_link, root_dir=root_dir, working_dir=1,
                 prism_file=prism_file_default, emul_type='default')
        Pipeline(model_link, root_dir=root_dir, working_dir=1,
                 prism_file=prism_file_default, emul_type='default')

    # Create a Pipeline object loading an existing working dir
    def test_load_existing_working_dir(self, tmpdir, model_link):
        root_dir = path.dirname(tmpdir.strpath)
        Pipeline(model_link, root_dir=root_dir, working_dir=1,
                 prism_file=prism_file_default, emul_type='default')
        Pipeline(model_link, root_dir=root_dir, working_dir=None,
                 prism_file=prism_file_default, emul_type='default')


# Pytest for Pipeline + ModelLink versatility
class Test_Pipeline_ModelLink_Versatility(object):
    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='function')
    def pipe2D(self, tmpdir):
        # Obtain paths
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a ModelLink object for testing the Pipeline
        model_link = GaussianLink2D()

        # Create a Pipeline object
        return(Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_impl, emul_type='default'))

    # Create a universal Pipeline object for testing request exceptions
    @pytest.fixture(scope='function')
    def pipe3D(self, tmpdir):
        # Obtain paths
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a ModelLink object for testing the Pipeline
        model_link = GaussianLink3D(model_parameters=model_parameters_3D,
                                    model_data=model_data_single)

        # Create a Pipeline object
        return(Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=prism_file_impl, emul_type='default'))

    # Test if multi_call can be used correctly
    def test_multi_call(self, pipe2D):
        # Change the modellink in pipe to use multi_call
        pipe2D._modellink.multi_call = True
        assert pipe2D._modellink._multi_call

        # Do a construction using multi_called model
        pipe2D.construct(1, 0)

    # Test if mock data takes log10 value spaces into account correctly
    def test_mock_data_spaces_log(self, pipe3D):
        # Change the modellink in pipe to have log10 value spaces
        pipe3D._modellink._data_spc = ['log10', 'log10', 'log10']

        # Request md_var
        pipe3D._emulator._create_new_emulator()

    # Test if mock data takes ln value spaces into account correctly
    def test_mock_data_spaces_ln(self, pipe3D):
        # Change the modellink in pipe to have ln value spaces
        pipe3D._modellink._data_spc = ['ln', 'ln', 'ln']

        # Request md_var
        pipe3D._emulator._create_new_emulator()

    # Test if an ext_real_set bigger than n_sam_init can be provided
    def test_ext_real_set_large(self, pipe2D):
        # Change the modellink in pipe to use multi_call
        pipe2D._modellink.multi_call = True
        assert pipe2D._modellink._multi_call

        # Create ext_real_set larger than n_sam_init
        sam_set = lhd(pipe2D._n_sam_init*2, pipe2D._modellink._n_par,
                      pipe2D._modellink._par_rng, 'center', pipe2D._criterion)
        sam_dict = dict(zip(pipe2D._modellink._par_name, sam_set.T))
        mod_set = pipe2D._modellink.call_model(
            1, sam_dict, np.array(pipe2D._modellink._data_idx))

        # Try to construct the iteration
        pipe2D.construct(1, 0, [sam_set, mod_set])

    # Test if an ext_real_set smaller than n_sam_init can be provided
    def test_ext_real_set_small(self, pipe2D):
        # Change the modellink in pipe to use multi_call
        pipe2D._modellink.multi_call = True
        assert pipe2D._modellink._multi_call

        # Create ext_real_set smaller than n_sam_init
        sam_set = lhd(pipe2D._n_sam_init//2, pipe2D._modellink._n_par,
                      pipe2D._modellink._par_rng, 'center', pipe2D._criterion)
        sam_dict = dict(zip(pipe2D._modellink._par_name, sam_set.T))
        mod_set = pipe2D._modellink.call_model(
            1, sam_dict, np.array(pipe2D._modellink._data_idx))

        # Try to construct the iteration
        pipe2D.construct(1, 0, [sam_set, mod_set])
