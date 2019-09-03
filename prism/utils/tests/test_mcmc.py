# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path

# Package imports
from e13tools.core import InputError
from e13tools.sampling import lhd
from emcee import EnsembleSampler
import numpy as np
from py.path import local
import pytest
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._internal import RequestError, RequestWarning
from prism._pipeline import Pipeline
from prism.modellink.tests.modellink import GaussianLink2D
from prism.utils.mcmc import get_hybrid_lnpost_fn, get_walkers

# Set the random seed of NumPy
np.random.seed(0)

# Set the current working directory to the temporary directory
local.get_temproot().chdir()


# %% GLOBALS
DIR_PATH = path.dirname(__file__)           # Path to directory of this file


# %% CUSTOM FUNCTIONS
# Create Pipeline object to use for testing purposes
@pytest.fixture(scope='module')
def pipe(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('test_mcmc')
    root_dir = path.dirname(tmpdir.strpath)
    working_dir = path.basename(tmpdir.strpath)
    prism_file = path.join(DIR_PATH, 'data/prism_default.txt')
    modellink_obj = GaussianLink2D()
    np.random.seed(0)
    pipeline_obj = Pipeline(modellink_obj, root_dir=root_dir,
                            working_dir=working_dir, prism_par=prism_file)
    pipeline_obj.construct(analyze=False)
    return(pipeline_obj)


# Create lnpost function
def lnpost(par_set, pipe):
    # Obtain mod_out
    emul_i = pipe._emulator._emul_i
    if not isinstance(par_set, dict):
        par_dict = sdict(zip(pipe._modellink._par_name, par_set))
    else:
        par_dict = par_set
    mod_out = pipe._modellink.call_model(emul_i, par_dict,
                                         pipe._modellink._data_idx)

    # As GaussianLink2D returns dicts, convert to NumPy array
    mod_out = np.array([mod_out[idx] for idx in pipe._modellink._data_idx])

    # Get model and data variances. Since val_spc is lin, data_err is centered
    md_var = pipe._modellink.get_md_var(emul_i, par_dict,
                                        pipe._modellink._data_idx)
    md_var = np.array([md_var[idx] for idx in pipe._modellink._data_idx])
    data_var = [err[0]**2 for err in pipe._modellink._data_err]
    sigma2 = md_var+data_var
    return(-0.5*(np.sum((pipe._modellink._data_val-mod_out)**2/sigma2)))


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for get_hybrid_lnpost_fn
class Test_get_hybrid_lnpost_fn(object):
    # Try to obtain the default hybrid_lnpost function definition
    def test_default(self, pipe):
        hybrid_lnpost = get_hybrid_lnpost_fn(lnpost, pipe, unit_space=True)
        assert hybrid_lnpost([-1, 2], pipe) == -np.infty
        assert hybrid_lnpost([0.5, 0.5], pipe) == -np.infty
        hybrid_lnpost(pipe._modellink._to_unit_space(pipe._modellink._par_est),
                      pipe)

    # Try to provide a non-callable function
    def test_non_callable(self, pipe):
        with pytest.raises(InputError):
            get_hybrid_lnpost_fn(np.array(1), pipe)

    # Try to provide a non-Pipeline object
    def test_no_Pipeline(self):
        with pytest.raises(TypeError):
            get_hybrid_lnpost_fn(lnpost, np.array(1))

    # Try to provide a Pipeline object that uses a non-default emulator
    def test_invalid_emulator(self, pipe):
        pipe._emulator._emul_type = 'test'
        with pytest.raises(InputError):
            get_hybrid_lnpost_fn(lnpost, pipe)
        pipe._emulator._emul_type = 'default'

    # Try to provide a bound ModelLink object solely requesting multi-calls
    def test_multi_call_ModelLink(self, pipe):
        with pytest.warns(RequestWarning):
            pipe._modellink.call_type = 'multi'
        if pipe._is_controller:
            with pytest.warns(UserWarning):
                get_hybrid_lnpost_fn(lnpost, pipe)
        else:
            get_hybrid_lnpost_fn(lnpost, pipe)
        pipe._modellink.call_type = 'hybrid'

    # Try to obtain the non-default hybrid_lnpost function definition
    def test_non_default(self, pipe):
        hybrid_lnpost = get_hybrid_lnpost_fn(lnpost, pipe, par_dict=True,
                                             unit_space=True, impl_prior=False)
        assert hybrid_lnpost({'A': 0.5, 'B': 0.5}, pipe) == -np.infty
        hybrid_lnpost(sdict(zip(pipe._modellink._par_name,
                                pipe._modellink._par_est)), pipe)


# Pytest for get_walkers
class Test_get_walkers(object):
    # Try to obtain the default walkers
    def test_default(self, pipe):
        if(pipe._size == 1):
            with pytest.raises(RequestError):
                get_walkers(pipe)
        pipe.analyze()
        get_walkers(pipe)
        pipe.construct()
        get_walkers(pipe, emul_i=1, unit_space=True)

    # Try to provide a non-Pipeline object
    def test_no_Pipeline(self):
        with pytest.raises(TypeError):
            get_walkers(np.array(1))

    # Try to provide a Pipeline object that uses a non-default emulator
    def test_invalid_emulator(self, pipe):
        pipe._emulator._emul_type = 'test'
        with pytest.raises(InputError):
            get_walkers(pipe)
        pipe._emulator._emul_type = 'default'

    # Try to provide a non-callable function
    def test_non_callable(self, pipe):
        with pytest.raises(InputError):
            get_walkers(pipe, lnpost_fn=np.array(1))

    # Try to provide a custom set of init_walkers
    def test_init_walkers_set(self, pipe):
        get_walkers(pipe, init_walkers=lhd(10, pipe._modellink._n_par, None,
                                           'center', pipe._criterion, 100),
                    unit_space=True)

    # Try to provide a custom dict of init_walkers
    def test_init_walkers_dict(self, pipe):
        sam_set = lhd(10, pipe._modellink._n_par, pipe._modellink._par_rng,
                      'center', pipe._criterion, 100)
        sam_dict = sdict(zip(pipe._modellink._par_name, sam_set.T))
        p0_walkers = get_walkers(pipe, init_walkers=sam_dict)[1]
        if pipe._is_controller:
            assert isinstance(p0_walkers, dict)

    # Try to provide a custom size of init_walkers
    def test_init_walkers_size(self, pipe):
        get_walkers(pipe, init_walkers=10)

    # Try to request a specific number of walkers
    def test_req_n_walkers(self, pipe):
        if pipe._is_controller:
            req_n_walkers = pipe._comm.bcast(pipe._n_impl_sam[-1]+1, 0)
        else:
            req_n_walkers = pipe._comm.bcast(None, 0)
        get_walkers(pipe, req_n_walkers=req_n_walkers)

    # Try to provide a set of implausible init_walkers
    def test_no_plausible_init_walkers(self, pipe):
        with pytest.raises(InputError):
            get_walkers(pipe, init_walkers=lhd(1, pipe._modellink._n_par,
                                               pipe._modellink._par_rng,
                                               'center', pipe._criterion, 100))

    # Try to provide a custom lnpost function
    def test_custom_lnpost(self, pipe):
        get_walkers(pipe, lnpost_fn=lnpost)


# Pytest for connecting to emcee
def test_hybrid_sampling(pipe):
    n_walkers, p0, hybrid_lnpost = get_walkers(pipe, lnpost_fn=lnpost)
    n_walkers *= 2
    p0 = np.concatenate([p0, p0])
    with pipe.worker_mode:
        if pipe._is_controller:
            sampler = EnsembleSampler(n_walkers, pipe._modellink._n_par,
                                      hybrid_lnpost, args=[pipe])
            sampler.run_mcmc(p0, 10)
