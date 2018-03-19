# -*- coding: utf-8 -*-

"""
Unittests for PRISM's Pipeline class

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import os
from os import path

# Package imports
import logging
import numpy as np

# PRISM imports
from prism import Pipeline
from prism.modellink.sine_wave_link import SineWaveLink


# %% TESTS
def test_pipeline():
    # Create instance of SineWaveLink
    model_link = SineWaveLink(path.join(path.dirname(__file__),
                                        'test_parameters_sine_wave.txt'),
                              path.join(path.dirname(__file__),
                                        'test_data_sine_wave.txt'))

    # Check if all model parameters details are correct
    assert model_link._par_dim == 4
    assert model_link._par_names == ['A', 'B', 'C', 'D']
    assert (model_link._par_rng == [[2, 7], [-1, 12], [0, 10], [1.5, 5]]).all()
    assert model_link._par_estimate == [4.0, 3.0, 5.0, 4.6]

    # Check if all model data details are correct
    assert model_link._data_val == [4.48395984, 3.53861229, 4.42958091,
                                    3.61082396, 4.34098181]
    assert model_link._data_err == [0.05, 0.05, 0.05, 0.05, 0.05]
    assert model_link._data_idx == [1, 2, 3, 4, 5]

    # Create instance of Pipeline
    pipe = Pipeline(model_link, root_dir=path.dirname(__file__),
                    working_dir='test_emul_sine_wave',
                    prism_file=path.join(path.dirname(__file__),
                                         'test_prism.txt'))

    # Check if Pipeline details are correct
    assert pipe._modellink_name == 'SineWaveLink'
    assert pipe._n_sam_init == 100
    assert pipe._criterion is None
    assert pipe._do_active_par == 1
    assert pipe._use_mock == 0

    # Check if Emulator details are correct
    assert pipe._emulator._emul_load == 0
    assert pipe._emulator._emul_i == 0

    # Set randomizer seed
    np.random.seed(1)

    # Try to construct the first iteration of the emulator system
    try:
        pipe.construct(1)
    except Exception:
        raise

    # If successful, do more checks
    # Check if emulator system has correct details
    assert pipe._emulator._sigma == 0.8
    assert (pipe._emulator._l_corr == [1.5, 3.9, 3., 1.05]).all()
    assert pipe._emulator._method == 'full'
    assert pipe._emulator._poly_order == 3

    # Check if emulator analysis has been done correctly
    assert (pipe._impl_cut[1] == [0.0, 4.0, 3.8, 3.5]).all()
    assert pipe._prc[1] == 1
    exp_impl_sam = np.load(path.join(path.dirname(__file__), 'impl_sam.npy'))
    assert np.allclose(pipe._impl_sam[1], exp_impl_sam)

    # Check if emulator system is correctly built
    assert (pipe._emulator._active_par[1] == [0, 1, 2, 3]).all()
    eval_results = pipe.evaluate([[4, 4, 4, 4]])
    exp_results =\
        [[1],
         [1],
         [[3.95821524, 3.80566275, 4.05909692, 3.96721204, 3.97736399]],
         [[0.04749773, 0.04724487, 0.04289714, 0.04705286, 0.04696085]],
         [[1.66037288, 0.93803417, 1.20550279, 1.24343962, 1.17009586]]]
    assert eval_results[0] == exp_results[0]
    assert eval_results[1] == exp_results[1]
    assert np.allclose(eval_results[2][0], exp_results[2][0])
    assert np.allclose(eval_results[3][0], exp_results[3][0])
    assert np.allclose(eval_results[4][0], exp_results[4][0])

    # Try to construct the projection figure of the first two active parameters
    try:
        pipe.create_projection(1, proj_par=(0, 1), figure=True)
    except Exception:
        raise

    # Check if projection data is as expected
    # Obtain generated projection data
    # Open hdf5-file
    file = pipe._open_hdf5('r')

    # Obtain data
    impl_los = file['1/proj_hcube/A-B/impl_los'][()]
    impl_min = file['1/proj_hcube/A-B/impl_min'][()]

    # Close hdf5-file
    pipe._close_hdf5(file)

    # Obtain expected projection data
    exp_impl_los = np.load(path.join(path.dirname(__file__), 'impl_los.npy'))
    exp_impl_min = np.load(path.join(path.dirname(__file__), 'impl_min.npy'))

    # Check if data is equal
    assert np.allclose(impl_los, exp_impl_los)
    assert np.allclose(impl_min, exp_impl_min)

    # If all of this was successful, delete the working_dir and return True
    del pipe
    logging.shutdown()
    os.remove('test_emul_sine_wave/prism.hdf5')
    os.remove('test_emul_sine_wave/prism_log.log')
    os.remove('test_emul_sine_wave/proj_1_hcube_(A-B).png')
    os.rmdir('test_emul_sine_wave')
    return(True)


# %% EXECUTION
if __name__ == '__main__':
    test_pipeline()
