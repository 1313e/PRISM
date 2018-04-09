# -*- coding: utf-8 -*-

"""
Unittests for PRISM's Pipeline class

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import argparse
import os
from os import path
import time

# Package imports
import logging
import numpy as np


# %% TESTS
# Function that tests the functionality of the PRISM pipeline
# TODO: Add check if emulator system can be reloaded correctly
# TODO: This function needs to be updated with the changes made to active_par
def test_pipeline(output='test_emul_sine_wave', save=False):
    """
    Test script that uses the :class:`~prism.modellink.SineWaveLink` class to
    test all functionalities of the PRISM pipeline.

    Optional
    --------
    output : str. Default: 'test_emul_sine_wave'
        String containing the name of the working directory the test results
        need to be saved to.
    save : bool. Default: False
        Whether or not to keep the working directory after a successful test
        run.

    """

    # Save the path to this file
    file_path = path.dirname(__file__)

    # Try to import all PRISM modules
    try:
        from prism import Pipeline
        from prism.modellink import SineWaveLink
    except Exception:
        raise

    # Create instance of SineWaveLink
    model_link = SineWaveLink(path.join(file_path,
                                        'data/test_parameters_sine_wave.txt'),
                              path.join(file_path,
                                        'data/test_data_sine_wave.txt'))

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
    pipe = Pipeline(model_link, root_dir=file_path, working_dir=output,
                    prism_file='data/test_prism.txt')

    # Check if Pipeline details are correct
    assert pipe._modellink_name == 'SineWaveLink'
    assert pipe._n_sam_init == 100
    assert pipe._criterion is None
    assert pipe._do_active_par == 1
    assert pipe._use_mock == 0

    # Check if Emulator details are correct
    assert pipe._emulator._emul_load == 0, ("Directory '%s' already contains "
                                            "a constructed emulator system!"
                                            % (output))
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
    exp_impl_sam = np.load(path.join(file_path, 'data/impl_sam.npy'))
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
    exp_impl_los = np.load(path.join(file_path, 'data/impl_los.npy'))
    exp_impl_min = np.load(path.join(file_path, 'data/impl_min.npy'))

    # Check if data is equal
    assert np.allclose(impl_los, exp_impl_los)
    assert np.allclose(impl_min, exp_impl_min)

    # If all of this was successful, delete the working_dir and return True
    print("Test was successful!")
    if not save:
        working_dir = pipe._working_dir
        del pipe
        logging.shutdown()
        os.remove(path.join(working_dir, 'prism.hdf5'))
        os.remove(path.join(working_dir, 'prism_log.log'))
        os.remove(path.join(working_dir, 'proj_1_hcube_(A-B).png'))
        os.rmdir(working_dir)
        print("Deleted test directory '%s' and all of its contents."
              % (output))
    return(True)


# %% EXECUTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                 description="Test script for checking the functionality of "
                             "the PRISM pipeline.",
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', action='store', type=str,
                        default='test_emul_sine_wave',
                        help="output directory name")
    parser.add_argument('-s', '--save', action='store_true',
                        help="save output")
    args = parser.parse_args()

    # Start testing
    start_time = time.time()
    test_pipeline(args.output, args.save)
    print("Time elapsed: %ss" % (time.time()-start_time))
