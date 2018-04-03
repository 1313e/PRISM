# -*- coding: utf-8 -*-

"""
Created on Mon Apr  2 15:17:20 2018

@author: 1313e
"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Package imports
import logging
import numpy as np

# PRISM imports
from .__version__ import version as _prism_version
from ._internal import RequestError, check_compatibility, docstring_copy
from .emulator import Emulator
from .modellink import ModelLink

# All declaration
__all__ = ['MasterEmulator']


# %% MASTER EMULATOR CLASS DEFINITION
class MasterEmulator(Emulator):
    """


    """

    # Identify this class as being a master emulator
    _emul_type = 'master'


# %% GENERAL CLASS METHODS
    def _load_data(self, emul_i):
        """


        """

        # Set the logger
        logger = logging.getLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing emulator data sets.")
        self._n_sam = [[]]*(emul_i+1)
        self._sam_set = [[]]*(emul_i+1)
        self._active_par = [[]]*(emul_i+1)
        self._mod_set = [[]]*(emul_i+1)
        self._cov_mat_inv = [[]]*(emul_i+1)
        self._prior_exp_sam_set = [[]]*(emul_i+1)
        self._active_par_data = [[]]*(emul_i+1)
        self._rsdl_var = [[]]*(emul_i+1)
        self._poly_coef = [[]]*(emul_i+1)
        self._poly_coef_cov = [[]]*(emul_i+1)
        self._poly_powers = [[]]*(emul_i+1)
        self._poly_idx = [[]]*(emul_i+1)
        self._n_data = [[]]*(emul_i+1)
        self._data_err = [[]]*(emul_i+1)
        self._data_idx = [[]]*(emul_i+1)

        # If no file has been provided
        if(emul_i == 0 or self._emul_load == 0):
            logger.info("Non-existent emulator file provided. No additional "
                        "data needs to be loaded.")
            return

        # Load emulator data from construction file
        elif(self._emul_load == 1):
            # Load the corresponding sam_set, mod_set and cov_mat_inv
            logger.info("Loading relevant emulator data of iteration %s."
                        % (self._emul_i))

            # Open hdf5-file
            file = self._open_hdf5('r')

            # Read in the data
            self._n_sam[emul_i] = file['%s' % (emul_i)].attrs['n_sam']
            self._sam_set[emul_i] = file['%s/sam_set' % (emul_i)][()]
            self._active_par[emul_i] = file['%s/active_par' % (emul_i)][()]
            mod_set = []
            cov_mat_inv = []
            prior_exp_sam_set = []
            active_par_data = []
            self._n_data[emul_i] = file['%s' % (emul_i)].attrs['n_data']
            data_err = []
            data_idx = []
            for i in range(self._n_data[emul_i]):
                mod_set.append(file['%s/data_set_%s/mod_set'
                                    % (emul_i, i)][()])
                cov_mat_inv.append(file['%s/data_set_%s/cov_mat_inv'
                                        % (emul_i, i)][()])
                prior_exp_sam_set.append(
                    file['%s/data_set_%s/prior_exp_sam_set'
                         % (emul_i, i)][()])
                active_par_data.append(file['%s/data_set_%s/active_par_data'
                                            % (emul_i, i)][()])
                data_err.append(file['%s/data_set_%s'
                                     % (emul_i, i)].attrs['data_err'])
                data_idx.append(file['%s/data_set_%s'
                                     % (emul_i, i)].attrs['data_idx'])
            self._mod_set[emul_i] = mod_set
            self._cov_mat_inv[emul_i] = cov_mat_inv
            self._prior_exp_sam_set[emul_i] = prior_exp_sam_set
            self._active_par_data[emul_i] = active_par_data
            self._data_err[emul_i] = data_err
            self._data_idx[emul_i] = data_idx

            if self._method.lower() in ('regression', 'full'):
                rsdl_var = []
                poly_coef = []
                poly_coef_cov = []
                poly_powers = []
                poly_idx = []
                for i in range(self._n_data[emul_i]):
                    rsdl_var.append(
                        file['%s/data_set_%s'
                             % (self._emul_i, i)].attrs['rsdl_var'])
                    poly_coef.append(file['%s/data_set_%s/poly_coef'
                                          % (emul_i, i)][()])
                    poly_coef_cov.append(file['%s/data_set_%s/poly_coef_cov'
                                              % (emul_i, i)][()])
                    poly_powers.append(file['%s/data_set_%s/poly_powers'
                                            % (emul_i, i)][()])
                    poly_idx.append(file['%s/data_set_%s/poly_idx'
                                         % (emul_i, i)][()])
                self._rsdl_var[emul_i] = rsdl_var
                self._poly_coef[emul_i] = poly_coef
                self._poly_coef_cov[emul_i] = poly_coef_cov
                self._poly_powers[emul_i] = poly_powers
                self._poly_idx[emul_i] = poly_idx

            # Close hdf5-file
            self._close_hdf5(file)

            # Log that loading is finished
            logger.info("Finished loading relevant emulator data.")
        else:
            raise RequestError("Invalid operation requested!")

    # This function saves emulator data to hdf5
    @docstring_copy(Emulator._save_data)
    def _save_data(self, emul_i, keyword, data):

        # Do some logging
        logger = logging.getLogger('SAVE_DATA')
        logger.info("Saving %s data at iteration %s to HDF5."
                    % (keyword, emul_i))

        # Open hdf5-file
        file = self._open_hdf5('r+')

        # Check what data keyword has been provided
        # ACTIVE PARAMETERS
        if(keyword == 'active_par'):
            file.create_dataset('%s/active_par' % (emul_i), data=data[0])
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/active_par_data'
                                    % (emul_i, i), data=data[1][i])
            self._active_par.insert(0, [])
            self._active_par_data.insert(0, [])

            self._active_par[emul_i] = data[0]
            self._active_par_data[emul_i] = data[1]

        # COV_MAT
        elif(keyword == 'cov_mat'):
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/cov_mat' % (emul_i, i),
                                    data=data[0][i])
                file.create_dataset('%s/data_set_%s/cov_mat_inv' % (emul_i, i),
                                    data=data[1][i])
            self._cov_mat_inv.insert(0, [])
            self._cov_mat_inv[emul_i] = data[1]

        # MOD_SET
        elif(keyword == 'mod_set'):
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/mod_set'
                                    % (emul_i, i), data=data[i])
            self._mod_set.insert(0, [])
            self._mod_set[emul_i] = data

        # PRIOR_EXP_SAM_SET
        elif(keyword == 'prior_exp_sam_set'):
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/prior_exp_sam_set'
                                    % (emul_i, i), data=data[i])
            self._prior_exp_sam_set.insert(0, [])
            self._prior_exp_sam_set[emul_i] = data

        # REGRESSION
        elif(keyword == 'regression'):
            for i in range(self._n_data[emul_i]):
                file['%s/data_set_%s' % (emul_i, i)].attrs['rsdl_var'] =\
                    data[0][i]
                file.create_dataset('%s/data_set_%s/poly_coef'
                                    % (emul_i, i), data=data[1][i])
                file.create_dataset('%s/data_set_%s/poly_coef_cov'
                                    % (emul_i, i), data=data[2][i])
                file.create_dataset('%s/data_set_%s/poly_powers'
                                    % (emul_i, i), data=data[3][i])
                file.create_dataset('%s/data_set_%s/poly_idx'
                                    % (emul_i, i), data=data[4][i])
            self._rsdl_var.insert(0, [])
            self._poly_coef.insert(0, [])
            self._poly_coef_cov.insert(0, [])
            self._poly_powers.insert(0, [])
            self._poly_idx.insert(0, [])

            self._rsdl_var[emul_i] = data[0]
            self._poly_coef[emul_i] = data[1]
            self._poly_coef_cov[emul_i] = data[2]
            self._poly_powers[emul_i] = data[3]
            self._poly_idx[emul_i] = data[4]

        # SAM_SET
        elif(keyword == 'sam_set'):
            file.create_dataset('%s/sam_set' % (emul_i), data=data)
            file['%s' % (emul_i)].attrs['n_sam'] = np.shape(data)[0]
            self._sam_set.insert(0, [])
            self._n_sam.insert(0, [])

            self._sam_set[emul_i] = data
            self._n_sam[emul_i] = np.shape(data)[0]

        # INVALID KEYWORD
        else:
            logger.error("Invalid keyword argument provided!")
            raise ValueError("Invalid keyword argument provided!")

        # Close hdf5-file
        self._close_hdf5(file)

        # More logging
        logger.info("Finished saving data to HDF5.")

    # Prepares the construction/improvement of a master emulator system
    @docstring_copy(Emulator._prepare_new_iteration)
    def _prepare_new_iteration(self, emul_i):

        # Logger
        logger = logging.getLogger('EMUL_PREP')
        logger.info("Preparing master emulator for construction.")

        # Check if new iteration can be constructed
        logger.info("Checking if emulator iteration can be prepared.")
        if(emul_i == 1):
            # Set reload flag to 1
            reload = 1
        elif not(emul_i-1 == self._emul_i):
            logger.error("Preparation of master emulator iteration %s is only "
                         "available when the previous iteration exists!"
                         % (emul_i))
            raise RequestError("Preparation of master emulator iteration %s is"
                               " only available when the previous iteration "
                               "exists!" % (emul_i))
        elif(emul_i-1 == self._emul_i):
            # Set reload flag to 0
            reload = 0
        else:
            # TODO: Also delete all projection figures?
            logger.info("Emulator iteration %s already exists. Deleting "
                        "requested and all subsequent iterations.")

            # Open hdf5-file
            file = self._open_hdf5('r+')

            # Delete requested and subsequent emulator iterations
            for i in range(emul_i, self._emul_i+1):
                del file['%s' % (i)]

            # Close hdf5-file
            self._close_hdf5(file)

            # Set last emul_i to preceding requested iteration
            self._emul_i = emul_i-1

            # Check if repreparation was actually necessary
            # TODO: Think about how to extend this check
            for i in range(self._n_data[emul_i]):
                if not(self._data_err[emul_i][i] in self._modellink._data_err):
                    break
                if not(self._data_idx[emul_i][i] in self._modellink._data_idx):
                    break
            # If not, give out a warning
            else:
                logger.warning("No differences in model comparison data "
                               "detected.\nUnless this repreparation was "
                               "intentional, using the 'analyze' method of "
                               "the Pipeline class is much faster for "
                               "reanalyzing the emulator with new pipeline "
                               "parameters.")
                print("No differences in model comparison data "
                      "detected.\nUnless this repreparation was "
                      "intentional, using the 'analyze' method of "
                      "the Pipeline class is much faster for "
                      "reanalyzing the emulator with new pipeline "
                      "parameters.")

            # Reload emulator data
            self._load_data(self._emul_i)

            # Set reload flag to 1
            reload = 1

        # Open hdf5-file
        file = self._open_hdf5('r+')

        # Make group for this emulator iteration
        file.create_group('%s' % (emul_i))

        # Save the number of data sets
        file['%s' % (emul_i)].attrs['n_data'] = self._modellink._n_data
        self._n_data.insert(0, [])
        self._n_data[emul_i] = self._modellink._n_data

        # Create empty lists for the three data arrays
        data_err = []
        data_idx = []

        # Create groups for all data sets
        # TODO: Add check if all previous data sets are still present?
        for i in range(self._modellink._n_data):
            file.create_group('%s/data_set_%s' % (emul_i, i))
            file['%s/data_set_%s' % (emul_i, i)].attrs['data_err'] =\
                self._modellink._data_err[i]
            data_err.append(self._modellink._data_err[i])
            file['%s/data_set_%s' % (emul_i, i)].attrs['data_idx'] =\
                self._modellink._data_idx[i]
            data_idx.append(self._modellink._data_idx[i])

        # Close hdf5-file
        self._close_hdf5(file)

        # Save model data arrays to memory
        self._data_err.insert(0, [])
        self._data_idx.insert(0, [])
        self._data_err[emul_i] = data_err
        self._data_idx[emul_i] = data_idx

        # Logging
        logger.info("Finished preparing emulator iteration.")

        # Return the result
        return(reload)
