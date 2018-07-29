# -*- coding: utf-8 -*-

"""
Emulator
========
Provides the definition of the class holding the emulator system of the PRISM
package, the :class:`~Emulator` class.


Available classes
-----------------
:class:`~Emulator`
    Defines the :class:`~Emulator` class of the PRISM package.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from time import time
import sys

# Package imports
from e13tools import InputError
from e13tools.math import diff, nearest_PD
import h5py
import logging
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mpi4py import MPI
import numpy as np
from numpy.linalg import inv, norm
# TODO: Do some research on sklearn.linear_model.SGDRegressor
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline as Pipeline_sk
from sklearn.preprocessing import PolynomialFeatures as PF
from sortedcontainers import SortedSet

# PRISM imports
from .__version__ import prism_version as _prism_version
from ._docstrings import get_emul_i_doc, std_emul_i_doc
from ._internal import (RequestError, check_bool, check_compatibility,
                        check_nzero_float, check_pos_float, check_pos_int,
                        docstring_substitute)
from .modellink import ModelLink

# All declaration
__all__ = ['Emulator']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% EMULATOR CLASS DEFINITION
class Emulator(object):
    """
    Defines the :class:`~Emulator` class of the PRISM package.

    """

    # Identify this class as being a default emulator
    _emul_type = 'default'

    def __init__(self, pipeline_obj, modellink_obj):
        """
        Initialize an instance of the :class:`~Emulator` class.

        Parameters
        ----------
        pipeline_obj : :obj:`~Pipeline` object
            Instance of the :class:`~Pipeline` class that initialized this
            class.
        modellink_obj : :obj:`~ModelLink` object
            Instance of the :class:`~ModelLink` class that links the emulated
            model to this :obj:`~Pipeline` object.

        """

        # Save the provided pipeline object
        self._pipeline = pipeline_obj

        # Load the emulator and data
        self._load_emulator(modellink_obj)


# %% CLASS PROPERTIES
    # General details
    @property
    def emul_load(self):
        """
        Bool indicating whether or not a previously constructed emulator system
        is currently loaded.

        """

        return(bool(self._emul_load))

    @property
    def emul_type(self):
        """
        String indicating what type of emulator system is currently loaded.

        """

        return(self._emul_type)

    @property
    def emul_i(self):
        """
        Integer indicating the last/latest available emulator iteration.

        """

        return(self._emul_i)

    @property
    def n_sam(self):
        """
        Number of model evaluation samples in currently loaded emulator
        iteration.

        """

        return(self._n_sam)

    @property
    def method(self):
        """
        String indicating which emulator method to use.
        Possible are 'gaussian', 'regression', 'auto' and 'full'.

        """

        return(self._method)

    @property
    def use_mock(self):
        """
        Bool indicating whether or not mock data has been used for the creation
        of this emulator instead of actual data.

        """

        return(bool(self._use_mock))

    # TODO: Allow selective regr_cov usage?
    @property
    def use_regr_cov(self):
        """
        Bool indicating whether or not to take into account the regression
        covariance when calculating the covariance of the emulator, in addition
        to the Gaussian covariance.
        If `method` == 'gaussian', this bool is not required.

        """

        return(bool(self._use_regr_cov))

    @property
    def poly_order(self):
        """
        Polynomial order that is considered for the regression process.
        If `method` == 'gaussian', this number is not required.

        """

        return(self._poly_order)

    # Active Parameters
    @property
    def active_par(self):
        """
        List containing the model parameter identifiers that are considered
        active in the currently loaded emulator iteration.

        """

        return(self._active_par)

    @property
    def active_par_data(self):
        """
        List containing the model parameter identifiers that are considered
        active in the currently loaded emulator iteration, separated for every
        data point.

        """

        return(self._active_par_data)

    # Regression
    @property
    def rsdl_var(self):
        """
        List with residual variances for every data point in the currently
        loaded emulator iteration.
        Obtained from regression process and replaces the Gaussian sigma.

        """

        return(self._rsdl_var)

    @property
    def poly_coef(self):
        """
        List with non-zero coefficients for the polynomial terms in the
        regression function in the currently loaded emulator iteration,
        separated per data point.

        """

        return(self._poly_coef)

    @property
    def poly_coef_cov(self):
        """
        List with covariances for all polynomial coefficients in the
        regression function in the currently loaded emulator iteration,
        separated per data point.

        """

        return(self._poly_coef_cov)

    @property
    def poly_powers(self):
        """
        List containing the polynomial term powers in the currently loaded
        emulator iteration, separated per data point.

        """

        return(self._poly_powers)

    @property
    def poly_idx(self):
        """
        List containing the indices of the polynomial terms with non-zero
        coefficients in the currently loaded emulator iteration, separated per
        data point.

        """

        return(self._poly_idx)

    # Emulator Data
    @property
    def sam_set(self):
        """
        Array containing all model evaluation samples in the currently loaded
        emulator iteration.

        """

        return(self._sam_set)

    @property
    def mod_set(self):
        """
        Array containing all model outputs in the currently loaded emulator
        iteration.

        """

        return(self._mod_set)

    @property
    def cov_mat_inv(self):
        """
        Array containing the inverses of the covariance matrices in the
        currently loaded emulator iteration, separated per data point.

        """

        return(self._cov_mat_inv)

    @property
    def prior_exp_sam_set(self):
        """
        Array containing the prior emulator expectation values of all model
        evaluation samples in the currently loaded emulator iteration.

        """

        return(self._prior_exp_sam_set)

    # Covariances
    @property
    def sigma(self):
        """
        List with Gaussian sigmas.
        If `method` == 'regression' or 'full', this value is not required,
        since it is obtained from the regression process instead.

        """

        return(self._sigma)

    @property
    def l_corr(self):
        """
        List with Gaussian correlation lengths for active parameters.

        """

        return(self._l_corr)


# %% GENERAL CLASS METHODS
    # Get correct emulator iteration
    @docstring_substitute(emul_i=get_emul_i_doc)
    def _get_emul_i(self, emul_i):
        """
        Checks if the provided emulator iteration `emul_i` can be requested or
        replaces it if *None* was provided.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        emul_i : int
            The requested emulator iteration that passed the check.

        """

        # Log that emul_i is being switched
        logger = logging.getLogger('INIT')
        logger.info("Selecting emulator iteration for user-method.")

        # Check if provided emul_i is correct/allowed
        if(emul_i == 0 or self._emul_load == 0):
            raise RequestError("Emulator HDF5-file is not built yet!")
        elif emul_i is None:
            emul_i = self._emul_i
        elif not(1 <= emul_i <= self._emul_i):
            logger.error("Requested emulator iteration %s does not exist!"
                         % (emul_i))
            raise RequestError("Requested emulator iteration %s does not "
                               "exist!" % (emul_i))
        else:
            emul_i = check_pos_int(emul_i, 'emul_i')

        # Do some logging
        logger.info("Selected emulator iteration %s." % (emul_i))

        # Return correct emul_i
        return(emul_i)

    # Creates a new emulator file and writes all information to it
    # TODO: Allow for user-provided code to be executed here
    # Like, copying files after creating a new emulator
    def _create_new_emulator(self):
        """
        Creates a new HDF5-file that holds all the information of a new
        emulator system and writes all important emulator details to it.
        Afterward, resets all loaded emulator data and prepares the HDF5-file
        and emulator system for the construction of the first emulator
        iteration.

        Generates
        ---------
        A new HDF5-file contained in the working directory specified in the
        :obj:`~Pipeline` instance, holding all information required to
        construct the first iteration of the emulator system.

        """

        # Start logger
        logger = logging.getLogger('INIT')
        logger.info("Creating a new emulator system in HDF5-file '%s'."
                    % (self._pipeline._hdf5_file))

        # If no constructed emulator was provided, it will be constructed now
        # Therefore, set emul_load to 1 and emul_i to 0
        self._emul_load = 1
        self._emul_i = 0

        # Read in parameters from provided parameter file
        self._read_parameters()

        # Create hdf5-file
        file = self._pipeline._open_hdf5('w')

        # Save all relevant emulator parameters to hdf5
        file.attrs['sigma'] = self._sigma
        file.attrs['l_corr'] = self._l_corr
        file.attrs['method'] = self._method.encode('ascii', 'ignore')
        file.attrs['use_regr_cov'] = bool(self._use_regr_cov)
        file.attrs['poly_order'] = self._poly_order
        file.attrs['modellink_name'] =\
            self._modellink._name.encode('ascii', 'ignore')
        file.attrs['prism_version'] = _prism_version.encode('ascii', 'ignore')
        file.attrs['emul_type'] = self._emul_type.encode('ascii', 'ignore')
        file.attrs['use_mock'] = bool(self._use_mock)

        # Check if mock data is requested
        if self._use_mock:
            # If so, let workers know to call _get_mock_data() as well
            for rank in range(1, self._pipeline._size):
                MPI.COMM_WORLD.send(1, dest=rank, tag=999+rank)

            # Controller calling _get_mock_data() and saving to hdf5
            self._pipeline._get_mock_data()
            file.attrs['mock_par'] = self._modellink._par_est
        else:
            # If not, let workers know
            for rank in range(1, self._pipeline._size):
                MPI.COMM_WORLD.send(0, dest=rank, tag=999+rank)

        # Close hdf5-file
        self._pipeline._close_hdf5(file)

        # Load relevant data
        self._load_data(0)

        # Prepare first emulator iteration to be constructed
        self._prepare_new_iteration(1)

        # Logging again
        logger.info("Finished creating new emulator system.")

    # Prepares the emulator for a new iteration
    # HINT: Should _create_new_emulator be combined with this method?
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _prepare_new_iteration(self, emul_i):
        """
        Prepares the emulator system for the construction of a new iteration
        `emul_i`. Checks if this iteration can be prepared or if it has been
        prepared before, and acts accordingly.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        reload : bool
            Bool indicating whether or not the :obj:`~Pipeline` instance needs
            to reload its data.

        Generates
        ---------
        A new group in the HDF5-file with the emulator iteration as its name,
        containing subgroups corresponding to all different model comparison
        data that will be used in this iteration.

        Notes
        -----
        Preparing an iteration that has already been constructed, causes that
        and all subsequent iterations of the emulator system to be deleted.
        A check is carried out to see if it was necessary to reprepare the
        requested iteration and a warning is given if this check fails.

        """

        # Logger
        logger = logging.getLogger('EMUL_PREP')
        logger.info("Preparing emulator iteration %s for construction."
                    % (emul_i))

        # Check if new iteration can be constructed
        logger.info("Checking if emulator iteration can be prepared.")
        if(emul_i == 1):
            # Set reload flag to 1
            reload = 1
        elif not(1 <= emul_i-1 <= self._emul_i):
            logger.error("Preparation of emulator iteration %s is only "
                         "available when all previous iterations exist!"
                         % (emul_i))
            raise RequestError("Preparation of emulator iteration %s is only "
                               "available when all previous iterations exist!"
                               % (emul_i))
        elif(emul_i-1 == self._emul_i):
            # Set reload flag to 0
            reload = 0
        else:
            # TODO: Also delete all projection figures?
            logger.info("Emulator iteration %s already exists. Deleting "
                        "requested and all subsequent iterations.")

            # Open hdf5-file
            file = self._pipeline._open_hdf5('r+')

            # Delete requested and subsequent emulator iterations
            for i in range(emul_i, self._emul_i+1):
                del file['%s' % (i)]

            # Close hdf5-file
            self._pipeline._close_hdf5(file)

            # Set last emul_i to preceding requested iteration
            self._emul_i = emul_i-1

            # Check if repreparation was actually necessary
            # TODO: Think about how to extend this check
            for i in range(self._n_data[emul_i]):
                if not(self._data_val[emul_i][i] in self._modellink._data_val):
                    break
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
        file = self._pipeline._open_hdf5('r+')

        # Make group for this emulator iteration
        file.create_group('%s' % (emul_i))

        # Save the number of data points
        file['%s' % (emul_i)].attrs['n_data'] = self._modellink._n_data
        self._n_data.append(self._modellink._n_data)

        # Create an empty data set for statistics as attributes
        file.create_dataset('%s/statistics' % (emul_i), data=h5py.Empty(float))

        # Create empty lists for the four data arrays
        data_val = []
        data_err = []
        data_idx = []
        data_prev = []

        # Create groups for all data points
        for i in range(self._modellink._n_data):
            data_set = file.create_group('%s/data_point_%s' % (emul_i, i))
            data_set.attrs['data_val'] = self._modellink._data_val[i]
            data_val.append(self._modellink._data_val[i])
            data_set.attrs['data_err'] = self._modellink._data_err[i]
            data_err.append(self._modellink._data_err[i])

            # Save the data_idx in portions to make it HDF5-compatible
            if isinstance(self._modellink._data_idx[i], list):
                for j, idx in enumerate(self._modellink._data_idx[i]):
                    if isinstance(idx, (str, unicode)):
                        data_set.attrs['data_idx_%s' % (j)] =\
                            idx.encode('ascii', 'ignore')
                    else:
                        data_set.attrs['data_idx_%s' % (j)] = idx
            else:
                if isinstance(self._modellink._data_idx[i], (str, unicode)):
                    data_set.attrs['data_idx'] =\
                        self._modellink._data_idx[i].encode('ascii', 'ignore')
                else:
                    data_set.attrs['data_idx'] = self._modellink._data_idx[i]
            data_idx.append(self._modellink._data_idx[i])

            # Check if this data point was present in previous iteration
            try:
                data_prev.append(self._data_idx[emul_i-1].index(data_idx[-1]))
            except ValueError:
                data_prev.append(None)
                data_set.attrs['data_prev'] = -1
            else:
                data_set.attrs['data_prev'] = data_prev[-1]

        # Close hdf5-file
        self._pipeline._close_hdf5(file)

        # Save model data arrays to memory
        self._data_val.append(data_val)
        self._data_err.append(data_err)
        self._data_idx.append(data_idx)
        self._data_prev.append(data_prev)

        # Logging
        logger.info("Finished preparing emulator iteration.")

        # Return the result
        return(reload)

    # This function constructs the emulator iteration emul_i
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _construct_iteration(self, emul_i):
        """
        Constructs the emulator iteration corresponding to the provided
        `emul_i`, by performing the given emulator method and calculating the
        prior expectation and variance values of the model evaluation samples.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        All data sets that are required to evaluate the emulator system.

        """

        # Save current time
        start_time = time()

        # Determine active parameters
        self._get_active_par(emul_i)

        # Check if regression is required
        if(self._method.lower() in ('regression', 'full')):
            self._do_regression(emul_i)

        # Calculate the prior expectation and variance values of sam_set
        self._get_prior_exp_sam_set(emul_i)
        self._get_cov_matrix(emul_i)

        # Set current emul_i to constructed emul_i
        self._emul_i = emul_i

        # Save time difference
        self._pipeline._save_statistics(emul_i, {
            'emul_construct_time': ['%.2f' % (time()-start_time), 's'],
            'MPI_comm_size_cons': ['%i' % (self._pipeline._size), '']})

    # This is function 'E_D(f(x'))'
    # This function gives the adjusted emulator expectation value back
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_adj_exp(self, emul_i, par_set, cov_vec):
        """
        Calculates the adjusted emulator expectation value at a given emulator
        iteration `emul_i` for specified parameter set `par_set` and
        corresponding covariance vector `cov_vec`.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the adjusted emulator
            expectation for.
        cov_vec : 2D :obj:`~numpy.ndarray` object
            Covariance vector corresponding to `par_set`.

        Returns
        -------
        adj_exp_val : 1D :obj:`~numpy.ndarray` object
            Adjusted emulator expectation value for every data point.

        """

        # Obtain prior expectation value of par_set
        prior_exp_par_set = self._get_prior_exp(emul_i, par_set)

        # Create empty adj_exp_val
        adj_exp_val = np.zeros(self._n_data[emul_i])

        # Calculate the adjusted emulator expectation value at given par_set
        for i in range(self._n_data[emul_i]):
            adj_exp_val[i] = prior_exp_par_set[i] +\
                np.dot(cov_vec[i].T, np.dot(self._cov_mat_inv[emul_i][i],
                       (self._mod_set[emul_i][i] -
                        self._prior_exp_sam_set[emul_i][i])))

        # Return it
        return(adj_exp_val)

    # This is function 'Var_D(f(x'))'
    # This function gives the adjusted emulator variance value back
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_adj_var(self, emul_i, par_set, cov_vec):
        """
        Calculates the adjusted emulator variance value at a given emulator
        iteration `emul_i` for specified parameter set `par_set` and
        corresponding covariance vector `cov_vec`.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the adjusted emulator
            variance for.
        cov_vec : 2D :obj:`~numpy.ndarray` object
            Covariance vector corresponding to `par_set`.

        Returns
        -------
        adj_var_val : 1D :obj:`~numpy.ndarray` object
            Adjusted emulator variance value for every data point.

        """

        # Obtain prior variance value of par_set
        prior_var_par_set = self._get_prior_var(emul_i, par_set)

        # Create empty adj_var_val
        adj_var_val = np.zeros(self._n_data[emul_i])

        # Calculate the adjusted emulator variance value at given par_set
        for i in range(self._n_data[emul_i]):
            adj_var_val[i] = prior_var_par_set[i] -\
                np.dot(cov_vec[i].T, np.dot(self._cov_mat_inv[emul_i][i],
                                            cov_vec[i]))

        # Return it
        return(adj_var_val)

    # This function evaluates the emulator at a given emul_i and par_set and
    # returns the adjusted expectation and variance values
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _evaluate(self, emul_i, par_set):
        """
        Evaluates the emulator system at iteration `emul_i` for given
        `par_set`.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to evaluate the emulator at.

        Returns
        -------
        adj_exp_val : 1D :obj:`~numpy.ndarray` object
            Adjusted emulator expectation value for every data point.
        adj_var_val : 1D :obj:`~numpy.ndarray` object
            Adjusted emulator variance value for every data point.

        """

        # Calculate the covariance vector for this par_set
        cov_vec = self._get_cov_vector(emul_i, par_set)

        # Calculate the adjusted expectation and variance values
        adj_exp_val = self._get_adj_exp(emul_i, par_set, cov_vec)
        adj_var_val = self._get_adj_var(emul_i, par_set, cov_vec)

        # Make sure that adj_var_val cannot drop below zero
        for i in range(self._n_data[emul_i]):
            if(adj_var_val[i] < 0):
                adj_var_val[i] = 0.0

        # Return adj_exp_val and adj_var_val
        return(adj_exp_val, adj_var_val)

    # This function extracts the set of active parameters
    # TODO: Write code cleaner, if possible
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_active_par(self, emul_i):
        """
        Determines the active parameters to be used for every individual data
        point in the provided emulator iteration `emul_i`. Uses backwards
        stepwise elimination to determine the set of active parameters.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        active_par : 1D :obj:`~numpy.ndarray` object
            Array containing the indices of all the parameters that are active
            in the emulator iteration `emul_i`.
        active_par_data : List of 1D :obj:`~numpy.ndarray` objects
            List containing the indices of all the parameters that are active
            in the emulator iteration `emul_i` for every individual data point.

        """

        # Log that active parameters are being determined
        logger = logging.getLogger('ACTIVE_PAR')
        logger.info("Determining active parameters.")

        # Initialize active parameter data sets
        active_par = SortedSet()
        active_par_data = [SortedSet() for _ in range(self._n_data[emul_i])]

        # Check if previously active parameters must be active again
        if self._pipeline._freeze_active_par:
            for i, j in enumerate(self._data_prev[emul_i]):
                if j is not None:
                    active_par_data[i].update(
                        self._active_par_data[emul_i-1][j])
                active_par.update(active_par_data[i])

        # Check if active parameters analysis has been requested
        if not self._pipeline._do_active_anal:
            # If not requested, then save all potentially active parameters
            active_par.update(self._pipeline._pot_active_par)
            for i in range(self._n_data[emul_i]):
                active_par_data[i].update(self._pipeline._pot_active_par)

        # If requested, perform a sequential backward stepwise regression
        else:
            # Determine active parameters for all data points
            for i in range(self._n_data[emul_i]):
                # Obtain frozen+potentially active parameters
                frz_pot_act_par = SortedSet(active_par_data[i])
                frz_pot_act_par.update(self._pipeline._pot_active_par)
                frz_pot_act_par = list(frz_pot_act_par)
                frz_pot_act_idx = list(range(len(frz_pot_act_par)))

                # Obtain non-frozen potentially active parameters
                non_frz_par = [j for j in self._pipeline._pot_active_par
                               if j not in active_par_data[i]]
                non_frz_idx = [frz_pot_act_par.index(j) for j in non_frz_par]

                # Obtain sam_set of frz_pot_act_par
                frz_pot_act_sam_set = self._sam_set[emul_i][:, frz_pot_act_par]

                # Obtain polynomial terms of frz_pot_act_sam_set
                pf_obj = PF(self._poly_order, include_bias=False)
                frz_pot_act_poly_terms =\
                    pf_obj.fit_transform(frz_pot_act_sam_set)

                # Create SequentialFeatureSelector object
                sfs_obj = SFS(LR(), k_features='parsimonious', forward=False,
                              floating=False, scoring='r2')

                # Perform linear regression with linear terms only
                sfs_obj.fit(frz_pot_act_sam_set, self._mod_set[emul_i][i])

                # Extract active parameters due to linear significance
                act_idx_lin = list(sfs_obj.k_feature_idx_)

                # Get passive non-frozen parameters in linear significance
                pas_idx_lin = [j for j in non_frz_idx if j not in act_idx_lin]

                # Make sure frozen parameters are considered active
                act_idx_lin = [j for j in frz_pot_act_idx
                               if j not in pas_idx_lin]
                act_idx = list(act_idx_lin)

                # Perform n-order polynomial regression for every passive par
                for j in pas_idx_lin:
                    # Obtain polynomial terms for this passive parameter
                    poly_idx = pf_obj.powers_[:, j] != 0
                    poly_idx[act_idx_lin] = 1
                    poly_idx = np.arange(len(poly_idx))[poly_idx]
                    poly_terms = frz_pot_act_poly_terms[:, poly_idx]

                    # Perform linear regression with addition of poly terms
                    sfs_obj.fit(poly_terms, self._mod_set[emul_i][i])

                    # Extract indices of active polynomial terms
                    act_idx_poly = poly_idx[list(sfs_obj.k_feature_idx_)]

                    # Check if any additional polynomial terms survived
                    # Add i to act_idx if this is the case
                    if np.any([k not in act_idx_lin for k in act_idx_poly]):
                        act_idx.append(j)

                # Update the active parameters for this data set
                active_par_data[i].update(np.array(frz_pot_act_par)[act_idx])

                # And update the unique active parameters for this iteration
                active_par.update(active_par_data[i])

                # Log the resulting active parameters
                logger.info("Active parameters for data point %s: %s"
                            % (i, [self._modellink._par_name[par]
                                   for par in active_par_data[i]]))

        # Convert active_par to a NumPy array
        active_par = np.array(list(active_par))

        # Convert active_par_data to a list of NumPy arrays
        for i in range(self._n_data[emul_i]):
            active_par_data[i] = np.array(list(active_par_data[i]))

        # Save the active parameters
        self._save_data(emul_i, {
            'active_par': [active_par, active_par_data]})

        # Log that active parameter determination is finished
        logger.info("Finished determining active parameters.")

    # This function performs a forward stepwise regression on sam_set
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _do_regression(self, emul_i):
        """
        Performs a forward stepwise linear regression on all model evaluation
        samples in emulator iteration `emul_i`. Calculates what the expectation
        values of all polynomial coefficients are. The polynomial order that is
        used in the regression depends on the `poly_order` parameter provided
        during class initialization.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        rsdl_var : 1D list
            List containing the residual variances of the regression function.
        regr_score : 1D list
            List containing the regression scores of the regression function.
        poly_coef : 2D list
            List containing the expectation values of the polynomial
            coefficients for all data points.
        poly_powers : 2D list
            List containing the powers every sample needs to be raised to, in
            order to obtain the polynomial terms used in the regression
            function.
        poly_idx : 2D list
            List containing the indices of the polynomial terms that are used
            in the regression function.
        poly_coef_cov : 2D list (if use_regr_cov is True)
            List containing the covariance values of the polynomial
            coefficients for all data points.

        """

        # Create logger
        logger = logging.getLogger('REGRESSION')
        logger.info("Performing regression.")

        # Create SequentialFeatureSelector object
        sfs_obj = SFS(LR(), k_features='best', forward=True, floating=True,
                      scoring='neg_mean_squared_error')

        # Create Pipeline object
        # The bias/intercept/constant-term is not included in the SFS object to
        # ensure that it is taken into account in the linear regression, since
        # it is required for getting the residual variance. It also ensures
        # that the SFS does not focus on the constant-term in its calculations.
        pipe = Pipeline_sk([('poly', PF(self._poly_order, include_bias=False)),
                            ('SFS', sfs_obj),
                            ('linear', LR())])

        # Create empty lists containing the various regression coefficients
        rsdl_var = []
        regr_score = []
        poly_coef = []
        poly_powers = []
        poly_idx = []
        if self._use_regr_cov:
            poly_coef_cov = []

        # Loop over all data points and perform a regression on all of them
        # TODO: Redetermine active parameters after regression process
        for i in range(self._n_data[emul_i]):
            # Extract active_sam_set
            active_sam_set = self._sam_set[emul_i][
                    :, self._active_par_data[emul_i][i]]

            # Perform regression on this data point
            pipe.fit(active_sam_set, self._mod_set[emul_i][i])

            # Obtain the corresponding polynomial indices
            poly_idx_temp = np.array(pipe.named_steps['SFS'].k_feature_idx_)

            # Extract sam_set_poly
            sam_set_poly = pipe.named_steps['poly'].transform(
                active_sam_set)[:, poly_idx_temp]

            # Extract the residual variance
            rsdl_var.append(mse(self._mod_set[emul_i][i],
                                pipe.named_steps['linear'].predict(
                                        sam_set_poly)))

            # Log the score of the regression process
            regr_score.append(pipe.named_steps['linear'].score(
                              sam_set_poly, self._mod_set[emul_i][i]))
            logger.info("Regression score for data point %s: %s."
                        % (i, regr_score[i]))

            # Add the intercept term to sam_set_poly
            sam_set_poly = np.concatenate([np.ones([self._n_sam[emul_i], 1]),
                                           sam_set_poly], axis=-1)

            # Calculate the poly_coef covariances
            if self._use_regr_cov:
                poly_coef_cov.append(rsdl_var[i]*inv(
                        np.dot(sam_set_poly.T, sam_set_poly)).flatten())

            # Obtain polynomial powers and include intercept term
            poly_powers_temp = pipe.named_steps['poly'].powers_[poly_idx_temp]
            poly_powers_intercept = [[0]*len(self._active_par_data[emul_i][i])]
            poly_powers.append(np.concatenate([poly_powers_intercept,
                                               poly_powers_temp]))

            # Add intercept term to polynomial indices
            poly_idx_temp += 1
            poly_idx.append(np.concatenate([[0], poly_idx_temp]))

            # Obtain polynomial coefficients and include intercept term
            poly_coef_temp = pipe.named_steps['linear'].coef_
            poly_coef_intercept = [pipe.named_steps['linear'].intercept_]
            poly_coef.append(np.concatenate([poly_coef_intercept,
                                             poly_coef_temp]))

        # Save everything to hdf5
        if self._use_regr_cov:
            self._save_data(emul_i, {
                'regression': [rsdl_var, regr_score, poly_coef, poly_powers,
                               poly_idx, poly_coef_cov]})
        else:
            self._save_data(emul_i, {
                'regression': [rsdl_var, regr_score, poly_coef, poly_powers,
                               poly_idx]})

        # Log that this is finished
        logger.info("Finished performing regression.")

    # This function gives the prior expectation value
    # This is function 'E(f(x'))' or 'u(x')'
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_prior_exp(self, emul_i, par_set):
        """
        Calculates the prior expectation value at a given emulator iteration
        `emul_i` for specified parameter set `par_set`. This expectation
        depends on the emulator method used.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object or None
            If *None*, calculate the prior expectation values of sam_set.
            If not *None*, calculate the prior expectation value for the given
            model parameter value set.

        Returns
        -------
        prior_exp : 1D or 2D :obj:`~numpy.ndarray` object
            Prior expectation values for either sam_set or `par_set` for every
            data point.

        """

        # If prior_exp of sam_set is requested (prior_exp_sam_set)
        if par_set is None:
            # Initialize empty prior expectation
            prior_exp = np.zeros([self._n_data[emul_i], self._n_sam[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Gaussian prior expectation is equal to the mean, which is 0
                prior_exp += 0
            if self._method.lower() in ('regression', 'full'):
                for i in range(self._n_data[emul_i]):
                    # Initialize PF object
                    pf_obj = PF(self._poly_order)

                    # Obtain the polynomial terms
                    poly_terms = pf_obj.fit_transform(
                        self._sam_set[emul_i][
                            :, self._active_par_data[emul_i][i]])[
                                :, self._poly_idx[emul_i][i]]
                    prior_exp[i] += np.sum(
                        self._poly_coef[emul_i][i]*poly_terms, axis=-1)

        # If prior_exp of par_set is requested (adj_exp)
        else:
            # Initialize empty prior expectation
            prior_exp = np.zeros(self._n_data[emul_i])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Gaussian prior expectation is equal to the mean, which is 0
                prior_exp += 0
            if self._method.lower() in ('regression', 'full'):
                for i in range(self._n_data[emul_i]):
                    poly_terms = np.product(pow(par_set[
                        self._active_par_data[emul_i][i]],
                        self._poly_powers[emul_i][i]), axis=-1)
                    prior_exp[i] += np.sum(
                        self._poly_coef[emul_i][i]*poly_terms, axis=-1)

        # Return it
        return(prior_exp)

    # This function calculates the prior expectation and variances values
    # This is function 'E(D)'
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_prior_exp_sam_set(self, emul_i):
        """
        Calculates the prior expectation values at a given emulator iteration
        `emul_i` for all model evaluation samples and saves it for later use.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        prior_exp_sam_set : 2D :obj:`~numpy.ndarray` object
            2D array containing the prior expectation values for all model
            evaluation samples for all data points. Has the same shape as the
            array with model outputs.

        """

        # Create logger
        logger = logging.getLogger('PRIOR_EXP')
        logger.info("Calculating prior expectation values for "
                    "known samples at emulator iteration %s." % (emul_i))

        # Obtain prior expectation value of sam_set
        prior_exp_sam_set = self._get_prior_exp(emul_i, None)

        # Save the prior expectation values to hdf5
        self._save_data(emul_i, {
            'prior_exp_sam_set': prior_exp_sam_set})

        # Log again
        logger.info("Finished calculating prior expectation values.")

    # This function gives the prior variance value
    # This is function 'Var(f(x'))'
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_prior_var(self, emul_i, par_set):
        """
        Calculates the prior variance value at a given emulator iteration
        `emul_i` for specified parameter set `par_set`. This variance depends
        on the emulator method used.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the prior variance for.

        Returns
        -------
        prior_var : 1D :obj:`~numpy.ndarray` object
            Prior variance value for every data point.

        """

        # Return it
        return(self._get_cov(emul_i, par_set, par_set))

    # This is function 'Cov(f(x), f(x'))' or 'k(x,x')
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_cov(self, emul_i, par_set1, par_set2):
        """
        Calculates the covariance between given model parameter value sets
        `par_set1` and `par_set2`. This covariance depends on the emulator
        method used.

        Parameters
        ----------
        %(emul_i)s
        par_set1, par_set2 : 1D :obj:`~numpy.ndarray` object or None
            If par_set1 and par_set2 are both not *None*, calculate covariance
            between par_set1 and par_set2.
            If par_set1 is not *None* and par_set2 is *None*, calculate
            covariances between par_set1 and sam_set (covariance vector).
            If par_set1 is *None*, calculate covariances between sam_set and
            sam_set (covariance matrix).
            When not *None*, par_set is the model parameter value set to
            calculate the covariance for.

        Returns
        -------
        cov : 1D, 2D or 3D :obj:`~numpy.ndarray` object
            Depending on the arguments provided, a covariance value, vector or
            matrix for all data points.

        """

        # Value for fraction of residual variance for variety in inactive pars
        weight = [1-len(active_par)/self._modellink._n_par
                  for active_par in self._active_par_data[emul_i]]

        # Determine which residual variance should be used
        if self._method.lower() in ('regression', 'full'):
            rsdl_var = self._rsdl_var[emul_i]
#            rsdl_var = pow(self._sigma, 2)
        elif(self.method.lower() == 'gaussian'):
            rsdl_var = pow(self._sigma, 2)

        # If cov of sam_set with sam_set is requested (cov_mat)
        if par_set1 is None:
            # Calculate covariance between sam_set and sam_set
            cov = np.zeros([self._n_data[emul_i], self._n_sam[emul_i],
                            self._n_sam[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Obtain the difference between sam_set and sam_set
                diff_sam_set = diff(self._sam_set[emul_i], flatten=False)

                # If Gaussian needs to be taken into account
                for i in range(self._n_data[emul_i]):
                    # Gaussian variance
                    cov[i] += (1-weight[i])*rsdl_var[i] *\
                        np.exp(-1*pow(norm(
                            diff_sam_set[:, :,
                                         self._active_par_data[emul_i][i]],
                            axis=-1), 2) /
                                pow(norm(
                                    self._l_corr[
                                        self._active_par_data[emul_i][i]]), 2))

                    # Inactive parameter variety
                    cov[i] += weight[i]*rsdl_var[i]*np.eye(self._n_sam[emul_i])

            if(self._method.lower() in ('regression', 'full') and
               self._use_regr_cov):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, None, None)

        # If cov of par_set1 with sam_set is requested (cov_vec)
        elif par_set2 is None:
            # Calculate covariance between par_set1 and sam_set
            cov = np.zeros([self._n_data[emul_i], self._n_sam[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Obtain the difference between par_set1 and sam_set
                diff_sam_set = par_set1-self._sam_set[emul_i]

                # If Gaussian needs to be taken into account
                for i in range(self._n_data[emul_i]):
                    # Gaussian variance
                    cov[i] += (1-weight[i])*rsdl_var[i] *\
                        np.exp(-1*pow(norm(
                            diff_sam_set[:, self._active_par_data[emul_i][i]],
                            axis=-1), 2) /
                                pow(norm(
                                    self._l_corr[
                                        self._active_par_data[emul_i][i]]), 2))

                    # Inactive parameter variety
                    cov[i] += weight[i]*rsdl_var[i] *\
                        (par_set1 == self._sam_set[emul_i]).all(axis=-1)

            if(self._method.lower() in ('regression', 'full') and
               self._use_regr_cov):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, par_set1, None)

        # If cov of par_set1 with par_set2 is requested (cov)
        else:
            # Calculate covariance between par_set1 and par_set2
            cov = np.zeros([self._n_data[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Obtain the difference between par_set1 and par_set2
                diff_sam_set = par_set1-par_set2

                # If Gaussian needs to be taken into account
                for i in range(self._n_data[emul_i]):
                    # Gaussian variance
                    cov[i] += (1-weight[i])*rsdl_var[i] *\
                        np.exp(-1*pow(norm(
                            diff_sam_set[self._active_par_data[emul_i][i]],
                            axis=-1), 2) /
                                pow(norm(
                                    self._l_corr[
                                        self._active_par_data[emul_i][i]]), 2))

                    # Inactive parameter variety
                    cov[i] += weight[i]*rsdl_var[i] *\
                        (par_set1 == par_set2).all()
            if(self._method.lower() in ('regression', 'full') and
               self._use_regr_cov):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, par_set1, par_set2)

        # Return it
        return(cov)

    # This function calculates the regression covariance.
    # This is function 'Cov(r(x), r(x'))'
    # OPTIMIZE: Takes roughly 45-50% of total evaluation time
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_regr_cov(self, emul_i, par_set1, par_set2):
        """
        Calculates the covariance of the regression function at emulator
        iteration `emul_i` for given parameter sets `par_set1` and `par_set2`.

        Parameters
        ----------
        %(emul_i)s
        par_set1, par_set2 : 1D :obj:`~numpy.ndarray` object or None
            If par_set1 and par_set2 are both not *None*, calculate regression
            covariance values for par_set1 with par_set2.
            If par_set1 is not *None* and par_set2 is *None*, calculate
            regression covariance values for par_set1 with sam_set
            (covariance vector).
            If par_set1 and par_set2 are both *None*, calculate regression
            covariance values for sam_set (covariance matrix).
            When not *None*, par_set is the model parameter value set to
            calculate the regression covariance values for.

        Returns
        -------
        regr_cov : 1D, 2D or 3D :obj:`~numpy.ndarray` object
            Depending on the arguments provided, a regression covariance
            value, vector or matrix for all data points.

        """

        # If regr_cov of sam_set is requested (cov_mat)
        if par_set1 is None:
            # Make empty array
            regr_cov = np.zeros([self._n_data[emul_i], self._n_sam[emul_i],
                                 self._n_sam[emul_i]])

            for i in range(self._n_data[emul_i]):
                # Initialize PF object
                pf_obj = PF(self._poly_order)

                # Obtain the polynomial terms for both parameter sets
                poly_terms1 = pf_obj.fit_transform(
                    self._sam_set[emul_i][
                        :, self._active_par_data[emul_i][i]])[
                            :, self._poly_idx[emul_i][i]]
                poly_terms2 = poly_terms1

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] = np.sum(self._poly_coef_cov[emul_i][i]*prod_terms,
                                     axis=-1).reshape([self._n_sam[emul_i],
                                                       self._n_sam[emul_i]])

        # If regr_cov of par_set with sam_set is requested (cov_vec)
        elif par_set2 is None:
            # Make empty array
            regr_cov = np.zeros([self._n_data[emul_i], self._n_sam[emul_i]])

            for i in range(self._n_data[emul_i]):
                # Initialize PF object
                pf_obj = PF(self._poly_order)

                # Obtain the polynomial terms for both parameter sets
                poly_terms1 =\
                    np.product(pow(par_set1[self._active_par_data[emul_i][i]],
                                   self._poly_powers[emul_i][i]),
                               axis=-1)
                poly_terms2 = pf_obj.fit_transform(
                    self._sam_set[emul_i][
                        :, self._active_par_data[emul_i][i]])[
                            :, self._poly_idx[emul_i][i]]

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] = np.sum(self._poly_coef_cov[emul_i][i]*prod_terms,
                                     axis=-1)

        # If regr_cov of par_set1 with par_set2 is requested (cov)
        else:
            # Make empty array
            regr_cov = np.zeros([self._n_data[emul_i]])

            for i in range(self._n_data[emul_i]):
                # Obtain the polynomial terms for both parameter sets
                poly_terms1 =\
                    np.product(pow(par_set1[self._active_par_data[emul_i][i]],
                                   self._poly_powers[emul_i][i]),
                               axis=-1)
                poly_terms2 =\
                    np.product(pow(par_set2[self._active_par_data[emul_i][i]],
                                   self._poly_powers[emul_i][i]),
                               axis=-1)

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] = np.sum(self._poly_coef_cov[emul_i][i]*prod_terms,
                                     axis=-1)

        # Return it
        return(regr_cov)

    # This is function 'Cov(f(x'), D)' or 't(x')'
    # HINT: Calculate cov_vec for all samples at once to save time?
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_cov_vector(self, emul_i, par_set):
        """
        Calculates the column vector of covariances between given (`par_set`)
        and known ('sam_set') model parameter value sets for a given emulator
        iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the covariances vector
            for.

        Returns
        -------
        cov_vec : 2D :obj:`~numpy.ndarray` object
            Column vector containing the covariances between given and
            known model parameter value sets for every data point.

        """

        # Calculate covariance vector
        cov_vec = self._get_cov(emul_i, par_set, None)

        # Return it
        return(cov_vec)

    # This is function 'Var(D)' or 'A'
    # Reminder that this function should only be called once per sample set
    # OPTIMIZE: Look into using GPUs for calculating and inverting cov_mat
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_cov_matrix(self, emul_i):
        """
        Calculates the (inverse) matrix of covariances between known model
        evaluation samples for a given emulator iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        cov_mat : 3D :obj:`~numpy.ndarray` object
            Matrix containing the covariances between all known model
            evaluation samples for every data point.
        cov_mat_inv : 3D :obj:`~numpy.ndarray` object
            Inverse of covariance matrix for every data point.

        """

        # Log the creation of the covariance matrix
        logger = logging.getLogger('COV_MAT')
        logger.info("Calculating covariance matrix for emulator iteration %s."
                    % (emul_i))

        # Calculate covariance matrix
        # Since this calculation can cause memory issues, catch error and try
        # slower but less memory-intensive method
        try:
            cov_mat = self._get_cov(emul_i, None, None)
        except MemoryError:
            # For some odd reason, overwriting cov_mat does not clear memory,
            # and the only way to remove cov_mat from it is by attempting to
            # delete it. It always raises a NameError, but does actually clear
            # memory successfully. I have no idea why this happens this way.
            # TODO: Find out why and improve the memory clearance
            try:
                del cov_mat
            except NameError:
                pass
            cov_mat = np.zeros([self._n_data[emul_i], self._n_sam[emul_i],
                                self._n_sam[emul_i]])
            for i in range(self._n_sam[emul_i]):
                cov_mat[:, i] = self._get_cov(emul_i, self._sam_set[emul_i][i],
                                              None)

        # Make empty array of inverse covariance matrices
        cov_mat_inv = np.zeros_like(cov_mat)

        # Loop over all data points
        for i in range(self._n_data[emul_i]):
            # Make sure that cov_mat is symmetric positive-definite by
            # finding the nearest one
            cov_mat[i] = nearest_PD(cov_mat[i])

            # Calculate the inverse of the covariance matrix
            logger.info("Calculating inverse of covariance matrix %s."
                        % (i))
#            cov_mat_inv[i] = nearest_PD(self._get_inv_matrix(cov_mat[i]))
            # TODO: Maybe I should put an error catch for memory overflow here?
            cov_mat_inv[i] = self._get_inv_matrix(cov_mat[i])

        # Save the covariance matrices to hdf5
        self._save_data(emul_i, {
            'cov_mat': [cov_mat, cov_mat_inv]})

        # Log that calculation has been finished
        logger.info("Finished calculating covariance matrix.")

    # This function calculates the inverse of a given matrix
    def _get_inv_matrix(self, matrix):
        """
        Calculates the inverse of a given matrix `matrix`.
        Right now only uses the :func:`~numpy.linalg.inv` function.

        Parameters
        ----------
        matrix : 2D array_like
            Matrix to be inverted.

        Returns
        -------
        matrix_inv : 2D :obj:`~numpy.ndarray` object
            Inverse of the given matrix `matrix`.

        """

        # Calculate the inverse of the given matrix
        matrix_inv = inv(matrix)

        # Return it
        return(matrix_inv)

    # Load the emulator
    def _load_emulator(self, modellink_obj):
        """
        Checks if the provided HDF5-file contains a constructed emulator system
        and loads in the emulator data accordingly.

        Parameters
        ----------
        modellink_obj : :obj:`~ModelLink` object
            Instance of the :class:`~ModelLink` class that links the emulated
            model to this :obj:`~Pipeline` object.

        """

        # Make logger
        logger = logging.getLogger('EMUL_LOAD')
        logger.info("Loading emulator system.")

        # Check if an existing hdf5-file is provided
        try:
            logger.info("Checking if provided emulator file '%s' is a "
                        "constructed emulator system."
                        % (self._pipeline._hdf5_file))
            file = self._pipeline._open_hdf5('r')
        except (OSError, IOError):
            # No existing emulator was provided
            logger.info("Non-existing HDF5-file provided.")
            self._emul_load = 0
            self._emul_i = 0

            # No emulator provided, so no loaded modellink either
            modellink_loaded = None
        else:
            # Existing emulator was provided
            logger.info("Constructed emulator HDF5-file provided.")
            self._emul_load = 1

            # Obtain the number of emulator iterations constructed
            self._emul_i = len(file.keys())

            # Close hdf5-file
            self._pipeline._close_hdf5(file)

            # Read all emulator parameters from the hdf5-file
            modellink_loaded = self._retrieve_parameters()

        # Load emulator data
        self._load_data(self._emul_i)

        # Link the provided ModelLink object to the pipeline
        self._set_modellink(modellink_obj, modellink_loaded)

        # Logging
        logger.info("Finished loading emulator system.")

    # This function connects the provided ModelLink class to the pipeline
    def _set_modellink(self, modellink_obj, modellink_loaded):
        """
        Sets the :obj:`~ModelLink` object that will be used for constructing
        this emulator system. If a constructed emulator system is present,
        checks if provided `modellink` argument matches the :class:`~ModelLink`
        subclass used to construct it.

        Parameters
        ----------
        modellink_obj : :obj:`~ModelLink` object
            Instance of the :class:`~ModelLink` class that links the emulated
            model to this :obj:`~Pipeline` object.
            The provided :obj:`~ModelLink` object must match the one used to
            construct the loaded emulator system.
        modellink_loaded : str or None
            If str, the name of the :class:`~ModelLink` subclass that was used
            to construct the loaded emulator system.
            If *None*, no emulator system is loaded.

        """

        # Logging
        logger = logging.getLogger('INIT')
        logger.info("Setting ModelLink object.")

        # Check if a subclass of the ModelLink class has been provided
        if not isinstance(modellink_obj, ModelLink):
            logger.error("Input argument 'modellink' must be an instance of "
                         "the ModelLink class!")
            raise TypeError("Input argument 'modellink' must be an instance "
                            "of the ModelLink class!")

        # If no existing emulator system is loaded, pass
        if modellink_loaded is None:
            logger.info("No constructed emulator system is loaded.")
            # Set ModelLink object
            self._modellink = modellink_obj
            self._pipeline._modellink = self._modellink

        # If an existing emulator system is loaded, check if classes are equal
        elif(modellink_obj._name == modellink_loaded):
            logger.info("Provided ModelLink subclass matches ModelLink "
                        "subclass used for emulator construction.")
            # Set ModelLink object for Emulator
            self._modellink = modellink_obj

            # If mock data has been used, set the ModelLink object to use it
            if self._use_mock:
                self._set_mock_data()

            # Set ModelLink object for Pipeline
            self._pipeline._modellink = self._modellink
        else:
            logger.error("Provided ModelLink subclass '%s' does not match "
                         "the ModelLink subclass '%s' used for emulator "
                         "construction!"
                         % (modellink_obj._name, modellink_loaded))
            raise InputError("Provided ModelLink subclass '%s' does not "
                             "match the ModelLink subclass '%s' used for "
                             "emulator construction!"
                             % (modellink_obj._name, modellink_loaded))

        # Send updated modellink object to workers
        for rank in range(1, self._pipeline._size):
            MPI.COMM_WORLD.send(self._modellink, dest=rank, tag=888+rank)

        # Logging
        logger.info("ModelLink object set to '%s'." % (self._modellink._name))

    # Function that loads in the emulator data
    # TODO: Write code that allows part of the data to be loaded in (crashing)
    # and forces the pipeline to continue where data starts missing
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _load_data(self, emul_i):
        """
        Loads in all the important emulator data up to emulator iteration
        `emul_i` into memory.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        All relevant emulator data up to emulator iteration `emul_i` is loaded
        into memory.

        """

        # Set the logger
        logger = logging.getLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing emulator data sets.")
        self._n_sam = [[]]
        self._sam_set = [[]]
        self._active_par = [[]]
        self._mod_set = [[]]
        self._cov_mat_inv = [[]]
        self._prior_exp_sam_set = [[]]
        self._active_par_data = [[]]
        self._rsdl_var = [[]]
        self._poly_coef = [[]]
        self._poly_coef_cov = [[]]
        self._poly_powers = [[]]
        self._poly_idx = [[]]
        self._n_data = [[]]
        self._data_val = [[]]
        self._data_err = [[]]
        self._data_idx = [[]]
        self._data_prev = [[]]

        # If no file has been provided
        if(emul_i == 0 or self._emul_load == 0):
            logger.info("Non-existent emulator file provided. No additional "
                        "data needs to be loaded.")
            return

        # Check if requested emulator iteration exists
        elif not(1 <= emul_i <= self._emul_i):
            logger.error("Requested emulator iteration %s does not exist!"
                         % (emul_i))
            raise RequestError("Requested emulator iteration %s does not "
                               "exist!" % (emul_i))

        # Load emulator data from construction file
        elif(self._emul_load == 1):
            # Load the corresponding sam_set, mod_set and cov_mat_inv
            logger.info("Loading relevant emulator data up to iteration %s."
                        % (emul_i))

            # Open hdf5-file
            file = self._pipeline._open_hdf5('r')

            # Read in the data
            for i in range(1, emul_i+1):
                self._n_sam.append(file['%s' % (i)].attrs['n_sam'])
                self._sam_set.append(file['%s/sam_set' % (i)][()])
                self._active_par.append(file['%s/active_par' % (i)][()])
                mod_set = []
                cov_mat_inv = []
                prior_exp_sam_set = []
                active_par_data = []
                self._n_data.append(file['%s' % (i)].attrs['n_data'])
                data_val = []
                data_err = []
                data_idx = []
                data_prev = []
                for j in range(self._n_data[i]):
                    data_set = file['%s/data_point_%s' % (i, j)]
                    mod_set.append(data_set['mod_set'][()])
                    cov_mat_inv.append(data_set['cov_mat_inv'][()])
                    prior_exp_sam_set.append(data_set['prior_exp_sam_set'][()])
                    active_par_data.append(data_set['active_par_data'][()])
                    data_val.append(data_set.attrs['data_val'])
                    data_err.append(data_set.attrs['data_err'])

                    # Read in all data_idx parts and combine them
                    idx_keys = [key for key in data_set.attrs.keys()
                                if key[:8] == 'data_idx']
                    idx_len = len(idx_keys)
                    if(idx_len == 1):
                        data_idx.append(data_set.attrs['data_idx'])
                    else:
                        tmp_data_idx = []
                        for key in idx_keys:
                            if isinstance(data_set.attrs[key], bytes):
                                idx_str = data_set.attrs[key].decode('utf-8')
                                tmp_data_idx.append(idx_str)
                            else:
                                tmp_data_idx.append(data_set.attrs[key])
                        data_idx.append(tmp_data_idx)

                    # Obtain data_prev
                    if(data_set.attrs['data_prev'] == -1):
                        data_prev.append(None)
                    else:
                        data_prev.append(data_set.attrs['data_prev'])

                self._mod_set.append(mod_set)
                self._cov_mat_inv.append(cov_mat_inv)
                self._prior_exp_sam_set.append(prior_exp_sam_set)
                self._active_par_data.append(active_par_data)
                self._data_val.append(data_val)
                self._data_err.append(data_err)
                self._data_idx.append(data_idx)
                self._data_prev.append(data_prev)

                if self._method.lower() in ('regression', 'full'):
                    rsdl_var = []
                    poly_coef = []
                    if self._use_regr_cov:
                        poly_coef_cov = []
                    poly_powers = []
                    poly_idx = []
                    for j in range(self._n_data[i]):
                        data_set = file['%s/data_point_%s' % (i, j)]
                        rsdl_var.append(data_set.attrs['rsdl_var'])
                        poly_coef.append(data_set['poly_coef'][()])
                        if self._use_regr_cov:
                            poly_coef_cov.append(data_set['poly_coef_cov'][()])
                        poly_powers.append(data_set['poly_powers'][()])
                        poly_idx.append(data_set['poly_idx'][()])
                    self._rsdl_var.append(rsdl_var)
                    self._poly_coef.append(poly_coef)
                    if self._use_regr_cov:
                        self._poly_coef_cov.append(poly_coef_cov)
                    self._poly_powers.append(poly_powers)
                    self._poly_idx.append(poly_idx)

            # Close hdf5-file
            self._pipeline._close_hdf5(file)

            # Log that loading is finished
            logger.info("Finished loading relevant emulator data.")
        else:
            raise RequestError("Invalid operation requested!")

    # This function saves emulator data to hdf5
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _save_data(self, emul_i, data_dict):
        """
        Saves a given data dict {`keyword`: `data`} at the given emulator
        iteration `emul_i` to the HDF5-file and as an data attribute to the
        current :obj:`~Emulator` instance.

        Parameters
        ----------
        %(emul_i)s

        Dict Variables
        --------------
        keyword : {'active_par', 'cov_mat', 'mod_real_set',\
                   'prior_exp_sam_set', 'regression'}
            String specifying the type of data that needs to be saved.
        data : int, float, list
            The actual data that needs to be saved at data keyword `keyword`.

        Generates
        ---------
        The specified data is saved to the HDF5-file.

        """

        # Do some logging
        logger = logging.getLogger('SAVE_DATA')

        # Open hdf5-file
        file = self._pipeline._open_hdf5('r+')

        # Loop over entire provided data dict
        for keyword, data in data_dict.items():
            # Log what data is being saved
            logger.info("Saving %s data at iteration %s to HDF5."
                        % (keyword, emul_i))

            # Check what data keyword has been provided
            # ACTIVE PARAMETERS
            if(keyword == 'active_par'):
                file.create_dataset('%s/active_par' % (emul_i), data=data[0])
                for i in range(self._n_data[emul_i]):
                    data_set = file['%s/data_point_%s' % (emul_i, i)]
                    data_set.create_dataset('active_par_data', data=data[1][i])
                self._active_par.append(data[0])
                self._active_par_data.append(data[1])

            # COV_MAT
            elif(keyword == 'cov_mat'):
                for i in range(self._n_data[emul_i]):
                    data_set = file['%s/data_point_%s' % (emul_i, i)]
                    data_set.create_dataset('cov_mat', data=data[0][i])
                    data_set.create_dataset('cov_mat_inv', data=data[1][i])
                self._cov_mat_inv.append(data[1])

            # MOD_REAL_SET
            elif(keyword == 'mod_real_set'):
                file.create_dataset('%s/sam_set' % (emul_i), data=data[0])
                file['%s' % (emul_i)].attrs['n_sam'] = np.shape(data[0])[0]
                self._sam_set.append(data[0])
                self._n_sam.append(np.shape(data[0])[0])

                for i in range(self._n_data[emul_i]):
                    data_set = file['%s/data_point_%s' % (emul_i, i)]
                    data_set.create_dataset('mod_set', data=data[1][i])
                self._mod_set.append(data[1])

                file['%s' % (emul_i)].attrs['use_ext_real_set'] = bool(data[2])

            # PRIOR_EXP_SAM_SET
            elif(keyword == 'prior_exp_sam_set'):
                for i in range(self._n_data[emul_i]):
                    data_set = file['%s/data_point_%s' % (emul_i, i)]
                    data_set.create_dataset('prior_exp_sam_set', data=data[i])
                self._prior_exp_sam_set.append(data)

            # REGRESSION
            elif(keyword == 'regression'):
                for i in range(self._n_data[emul_i]):
                    data_set = file['%s/data_point_%s' % (emul_i, i)]
                    data_set.attrs['rsdl_var'] = data[0][i]
                    data_set.attrs['regr_score'] = data[1][i]
                    data_set.create_dataset('poly_coef', data=data[2][i])
                    data_set.create_dataset('poly_powers', data=data[3][i])
                    data_set.create_dataset('poly_idx', data=data[4][i])
                    if self._use_regr_cov:
                        data_set.create_dataset('poly_coef_cov',
                                                data=data[5][i])
                self._rsdl_var.append(data[0])
                self._poly_coef.append(data[2])
                self._poly_powers.append(data[3])
                self._poly_idx.append(data[4])
                if self._use_regr_cov:
                    self._poly_coef_cov.append(data[5])

            # INVALID KEYWORD
            else:
                logger.error("Invalid keyword argument provided!")
                raise ValueError("Invalid keyword argument provided!")

        # Close hdf5-file
        self._pipeline._close_hdf5(file)

        # More logging
        logger.info("Finished saving data to HDF5.")

    # Read in the emulator attributes
    def _retrieve_parameters(self):
        """
        Reads in the emulator parameters from the provided HDF5-file and saves
        them in the current :obj:`~Emulator` instance.

        """

        # Log that parameters are being read
        logger = logging.getLogger('INIT')
        logger.info("Retrieving emulator parameters from provided HDF5-file.")

        # Open hdf5-file
        file = self._pipeline._open_hdf5('r')

        # Read in all the emulator parameters
        self._sigma = file.attrs['sigma']
        self._l_corr = file.attrs['l_corr']
        self._method = file.attrs['method'].decode('utf-8')
        self._use_regr_cov = int(file.attrs['use_regr_cov'])
        self._poly_order = file.attrs['poly_order']
        modellink_name = file.attrs['modellink_name'].decode('utf-8')
        self._use_mock = int(file.attrs['use_mock'])

        # TODO: This try-statement becomes obsolete when PRISM is released
        try:
            emul_version = file.attrs['prism_version'].decode('utf-8')
        except KeyError:
            emul_version = '0.3.0'

        # TODO: Same for this try-statement
        try:
            emul_type = file.attrs['emul_type'].decode('utf-8')
        except KeyError:
            emul_type = 'default'

        # Check if provided emulator is the same as requested
        if(emul_type != self._emul_type):
            logger.error("Provided emulator system type ('%s') does not match "
                         "the requested type ('%s')!"
                         % (emul_type, self._emul_type))
            raise RequestError("Provided emulator system type ('%s') does not "
                               "match the requested type ('%s')!"
                               % (emul_type, self._emul_type))

        # Close hdf5-file
        self._pipeline._close_hdf5(file)

        # Check if provided emul_version is compatible
        check_compatibility(emul_version)

        # Log that reading is finished
        logger.info("Finished retrieving parameters.")

        # Return the name of the modellink class used to construct the loaded
        # emulator system
        return(modellink_name)

    # This function automatically loads default emulator parameters
    def _get_default_parameters(self):
        """
        Generates a dict containing default values for all emulator parameters.

        Returns
        -------
        par_dict : dict
            Dict containing all default emulator parameter values.

        """

        # Log this
        logger = logging.getLogger('INIT')
        logger.info("Generating default emulator parameter dict.")

        # Create parameter dict with default parameters
        par_dict = {'sigma': '0.8',
                    'l_corr': '0.3',
                    'method': "'full'",
                    'use_regr_cov': 'False',
                    'poly_order': '3',
                    'use_mock': 'False'}

        # Log end
        logger.info("Finished generating default emulator parameter dict.")

        # Return it
        return(par_dict)

    # Read in the parameters from the provided parameter file
    def _read_parameters(self):
        """
        Reads in the emulator parameters from the provided PRISM parameter file
        saves them in the current :obj:`~Emulator` instance.

        """

        # Log that the PRISM parameter file is being read
        logger = logging.getLogger('INIT')
        logger.info("Reading emulator parameters.")

        # Obtaining default emulator parameter dict
        par_dict = self._get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._pipeline._prism_file is not None:
            emul_par = np.genfromtxt(self._pipeline._prism_file, dtype=(str),
                                     delimiter=':', autostrip=True)

            # Make sure that emul_par is 2D
            emul_par = np.array(emul_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(emul_par)

        # More logging
        logger.info("Checking compatibility of provided emulator parameters.")

        # GENERAL
        # Gaussian sigma
        self._sigma = check_nzero_float(float(par_dict['sigma']), 'sigma')

        # Gaussian correlation length
        self._l_corr = check_pos_float(float(par_dict['l_corr']), 'l_corr') *\
            abs(self._modellink._par_rng[:, 1]-self._modellink._par_rng[:, 0])

        # Method used to calculate emulator functions
        # Future will include 'gaussian', 'regression', 'auto' and 'full'
        self._method = str(par_dict['method']).replace("'", '')
        if self._method.lower() in ('gaussian', 'regression', 'full'):
            pass
        elif(self._method.lower() == 'auto'):
            raise NotImplementedError
        else:
            logger.error("Input argument 'method' is invalid! (%s)"
                         % (self._method.lower()))
            raise ValueError("Input argument 'method' is invalid! (%s)"
                             % (self._method.lower()))

        # Obtain the bool determining whether or not to use regr_cov
        self._use_regr_cov = check_bool(par_dict['use_regr_cov'],
                                        'use_regr_cov')

        # Obtain the polynomial order for the regression selection process
        self._poly_order = check_pos_int(int(par_dict['poly_order']),
                                         'poly_order')

        # Obtain the bool determining whether or not to use mock data
        self._use_mock = check_bool(par_dict['use_mock'], 'use_mock')

        # Log that reading has been finished
        logger.info("Finished reading emulator parameters.")

    # This function loads previously generated mock data into ModelLink
    # TODO: Allow user to add/remove mock data? Requires consistency check
    def _set_mock_data(self):
        """
        Loads previously used mock data into the :class:`~ModelLink` object,
        overwriting the parameter estimates, data values, data errors and data
        identifiers with their mock equivalents.

        Generates
        ---------
        Overwrites the corresponding :class:`~ModelLink` class properties with
        the previously used values.

        """

        # Start logger
        logger = logging.getLogger('MOCK_DATA')

        # Overwrite ModelLink properties with previously generated values
        # Log that mock_data is being loaded in
        logger.info("Loading previously used mock data into ModelLink object.")

        # Open hdf5-file
        file = self._pipeline._open_hdf5('r')

        # Overwrite ModelLink properties
        self._modellink._par_est = file.attrs['mock_par'].tolist()
        self._modellink._n_data = self._n_data[1]
        self._modellink._data_val = self._data_val[1]
        self._modellink._data_err = self._data_err[1]
        self._modellink._data_idx = self._data_idx[1]

        # Close hdf5-file
        self._pipeline._close_hdf5(file)

        # Log end
        logger.info("Loaded mock data.")
