# -*- coding: utf-8 -*-

"""
Emulator
========

"""

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from itertools import chain, combinations
import os
from os import path
from time import strftime, strptime

# Package imports
from e13tools import InputError, ShapeError
from e13tools.math import diff, nearest_PD
from e13tools.pyplot import draw_textline
from e13tools.sampling import lhd
import h5py
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
from numpy.linalg import inv, norm
from numpy.random import normal, random
# TODO: Do some research on scipy.interpolate.Rbf later
from scipy.interpolate import interp2d
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline as Pipeline_sk
from sklearn.preprocessing import PolynomialFeatures as PF

# PRISM imports
from _internal import RequestError, docstring_copy, move_logger, start_logger
from modellink import ModelLink

# All declaration
__all__ = ['Emulator']


# %% EMULATOR CLASS DEFINITION
# TODO: Write docstrings
class Emulator(object):
    """
    Defines the :class:`~Emulator` class of the PRISM package.

    """

    def __init__(self, prism_file, open_hdf5, close_hdf5):
        """
        Initialize an instance of the :class:`~Emulator` class.

        """

        # Save keyword arguments
        self._prism_file = prism_file
        self._open_hdf5 = open_hdf5
        self._close_hdf5 = close_hdf5

        # Load the emulator and data
        self._load_emulator()


# %% CLASS PROPERTIES
    @property
    def prism_file(self):
        """
        Absolute path to PRISM parameters file.

        """

        return(self._prism_file)

    @property
    def emul_load(self):
        """
        Bool indicating whether or not a previously constructed emulator system
        is currently loaded.

        """

        return(bool(self._emul_load))

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
        Obtained from regression process and replaces the Gaussian sigma .

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
        Array containg all model outputs in the currently loaded emulator
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

    @property
    def modellink(self):
        """
        The :obj:`~ModelLink` instance provided during Pipeline initialization.

        """

        return(self._modellink)

    @modellink.setter
    def modellink(self, modellink):
        """
        Sets the `modellink` property.

        Parameters
        ----------
        modellink : :obj:`~ModelLink` object
            Instance of the :class:`~ModelLink` class that links the emulated
            model to this :obj:`~Pipeline` object.
            The provided :obj:`~ModelLink` object must be the same as the one
            used to construct the loaded emulator system.

        """

        # Check if a subclass of the ModelLink class has been provided
        if not isinstance(modellink, ModelLink):
            raise TypeError("Input argument 'model_link' must be an instance "
                            "of the ModelLink class!")

        # If an existing emulator system is loaded, check if classes are equal
        if self._emul_load:
            # Make abbreviations
            modellink_loaded = self._modellink_name
            modellink_given = modellink.__class__.__name__

            if(modellink_given != modellink_loaded):
                raise InputError("Provided ModelLink subclass %s is not equal "
                                 "to ModelLink subclass %s used for emulator "
                                 "construction!"
                                 % (modellink_given, modellink_loaded))
        # If not, set the name
        else:
            self._modellink_name = modellink.__class__.__name__

        # Set ModelLink class
        self._modellink = modellink

    @property
    def modellink_name(self):
        """
        Name of the :obj:`~ModelLink` instance provided during Pipeline
        initialization.

        """

        return(self._modellink_name)

    @property
    def method(self):
        """
        String indicating which emulator method to use.
        Possible are 'gaussian', 'regression' and 'full'.

        """

        return(self._method)

    @property
    def poly_order(self):
        """
        Polynomial order that is considered for the regression process.
        If method='gaussian', this number is not required.

        """

        return(self._poly_order)

    @property
    def sigma(self):
        """
        List with Gaussian sigmas.
        This value is not required to be provided if regression is used, since
        it is obtained from the regression process instead.

        """

        return(self._sigma)

    @property
    def l_corr(self):
        """
        List with Gaussian correlation lengths for active parameters.

        """

        return(self._l_corr)


# %% GENERAL CLASS METHODS
    # Check for hdf5-file type
    def _get_emul_i(self, emul_i):
        """
        Checks what type of HDF5-file is provided and loads in the
        corresponding emulator data. This method is called by visible methods
        that allows for flexibility in the chosen emulator iteration `emul_i`.

        Parameters
        ----------
        emul_i : int or None
            Number indicating the current emulator iteration.

        Generates
        ---------
        All relevant emulator data corresponding to emulator iteration `emul_i`
        is loaded into memory.

        Returns
        -------
        emul_i : int
            Correct number indicating the current emulator iteration.

        """

        # Log that emul_i is being switched
        logger = logging.getLogger('INIT')
        logger.info("Selecting emulator iteration for user-method.")

        # Check what kind of hdf5-file was provided
        if(emul_i == 0 or self._emul_load == 0):
            raise RequestError("Emulator HDF5-file is not built yet!")
        elif emul_i is None:
            emul_i = self._emul_i
        elif not(1 <= emul_i <= self._emul_i):
            logger.error("Requested emulator iteration %s does not exist!"
                         % (emul_i))
            raise RequestError("Requested emulator iteration %s does not "
                               "exist!" % (emul_i))
        logger.info("Selected emulator iteration %s." % (emul_i))

        # Return correct emul_i
        return(emul_i)

    # Creates a new emulator file and writes all information to it
    def _create_new_emulator(self):
        """
        Creates a new emulator file.

        """

        # If no constructed emulator was provided, it will be constructed now
        self._emul_load = 1
        self._emul_i = 0

        # Read in parameters from provided parameter file
        self._read_parameters()

        # Create hdf5-file
        file = self._open_hdf5('w')

        # Save all relevant emulator parameters to hdf5
        file.attrs['sigma'] = self._sigma
        file.attrs['l_corr'] = self._l_corr
        file.attrs['method'] = self._method.encode('ascii', 'ignore')
        file.attrs['poly_order'] = self._poly_order
        file.attrs['modellink_name'] =\
            self._modellink_name.encode('ascii', 'ignore')

        # Close hdf5-file
        self._close_hdf5(file)

        # Load relevant data
        self._load_data(0)

        # Prepare first emulator iteration to be constructed
        self._prepare_new_iteration(1)

    # Prepares the emulator for a new iteration
    # TODO: Should _create_new_emulator be combined with this method?
    def _prepare_new_iteration(self, emul_i):
        """
        Prepares the emulator system for a new iteration.

        """

        logger = logging.getLogger('EMUL-PREP')

        # Check if new iteration can be constructed
        if(emul_i == 1):
            result = 0
        elif not(1 <= emul_i-1 <= self._emul_i):
            raise RequestError("Preparation of emulator iteration %s is only "
                               "available when all previous iterations exist!"
                               % (emul_i))
        elif(emul_i-1 == self._emul_i):
            result = 0
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
                if not(self._data_val[emul_i][i] in self._modellink._data_val):
                    break
                if not(self._data_err[emul_i][i] in self._modellink._data_err):
                    break
                if not(self._data_idx[emul_i][i] in self._modellink._data_idx):
                    break
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

            result = 1

        # Open hdf5-file
        file = self._open_hdf5('r+')

        # Make group for this emulator iteration
        file.create_group('%s' % (emul_i))

        # Save the number of data sets
        file['%s' % (emul_i)].attrs['n_data'] = self._modellink._n_data
        self._n_data.append(self._modellink._n_data)

        # Create empty lists for the three data arrays
        data_val = []
        data_err = []
        data_idx = []

        # Create groups for all data sets
        # TODO: Add check if all previous data sets are still present?
        for i in range(self._modellink._n_data):
            file.create_group('%s/data_set_%s' % (emul_i, i))
            file['%s/data_set_%s' % (emul_i, i)].attrs['data_val'] =\
                self._modellink._data_val[i]
            data_val.append(self._modellink._data_val[i])
            file['%s/data_set_%s' % (emul_i, i)].attrs['data_err'] =\
                self._modellink._data_err[i]
            data_err.append(self._modellink._data_err[i])
            file['%s/data_set_%s' % (emul_i, i)].attrs['data_idx'] =\
                self._modellink._data_idx[i]
            data_idx.append(self._modellink._data_idx[i])

        # Close hdf5-file
        self._close_hdf5(file)

        # Save model data arrays to memory
        self._data_val.append(data_val)
        self._data_err.append(data_err)
        self._data_idx.append(data_idx)

        # Return the result
        return(result)

    # This function constructs the emulator iteration emul_i
    def _construct_iteration(self, emul_i):
        """
        Constructs the emulator iteration corresponding to the provided
        `emul_i`.

        """

        # Check if regression is required
        if(self._method.lower() in ('regression', 'full')):
            self._do_regression(emul_i)

        # Calculate the prior expectation and variance values of sam_set
        self._get_prior_exp_sam_set(emul_i)
        self._get_cov_matrix(emul_i)

    # This is function 'E_D(f(x'))'
    # This function gives the adjusted emulator expectation value back
    def _get_adj_exp(self, emul_i, par_setp, cov_vec):
        """
        Calculates the adjusted emulator expectation value at a given emulator
        iteration `emul_i` for specified parameter set `par_setp` and
        corresponding covariance vector `cov_vec`.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_setp : 1D array_like
            Model parameter value set to calculate the adjusted emulator
            expectation for.
        cov_vec : 2D array_like
            Covariance vector corresponding to `par_setp`.

        Returns
        -------
        adj_exp_val : 1D :obj:`~numpy.ndarray` object
            Adjusted emulator expectation value for every data point.

        """

        # Log that adjusted expectation value is calculated
        logger = logging.getLogger('EMUL_EXP')
        logger.info("Calculating adjusted emulator expectation value at %s."
                    % (par_setp))

        # Obtain prior expectation value of par_setp
        prior_exp_par_setp = self._get_prior_exp(emul_i, par_setp)

        # Create empty adj_exp_val
        adj_exp_val = np.zeros(self._n_data[emul_i])

        # Calculate the adjusted emulator expectation value at given par_setp
        for i in range(self._n_data[emul_i]):
            adj_exp_val[i] = prior_exp_par_setp[i] +\
                np.dot(np.transpose(cov_vec[i]),
                       np.dot(self._cov_mat_inv[emul_i][i],
                       (self._mod_set[emul_i][i] -
                        self._prior_exp_sam_set[emul_i][i])))

        # Log the result
        logger.info("Adjusted emulator expectation value is %s."
                    % (adj_exp_val))

        # Return it
        return(adj_exp_val)

    # This is function 'Var_D(f(x'))'
    # This function gives the adjusted emulator variance value back
    def _get_adj_var(self, emul_i, par_setp, cov_vec):
        """
        Calculates the adjusted emulator variance value at a given emulator
        iteration `emul_i` for specified parameter set `par_setp` and
        corresponding covariance vector `cov_vec`.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_setp : 1D array_like
            Model parameter value set to calculate the adjusted emulator
            variance for.
        cov_vec : 2D array_like
            Covariance vector corresponding to `par_setp`.

        Returns
        -------
        adj_var_val : 1D :obj:`~numpy.ndarray` object
            Adjusted emulator variance value for every data point.

        """

        # Log that adjusted emulator variance value is calculated
        logger = logging.getLogger('EMUL_VAR')
        logger.info("Calculating adjusted emulator variance value at %s."
                    % (par_setp))

        # Obtain prior variance value of par_setp
        prior_var_par_setp = self._get_prior_var(emul_i, par_setp)

        # Create empty adj_var_val
        adj_var_val = np.zeros(self._n_data[emul_i])

        # Calculate the adjusted emulator variance value at given par_setp
        for i in range(self._n_data[emul_i]):
            adj_var_val[i] = prior_var_par_setp[i] -\
                np.dot(np.transpose(cov_vec[i]),
                       np.dot(self._cov_mat_inv[emul_i][i], cov_vec[i]))

        # Log the result
        logger.info("Adjusted emulator variance value is %s."
                    % (adj_var_val))

        # Return it
        return(adj_var_val)

    # This function evaluates the emulator at a given emul_i and par_set and
    # returns the adjusted expectation and variance values
    def _evaluate(self, emul_i, par_set):
        """
        Evaluates the emulator system at iteration `emul_i` for given
        `par_set`.

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

    # This function performs a forward stepwise regression on sam_set
    def _do_regression(self, emul_i):
        """
        Performs a forward stepwise linear regression on all model evaluation
        samples in emulator iteration `emul_i`. Calculates what the expectation
        values of all polynomial coefficients are. The polynomial order that is
        used in the regression depends on the `poly_order` parameter provided
        during class initialization.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.

        Generates
        ---------
        rsdl_var : 1D list
            List containing the residual variances of the regression function.
        poly_coef : 2D list
            List containing the expectation values of the polynomial
            coefficients for all data points.
        poly_coef_cov : 2D list
            List containing the covariance values of the polynomial
            coefficients for all data points.
        poly_powers : 2D list
            List containing the powers every sample needs to be raised to, in
            order to obtain the polynomial terms used in the regression
            function.
        poly_idx : 2D list
            List containing the indices of the polynomial terms that are used
            in the regression function.

        """

        # Create logger
        logger = logging.getLogger('REGRESSION')
        logger.info("Performing regression.")

        # Create SequentialFeatureSelector object
        sfs_obj = SFS(LR(), k_features='best', forward=True,
                      floating=True, scoring='neg_mean_squared_error')

        # Create Pipeline object
        pipe = Pipeline_sk([('poly', PF(self._poly_order, include_bias=False)),
                            ('SFS', sfs_obj),
                            ('linear', LR())])

        # Create empty array containing the polynomial coefficients
        rsdl_var = []
        poly_coef = []
        poly_coef_cov = []
        poly_powers = []
        poly_idx = []

        # Loop over all data points and perform a regression on all of them
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
            logger.info("Regression score for data point %s: %s."
                        % (i, pipe.named_steps['linear'].score(
                           sam_set_poly, self._mod_set[emul_i][i])))

            # Add the intercept term to sam_set_poly
            sam_set_poly = np.concatenate([np.ones([self._n_sam[emul_i], 1]),
                                           sam_set_poly], axis=-1)

            # Calculate the poly_coef covariances
            poly_coef_cov.append(rsdl_var[i]*inv(
                    np.dot(np.transpose(sam_set_poly),
                           sam_set_poly)).flatten())

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
        self._save_data(emul_i, 'regression',
                        [rsdl_var, poly_coef, poly_coef_cov, poly_powers,
                         poly_idx])

        # Log that this is finished
        logger.info("Finished performing regression.")

    # This function gives the prior expectation value
    # This is function 'E(f(x'))' or 'u(x')'
    def _get_prior_exp(self, emul_i, par_set):
        """
        Calculates the prior expectation value at a given emulator iteration
        `emul_i` for specified parameter set `par_set`. This expectation
        depends on the emulator method used.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_set : 1D array_like or None
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

        # If prior_exp of par_set is requested (cov, adj_exp)
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
    def _get_prior_exp_sam_set(self, emul_i):
        """
        Calculates the prior expectation values at a given emulator iteration
        `emul_i` for all model evaluation samples and saves it for later use.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.

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
        self._save_data(emul_i, 'prior_exp_sam_set', prior_exp_sam_set)

        # Log again
        logger.info("Finished calculating prior expectation values.")

    # This function gives the prior variance value
    # This is function 'Var(f(x'))'
    def _get_prior_var(self, emul_i, par_set):
        """
        Calculates the prior variance value at a given emulator iteration
        `emul_i` for specified parameter set `par_set`. This variance depends
        on the emulator method used.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_set : 1D array_like
            Model parameter value set to calculate the prior variance for.

        Returns
        -------
        prior_var : 1D :obj:`~numpy.ndarray` object
            Prior variance value for every data point.

        """

        # Return it
        return(self._get_cov(emul_i, par_set, par_set))

    # This is function 'Cov(f(x), f(x'))' or 'k(x,x')
    def _get_cov(self, emul_i, par_set1, par_set2):
        """
        Calculates the covariance between given model parameter value sets
        `par_set1` and `par_set2`. This covariance depends on the emulator
        method used.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_set1, par_set2 : 1D array_like or None
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
        weight = 0.2

        # Determine which residual variance should be used
        if self._method.lower() in ('regression', 'full'):
            rsdl_var = self._rsdl_var[emul_i]
#            rsdl_var = pow(self._sigma, 2)
        elif self.method.lower() in ('gaussian'):
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
                    cov[i] += (1-weight)*rsdl_var[i] *\
                        np.exp(-1*pow(norm(
                            diff_sam_set[:, :,
                                         self._active_par_data[emul_i][i]],
                            axis=-1), 2) /
                                pow(norm(
                                    self._l_corr[
                                        self._active_par_data[emul_i][i]]), 2))

                    # Inactive parameter variety
                    cov[i] += weight*rsdl_var[i]*np.eye(self._n_sam[emul_i])

            if self._method.lower() in ('regression', 'full'):
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
                    cov[i] += (1-weight)*rsdl_var[i] *\
                        np.exp(-1*pow(norm(
                            diff_sam_set[:, self._active_par_data[emul_i][i]],
                            axis=-1), 2) /
                                pow(norm(
                                    self._l_corr[
                                        self._active_par_data[emul_i][i]]), 2))

                    # Inactive parameter variety
                    cov[i] += weight*rsdl_var[i] *\
                        (par_set1 == self._sam_set[emul_i]).all(axis=-1)

            if self._method.lower() in ('regression', 'full'):
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
                    cov[i] += (1-weight)*rsdl_var[i] *\
                        np.exp(-1*pow(norm(
                            diff_sam_set[self._active_par_data[emul_i][i]],
                            axis=-1), 2) /
                                pow(norm(
                                    self._l_corr[
                                        self._active_par_data[emul_i][i]]), 2))

                    # Inactive parameter variety
                    cov[i] += weight*rsdl_var[i] *\
                        (par_set1 == par_set2).all()
            if self._method.lower() in ('regression', 'full'):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, par_set1, par_set2)

        # Return it
        return(cov)

    # This function calculates the regression covariance.
    # This is function 'Cov(r(x), r(x'))'
    def _get_regr_cov(self, emul_i, par_set1, par_set2):
        """
        Calculates the covariance of the regression function at emulator
        iteration `emul_i` for given parameter sets `par_set1` and `par_set2`.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_set1, par_set2 : 1D array_like or None
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
    # OPTIMIZE: Calculate cov_vec for all samples at once to save time?
    def _get_cov_vector(self, emul_i, par_setp):
        """
        Calculates the column vector of covariances between given (`par_setp`)
        and known ('sam_set') model parameter value sets for a given emulator
        iteration `emul_i`.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        par_setp : 1D array_like
            Model parameter value set to calculate the covariances vector
            for.

        Returns
        -------
        cov_vec : 2D :obj:`~numpy.ndarray` object
            Column vector containing the covariances between given and
            known model parameter value sets for every data point.

        """

        # Log that covariance vector is being calculated
        logger = logging.getLogger('COV_VEC')
        logger.info("Calculating covariance vector at %s." % (par_setp))

        # Calculate covariance vector
        cov_vec = self._get_cov(emul_i, par_setp, None)

        # Log that it has been finished
        logger.info("Finished calculating covariance vector.")

        # Return it
        return(cov_vec)

    # This is function 'Var(D)' or 'A'
    # Reminder that this function should only be called once per sample set
    def _get_cov_matrix(self, emul_i):  # OPTIMIZE: Look at cov_mat once more
        """
        Calculates the (inverse) matrix of covariances between known model
        evaluation samples for a given emulator iteration `emul_i`.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.

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
        self._save_data(emul_i, 'cov_mat', [cov_mat, cov_mat_inv])

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
    def _load_emulator(self):
        """
        Loads the emulator.

        """

        # Make logger
        logger = logging.getLogger('EMUL-LOAD')

        # Check if an existing hdf5-file is provided
        try:
            logger.info("Checking if a constructed emulator HDF5-file is "
                        "provided.")
            file = self._open_hdf5('r')
        except (OSError, IOError):
            # No existing emulator was provided
            self._emul_load = 0
            logger.info("Non-existing HDF5-file provided.")
            self._emul_i = 0
#            self._read_parameters()     # TODO: Is this line necessary?
        else:
            self._emul_load = 1
            logger.info("Constructed emulator HDF5-file provided.")
            self._emul_i = len(file.items())
            self._close_hdf5(file)
            self._retrieve_parameters()

        # Load emulator data
        self._load_data(self._emul_i)

    # Function that loads in the emulator data
    # TODO: Write code that allows part of the data to be loaded in (crashing)
    # and forces the pipeline to continue where data starts missing
    def _load_data(self, emul_i):
        """
        Loads in all the important emulator data corresponding to emulator
        iteration `emul_i` into memory, if this is not loaded already.

        Parameters
        ----------
        emul_i : int or None
            Number indicating the current emulator iteration.

        Optional
        --------
        init : bool. Default: False
            Bool indicating whether or not this method is called during class
            initialization.

        Generates
        ---------
        All relevant emulator data corresponding to emulator iteration `emul_i`
        is loaded into memory, if not loaded already.

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
            file = self._open_hdf5('r')

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
                for j in range(self._n_data[i]):
                    mod_set.append(file['%s/data_set_%s/mod_set' % (i, j)][()])
                    cov_mat_inv.append(file['%s/data_set_%s/cov_mat_inv'
                                            % (i, j)][()])
                    prior_exp_sam_set.append(
                        file['%s/data_set_%s/prior_exp_sam_set'
                             % (i, j)][()])
                    active_par_data.append(
                        file['%s/data_set_%s/active_par_data' % (i, j)][()])
                    data_val.append(
                        file['%s/data_set_%s' % (i, j)].attrs['data_val'])
                    data_err.append(
                        file['%s/data_set_%s' % (i, j)].attrs['data_err'])
                    data_idx.append(
                        file['%s/data_set_%s' % (i, j)].attrs['data_idx'])
                self._mod_set.append(mod_set)
                self._cov_mat_inv.append(cov_mat_inv)
                self._prior_exp_sam_set.append(prior_exp_sam_set)
                self._active_par_data.append(active_par_data)
                self._data_val.append(data_val)
                self._data_err.append(data_err)
                self._data_idx.append(data_idx)

                if self._method.lower() in ('regression', 'full'):
                    rsdl_var = []
                    poly_coef = []
                    poly_coef_cov = []
                    poly_powers = []
                    poly_idx = []
                    for j in range(file['%s' % (i)].attrs['n_data']):
                        rsdl_var.append(file['%s/data_set_%s'
                                             % (i, j)].attrs['rsdl_var'])
                        poly_coef.append(file['%s/data_set_%s/poly_coef'
                                              % (i, j)][()])
                        poly_coef_cov.append(
                            file['%s/data_set_%s/poly_coef_cov' % (i, j)][()])
                        poly_powers.append(file['%s/data_set_%s/poly_powers'
                                                % (i, j)][()])
                        poly_idx.append(file['%s/data_set_%s/poly_idx'
                                             % (i, j)][()])
                    self._rsdl_var.append(rsdl_var)
                    self._poly_coef.append(poly_coef)
                    self._poly_coef_cov.append(poly_coef_cov)
                    self._poly_powers.append(poly_powers)
                    self._poly_idx.append(poly_idx)

            # Close hdf5-file
            self._close_hdf5(file)

            # Log that loading is finished
            logger.info("Finished loading relevant emulator data.")
        else:
            raise RequestError("Invalid operation requested!")

    # This function saves emulator data to hdf5
    def _save_data(self, emul_i, keyword, data):
        """
        Saves the provided `data` for the specified data-type `keyword` at the
        given emulator iteration `emul_i` to the HDF5-file and as an data
        attribute to the current :obj:`~Emulator` instance if required.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        keyword : str
            String specifying the type of data that needs to be saved. Possible
            are 'sam_set', 'mod_set', 'cov_mat', 'prior_exp_sam_set',
            'impl_sam', '1D_proj_hcube', '2D_proj_hcube', 'nD_proj_hcube',
            'regression' and 'active_par'.
        data : array_like
            The actual data that needs to be saved at data keyword `keyword`.

        Generates
        ---------
        The specified data is saved to the HDF5-file.

        """

        # Do some logging
        logger = logging.getLogger('SAVE_DATA')
        logger.info("Saving %s data at iteration %s to HDF5."
                    % (keyword, emul_i))

        # Open hdf5-file
        file = self._open_hdf5('r+')

        # Check what data keyword has been provided
        # SAM_SET
        if keyword in ('sam_set'):
            file.create_dataset('%s/sam_set' % (emul_i), data=data)
            file['%s' % (emul_i)].attrs['n_sam'] = np.shape(data)[0]
            self._sam_set.append(data)
            self._n_sam.append(np.shape(data)[0])

        # MOD_SET
        elif keyword in ('mod_set'):
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/mod_set'
                                    % (emul_i, i), data=data[i])
            self._mod_set.append(data)

        # REGRESSION
        elif keyword in ('regression'):
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
            self._rsdl_var.append(data[0])
            self._poly_coef.append(data[1])
            self._poly_coef_cov.append(data[2])
            self._poly_powers.append(data[3])
            self._poly_idx.append(data[4])

        # IMPL_SAM
        elif keyword in ('impl_sam'):
            # Save if any plausible regions have been found at all
            if(np.shape(data)[0] == 0):
                file['%s' % (emul_i)].attrs['prc'] = 0
                self._prc.append(0)
            else:
                file['%s' % (emul_i)].attrs['prc'] = 1
                self._prc.append(1)

            # Save impl_sam
            file.create_dataset('%s/impl_sam' % (emul_i), data=data)
            self._impl_sam.append(data)

        # PRIOR_EXP_SAM_SET
        elif keyword in ('prior_exp_sam_set'):
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/prior_exp_sam_set'
                                    % (emul_i, i), data=data[i])
            self._prior_exp_sam_set.append(data)

        # COV_MAT
        elif keyword in ('cov_mat'):
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/cov_mat' % (emul_i, i),
                                    data=data[0][i])
                file.create_dataset('%s/data_set_%s/cov_mat_inv' % (emul_i, i),
                                    data=data[1][i])
            self._cov_mat_inv.append(data[1])

        # 2D_PROJ_HCUBE
        elif keyword in ('2D_proj_hcube'):
            file.create_dataset('%s/proj_hcube/%s/impl_min'
                                % (emul_i, self._par_names[data[0][0]]),
                                data=data[1])
            file.create_dataset('%s/proj_hcube/%s/impl_los'
                                % (emul_i, self._par_names[data[0][0]]),
                                data=data[2])

        # ND_PROJ_HCUBE
        elif keyword in ('nD_proj_hcube'):
            file.create_dataset('%s/proj_hcube/%s-%s/impl_min'
                                % (emul_i, self._par_names[data[0][0]],
                                   self._par_names[data[0][1]]),
                                data=data[1])
            file.create_dataset('%s/proj_hcube/%s-%s/impl_los'
                                % (emul_i, self._par_names[data[0][0]],
                                   self._par_names[data[0][1]]),
                                data=data[2])

        # ACTIVE PARAMETERS
        elif keyword in ('active_par'):
            file.create_dataset('%s/active_par' % (emul_i), data=data[0])
            for i in range(self._n_data[emul_i]):
                file.create_dataset('%s/data_set_%s/active_par_data'
                                    % (emul_i, i), data=data[1][i])
            self._active_par.append(data[0])
            self._active_par_data.append(data[1])

        else:
            raise ValueError("Invalid keyword argument provided!")

        # Close hdf5-file
        self._close_hdf5(file)

        # More logging
        logger.info("Finished saving data to HDF5.")

    # Read in the emulator attributes
    def _retrieve_parameters(self):
        """
        Reads in the emulator parameters from the provided HDF5-file and saves
        them in the current :obj:`~Emulator` instance.

        """

        # Log that attributes are being read
        logger = logging.getLogger('INIT')
        logger.info("Reading attributes from provided HDF5-file.")

        # Open hdf5-file
        file = self._open_hdf5('r')

        # Read in all the emulator attributes
        self._sigma = file.attrs['sigma']
        self._l_corr = file.attrs['l_corr']
        self._method = file.attrs['method'].decode('utf-8')
        self._poly_order = file.attrs['poly_order']
        self._modellink_name = file.attrs['modellink_name'].decode('utf-8')

        # Close hdf5-file
        self._close_hdf5(file)

        # Log that reading is finished
        logger.info("Finished reading attributes.")

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
                    'poly_order': '3'}

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
        if self._prism_file is not None:
            emul_par = np.genfromtxt(self._prism_file, dtype=(str),
                                     delimiter=': ', autostrip=True)

            # Make sure that emul_par is 2D
            emul_par = np.array(emul_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(emul_par)

        # GENERAL
        # Gaussian sigma
        self._sigma = float(par_dict['sigma'])

        # Gaussian correlation length
        self._l_corr = float(par_dict['l_corr']) *\
            (self._modellink._par_rng[:, 1]-self._modellink._par_rng[:, 0])

        # Method used to calculate emulator functions
        # Future will include 'gaussian', 'regression', 'auto' and 'full'
        self._method = str(par_dict['method']).replace("'", '')

        # Obtain the polynomial order for the regression selection process
        self._poly_order = int(par_dict['poly_order'])

        # Log that reading has been finished
        logger.info("Finished reading emulator parameters.")
