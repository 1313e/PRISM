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
from sklearn.pipeline import Pipeline as Pipeline_skl
from sklearn.preprocessing import PolynomialFeatures as PF

# PRISM imports
from ._internal import RequestError, move_logger, start_logger
from .modellink import ModelLink

# All declaration
__all__ = ['Emulator']


# %% EMULATOR CLASS DEFINITION
# TODO: Write pydocs
class Emulator(object):
    """
    Defines the :class:`~Emulator` class of the PRISM package.

    """

    def __init__(self, hdf5_file):
        """
        Initialize an instance of the :class:`~Emulator` class.

        """

        # Save name of HDF5-file
        # TODO: Might want to make a test for checking the correctness of the
        # name
        self._hdf5_file = hdf5_file

        # Load the emulator and data
        self._load_emulator()


# %% CLASS PROPERTIES
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
    def prc(self):
        """
        Bool indicating whether or not plausible regions have been found in the
        currently loaded emulator iteration.

        """

        return([bool(self._prc[i]) for i in range(self._emul_i+1)])

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


# %% GENERAL CLASS METHODS
    # Open hdf5-file
    def _open_hdf5(self, mode, filename=None, **kwargs):
        """
        Opens the HDF5-file `filename` according to some set of default
        parameters and returns the opened HDF5-object.

        Parameters
        ----------
        mode : {'r', 'r+', 'w', 'w-'/'x', 'a'}
            String indicating how the HDF5-file needs to be opened.

        Optional
        --------
        filename : str. Default: None
            The name/path of the HDF5-file that needs to be opened in
            `working_dir`. Default is to open the HDF5-file that was provided
            during class initialization.
        **kwargs : dict. Default: {'driver': None, 'libver': 'earliest'}
            Other keyword arguments that need to be given to the
            :func:`~h5py.File` function.

        Returns
        -------
        file : :obj:`~h5py._hl.files.File` object
            Opened HDF5-file object.

        """

        # Log that an HDF5-file is being opened
        logger = logging.getLogger('HDF5-FILE')

        # Set default settings
        driver = None
        libver = 'earliest'     # Set to 'earliest' to ensure Linux can read it

        # Check filename
        if filename is None:
            filename = self._hdf5_file
        else:
            pass

        # Check if certain keywords have been provided manually
        for key, value in kwargs.items():
            if(key == 'driver'):
                driver = value
            elif(key == 'libver'):
                libver = value
            else:
                pass

        # Make sure that these keywords have the correct values
        kwargs['driver'] = driver
        kwargs['libver'] = libver

        # Open hdf5-file
        logger.info("Opening HDF5-file '%s' (mode: '%s')." % (filename, mode))
        file = h5py.File(filename, mode, **kwargs)

        # Return the opened hdf5-file
        return(file)

    # Close hdf5-file
    def _close_hdf5(self, file):
        """
        Closes the opened HDF5-file object `file`. This method exists only
        for logging purposes.

        Parameters
        ----------
        file : :obj:`~h5py._hl.files.File` object
            Opened HDF5-file object requiring closing.

        """

        # Log that an HDF5-file will be closed
        logger = logging.getLogger('HDF5-FILE')

        # Close hdf5-file
        file.close()

        # Log about closing the file
        logger.info("Closed HDF5-file.")

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
        else:
            self._emul_load = 1
            logger.info("Constructed emulator HDF5-file provided.")
            self._emul_i = len(file.items())
            self._close_hdf5(file)

        # Load emulator data
        self._load_data(self._emul_i)

    # Function that loads in the emulator data
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
        self._prc = [[]]
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
                self._prc.append(file['%s' % (i)].attrs['prc'])
                self._n_sam.append(file['%s' % (i)].attrs['n_sam'])
                self._sam_set.append(file['%s/sam_set' % (i)][()])
                self._active_par.append(file['%s/active_par' % (i)][()])
                mod_set = np.zeros([self._n_sam[i], self._n_data])
                cov_mat_inv = np.zeros([self._n_data, self._n_sam[i],
                                        self._n_sam[i]])
                prior_exp_sam_set = np.zeros([self._n_data, self._n_sam[i]])
                active_par_data = []
                for j in range(self._n_data):
                    mod_set[:, j] = file['%s/data_set_%s/mod_set'
                                         % (i, j)][()]
                    cov_mat_inv[j] = file['%s/data_set_%s/cov_mat_inv'
                                          % (i, j)][()]
                    prior_exp_sam_set[j] =\
                        file['%s/data_set_%s/prior_exp_sam_set'
                             % (i, j)][()]
                    active_par_data.append(
                        file['%s/data_set_%s/active_par_data' % (i, j)][()])
                self._mod_set.append(mod_set)
                self._cov_mat_inv.append(cov_mat_inv)
                self._prior_exp_sam_set.append(prior_exp_sam_set)
                self._active_par_data.append(active_par_data)

                if self._method.lower() in ('regression', 'full'):
                    rsdl_var = []
                    poly_coef = []
                    poly_coef_cov = []
                    poly_powers = []
                    poly_idx = []
                    for j in range(self._n_data):
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
