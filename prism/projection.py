# -*- coding: utf-8 -*-

"""
Projection
==========

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
from scipy.interpolate import interp1d, interp2d
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline as Pipeline_skl
from sklearn.preprocessing import PolynomialFeatures as PF

# PRISM imports
from _internal import RequestError, docstring_copy, move_logger, start_logger
from emulator import Emulator
from modellink import ModelLink

# All declaration
__all__ = ['Projection']


# %% PROJECTION CLASS DEFINITION
# TODO: Write docstrings
class Projection(object):
    """
    Defines the :class:`~Projection` class of the PRISM package.

    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._modellink = self._pipeline._modellink
        self._emulator = self._pipeline._emulator

    def __call__(self, emul_i=None, proj_par=None, figure=True, show=False,
                 force=False):
        """
        Analyzes the emulator iteration `emul_i` and constructs a series of
        projection figures detailing the behavior of the model parameters
        corresponding to the given `proj_par`.
        The output depends on the number of model parameters.

        Optional
        --------
        emul_i : int or None. Default: None
            If an existing constructor file was provided during class
            initialization, the iteration of the emulator corresponding to
            `emul_i` will be used automatically.
            If *None*, the last iteration of the currently loaded emulator will
            be used.
        proj_par : 1D array_like of {int, str} or None. Default: None
            For which model parameters to construct the projection figures.
            If 1D array_like, construct projection figures for all combinations
            of active model parameters.
            If 1D array_like of str, the strings refer to the names of the
            model parameters.
            If 1D array_like of int, the integers refer to the order in which
            the model parameters are shown in the :func:`~details` method.
            If *None*, projection figures are made for all model parameters.
        figure : bool. Default: True
            Whether or not to create the projection figures. If *False*, only
            the data required to create the figures is calculated and saved in
            the HDF5-file, but the figures themselves are not created.
        show : bool. Default: False
            If `figure` is *True*, whether or not to show the figure after it
            has been created.
        force : bool. Default: False
            Controls what to do if a projection cube has been calculated at the
            emulator iteration `emul_i` before.
            If *False*, it will use the previously acquired projection data to
            create the projection figures.
            If *True*, it will recalculate all the data required to create the
            projection figures.

        Generates
        ---------
        A series of projection figures detailing the behavior of the model.
        The lay-out and output of the projection figures depend on the number
        of model parameters `par_dim`:
            *par_dim = 1*: The output will feature two subplots for every data
            point used in the emulator construction. The top subplot shows the
            emulators expectation and variance values, while the bottom subplot
            shows the corresponding implausibility values at all tested
            emulator evaluation samples.

            *par_dim = 2*: The output will feature two figures for the two
            model parameters with two subplots each. Every figure gives details
            about the behavior of the corresponding model parameter, by showing
            the minimum implausibility value (top) and the line-of-sight depth
            (bottom) obtained at the specified parameter value, independent of
            the value of the other parameter.

            *par_dim > 2*: The output will feature a figure with two subplots
            for every combination of two model parameters that can be made
            (par_dim*(par_dim-1)/2). Every figure gives details about the
            behavior of the corresponding model parameters, as well as their
            dependency on each other. This is done by showing the minimum
            implausibility (top) and the line-of-sight depth (bottom) obtained
            at the specified parameter values, independent of the values of the
            remaining model parameters.

        """

        # Log the start of the creation of the projection
        logger = logging.getLogger('PROJECTION')
        logger.info("Starting the projection process.")

        # Check what kind of hdf5-file has been provided
        self._emul_i = self._emulator._get_emul_i(emul_i)

        # Get the impl_cut list and n_proj_samples/n_hidden_samples
        # TODO: Make sure that the same impl_cut is used for all figures
        # TODO: If desync is True, maybe not require force parameter?
        self._get_impl_par()

        # Make abbreviations for certain variables
        nps = self._n_proj_samples

        # Check the proj_par that were provided
        # If none was provided, make figures for all model parameters
        if proj_par is None:
            proj_par = self._emulator._active_par[self._emul_i]

        # Else, an array of str/int must be provided
        else:
            # Check if proj_par can be converted to a numpy array
            try:
                proj_par = np.array(proj_par, ndmin=1)
            except Exception as error:
                raise InputError("Input argument 'proj_par' is invalid (%s)"
                                 % (error))

            # Check if converted numpy array is 1D
            if(proj_par.ndim != 1):
                raise ShapeError("Input argument 'proj_par' must be "
                                 "one-dimensional!")

            # If array contains strings, convert them to ints
            if(type(proj_par[0]) == np.str_):
                tmp_list = []
                for s in proj_par:
                    tmp_list.append(self._modellink._par_names.index(s))
                proj_par = np.array(tmp_list)

            # Make sure that proj_par is sorted
            proj_par.sort()

            # Check which values in proj_par are also in active_par
            proj_par = np.array(
                [i for i in self._emulator._active_par[self._emul_i] if
                 i in proj_par])

            # Make sure that there are still enough values left
            if(self._modellink._par_dim == 2 and len(proj_par) >= 1):
                pass
            elif(self._modellink._par_dim > 2 and len(proj_par) >= 2):
                pass
            else:
                raise InputError("Not enough active model parameters have been"
                                 " provided to make a projection figure!")

        # Obtain list of cube names
        if(self._modellink._par_dim == 2):
            cube_idx = list(combinations(range(len(proj_par)), 1))
            cube_par = proj_par[np.array(cube_idx)].tolist()
        else:
            cube_idx = list(combinations(range(len(proj_par)), 2))
            cube_par = proj_par[np.array(cube_idx)].tolist()

        # Create empty list holding cube_par that needs to be created
        create_cube_par = []

        # Open hdf5-file
        logger.info("Checking if projection data already exists.")
        file = self._pipeline._open_hdf5('r+')

        # Check if data is already there and act accordingly
        for cube in cube_par:
            if(self._modellink._par_dim == 2):
                # Make abbreviation for the parameter name
                par_name = self._modellink._par_names[cube[0]]

                try:
                    file.create_group('%s/proj_hcube/%s'
                                      % (self._emul_i, par_name))
                except ValueError:
                    if force is True:
                        del file['%s/proj_hcube/%s'
                                 % (self._emul_i, par_name)]
                        logger.info("Projection data '%s' already exists. "
                                    "Deleting."
                                    % (par_name))
                        create_cube_par.append(cube)
                    else:
                        logger.info("Projection data '%s' already exists. "
                                    "Skipping data creation."
                                    % (par_name))
                else:
                    logger.info("Projection data '%s' not found. Will be "
                                "created."
                                % (par_name))
                    create_cube_par.append(cube)
            else:
                # Make abbreviation for the parameter names
                par1_name = self._modellink._par_names[cube[0]]
                par2_name = self._modellink._par_names[cube[1]]

                try:
                    file.create_group('%s/proj_hcube/%s-%s'
                                      % (self._emul_i, par1_name, par2_name))
                except ValueError:
                    if force is True:
                        del file['%s/proj_hcube/%s-%s'
                                 % (self._emul_i, par1_name, par2_name)]
                        logger.info("Projection data '%s-%s' already "
                                    "exists. Deleting."
                                    % (par1_name, par2_name))
                        create_cube_par.append(cube)
                    else:
                        logger.info("Projection data '%s-%s' already "
                                    "exists. Skipping data creation."
                                    % (par1_name, par2_name))
                else:
                    logger.info("Projection data '%s-%s' not found. Will "
                                "be created."
                                % (par1_name, par2_name))
                    create_cube_par.append(cube)

        # Close hdf5-file
        self._pipeline._close_hdf5(file)

        # Make sure create_cube_par is a numpy array
#       create_cube_par = np.array(create_cube_par)

        # 2+D
        # Loop over all requested projection cubes
        for cube in cube_par:
            # ANALYZE PROJECTION 2+D
            # Create projection hypercube containing all samples if required
            if(self._modellink._par_dim == 2 and cube in create_cube_par):
                # Log that projection data is being created
                logger.info("Calculating projection data '%s'."
                            % (self._modellink._par_names[cube[0]]))

                proj_hcube = self._get_proj_hcube(cube)

                # Analyze this proj_hcube
                impl_min, impl_los =\
                    self._analyze_proj_hcube(proj_hcube)

                # Log that projection data has been created
                logger.info("Finished calculating projection data '%s'."
                            % (self._modellink._par_names[cube[0]]))

                # Save projection data to hdf5
                self._save_data('2D_proj_hcube',
                                [cube, impl_min, impl_los])
            elif(self._modellink._par_dim > 2 and cube in create_cube_par):
                # Log that projection data is being created
                logger.info("Calculating projection data '%s-%s'."
                            % (self._modellink._par_names[cube[0]],
                               self._modellink._par_names[cube[1]]))

                proj_hcube = self._get_proj_hcube(cube)

                # Analyze this proj_hcube
                impl_min, impl_los =\
                    self._analyze_proj_hcube(proj_hcube)

                # Log that projection data has been created
                logger.info("Finished calculating projection data '%s-%s'."
                            % (self._modellink._par_names[cube[0]],
                               self._modellink._par_names[cube[1]]))

                # Save projection data to hdf5
                self._save_data('nD_proj_hcube',
                                [cube, impl_min, impl_los])

            # PLOTTING
            # Plotting 2D
            if(self._modellink._par_dim == 2 and figure is True):
                # Get the parameter this cube is about
                par = cube[0]

                # Make abbreviation for parameter name
                par_name = self._modellink._par_names[par]

                # Log that figures are being created
                logger.info("Drawing projection figure '%s'."
                            % (par_name))

                # Open hdf5-file
                file = self._pipeline._open_hdf5('r')

                # Log that projection data is being obtained
                logger.info("Obtaining projection data '%s'."
                            % (par_name))

                # Obtain data
                impl_los = file['%s/proj_hcube/%s/impl_los'
                                % (self._emul_i, par_name)][()]
                impl_min = file['%s/proj_hcube/%s/impl_min'
                                % (self._emul_i, par_name)][()]

                # Log that projection data was obtained successfully
                logger.info("Finished obtaining projection data '%s'."
                            % (par_name))

                # Close hdf5-file
                self._pipeline._close_hdf5(file)

                # Obtain figure name prefix
                fig_prefix =\
                    path.join(self._pipeline._working_dir, 'proj_%s_cube_'
                              % (self._emul_i))

                # Recreate the parameter value array used to create the hcube
                proj_sam_set = np.linspace(self._modellink._par_rng[par, 0],
                                           self._modellink._par_rng[par, 1],
                                           nps)

                # Create parameter value array used in interpolation functions
                x = np.linspace(self._modellink._par_rng[par, 0],
                                self._modellink._par_rng[par, 1],
                                1000)

                # Get the interpolated functions describing the minimum
                # implausibility and line-of-sight depth obtained in every
                # point
                f_min = interp1d(proj_sam_set, impl_min, kind='cubic')
                f_los = interp1d(proj_sam_set, impl_los, kind='cubic')

                # Do plotting
                f, axarr = plt.subplots(2)
                plt.rc('text', usetex=True)
                f.suptitle(r"2D Projection %s Cube (%s)"
                           % (self._emul_i, par_name), fontsize='xx-large')

                # Plot minimum implausibility
                axarr[0].plot(x, f_min(x))
                draw_y =\
                    self._impl_cut[self._emul_i][self._cut_idx[self._emul_i]]
                draw_textline(r"$I_{\mathrm{cut-off, 1}}$", y=draw_y,
                              linestyle='g', ax=axarr[0])
                if self._modellink._par_estimate[par] is not None:
                    draw_textline(r"", x=self._modellink._par_estimate[par],
                                  pos='end', linestyle='k--', ax=axarr[0])
                axarr[0].set_ylabel(r"Minimum Implausibility "
                                    r"$I_{\mathrm{min}}(%s)$"
                                    % (par_name), fontsize='x-large')
#                axarr[0].tick_params(axis='both', labelsize='large')

                # Plot line-of-sight depth
                axarr[1].plot(x, f_los(x))
                if self._modellink._par_estimate[par] is not None:
                    draw_textline(r"", x=self._modellink._par_estimate[par],
                                  pos='end', linestyle='k--', ax=axarr[1])
                axarr[1].set_xlabel("Input Parameter %s" % (par_name),
                                    fontsize='x-large')
                axarr[1].set_ylabel("Line-of-Sight Depth", fontsize='x-large')
#                axarr[1].tick_params(axis='both', labelsize='large')

                # Save the figure
                plt.savefig('%s(%s).png' % (fig_prefix, par_name))

                # If show is set to True, show the figure
                if show is True:
                    f.show()
                else:
                    plt.close(f)

                # Log that this cube has been drawn
                logger.info("Finished drawing projection figure '%s'."
                            % (par_name))

            # Plotting 3D
            elif(self._modellink._par_dim > 2 and figure is True):
                # Get the parameter on the x-axis and y-axis this cube is about
                par1 = cube[0]
                par2 = cube[1]

                # Make abbreviation for the parameter names
                par1_name = self._modellink._par_names[par1]
                par2_name = self._modellink._par_names[par2]

                # Log that figures are being created
                logger.info("Drawing projection figure '%s-%s'."
                            % (par1_name, par2_name))

                # Open hdf5-file
                file = self._pipeline._open_hdf5('r')

                # Log that projection data is being obtained
                logger.info("Obtaining projection data '%s-%s'."
                            % (par1_name, par2_name))

                # Obtain data
                impl_los = file['%s/proj_hcube/%s-%s/impl_los'
                                % (self._emul_i, par1_name, par2_name)][()]
                impl_min = file['%s/proj_hcube/%s-%s/impl_min'
                                % (self._emul_i, par1_name, par2_name)][()]

                # Log that projection data was obtained successfully
                logger.info("Finished obtaining projection data '%s-%s'."
                            % (par1_name, par2_name))

                # Close hdf5-file
                self._pipeline._close_hdf5(file)

                # Obtain figure name prefix
                fig_prefix =\
                    path.join(self._pipeline._working_dir, 'proj_%s_cube_'
                              % (self._emul_i))

                # Recreate the parameter value arrays used to create the hcube
                proj_sam_set1 = np.linspace(self._modellink._par_rng[par1, 0],
                                            self._modellink._par_rng[par1, 1],
                                            nps)
                proj_sam_set2 = np.linspace(self._modellink._par_rng[par2, 0],
                                            self._modellink._par_rng[par2, 1],
                                            nps)

                # Create parameter value array used in interpolation functions
                x = np.linspace(self._modellink._par_rng[par1, 0],
                                self._modellink._par_rng[par1, 1],
                                1000)
                y = np.linspace(self._modellink._par_rng[par2, 0],
                                self._modellink._par_rng[par2, 1],
                                1000)

                # Get the interpolated functions describing the minimum
                # implausibility and line-of-sight depth obtained in every
                # grid point
                f_min = interp2d(proj_sam_set1, proj_sam_set2, impl_min,
                                 kind='quintic')
                f_los = interp2d(proj_sam_set1, proj_sam_set2, impl_los,
                                 kind='quintic')

                # Create 3D grid to be used in the hexbin plotting routine
                X, Y = np.meshgrid(x, y)
                Z_min = f_min(x, y)
                Z_los = f_los(x, y)
                x = X.ravel()
                y = Y.ravel()
                z_min = Z_min.ravel()
                z_los = Z_los.ravel()

                # Set the size of the hexbin grid
                gridsize = 1000

                # Do plotting
                f, axarr = plt.subplots(2, figsize=(6.4, 4.8), dpi=100)
                f.suptitle(r"3D Projection %s Cube (%s-%s)"
                           % (self._emul_i, par1_name, par2_name),
                           fontsize='xx-large')

                # Plot minimum implausibility
                vmax =\
                    self._impl_cut[self._emul_i][self._cut_idx[self._emul_i]]
                fig1 = axarr[0].hexbin(
                    x, y, z_min, gridsize, cmap=cm.jet, vmin=0,
                    vmax=vmax)
                axarr[0].set_ylabel("%s" % (par2_name), fontsize='x-large')
#                axarr[0].tick_params(axis='both', labelsize='large')
                if self._modellink._par_estimate[par1] is not None:
                    draw_textline(r"", x=self._modellink._par_estimate[par1],
                                  linestyle='k--', ax=axarr[0])
                if self._modellink._par_estimate[par2] is not None:
                    draw_textline(r"", y=self._modellink._par_estimate[par2],
                                  linestyle='k--', ax=axarr[0])
                axarr[0].axis([self._modellink._par_rng[par1, 0],
                               self._modellink._par_rng[par1, 1],
                               self._modellink._par_rng[par2, 0],
                               self._modellink._par_rng[par2, 1]])
                plt.colorbar(fig1, ax=axarr[0]).set_label(
                    "Minimum Implausibility", fontsize='large')

                # Plot line-of-sight depth
                fig2 = axarr[1].hexbin(x, y, z_los, gridsize, cmap=cm.hot,
                                       vmin=0, vmax=min(1, np.max(z_los)))
                axarr[1].set_xlabel("%s" % (par1_name), fontsize='x-large')
                axarr[1].set_ylabel("%s" % (par2_name), fontsize='x-large')
#                axarr[1].tick_params(axis='both', labelsize='large')
                if self._modellink._par_estimate[par1] is not None:
                    draw_textline(r"", x=self._modellink._par_estimate[par1],
                                  linestyle='k--', ax=axarr[1])
                if self._modellink._par_estimate[par2] is not None:
                    draw_textline(r"", y=self._modellink._par_estimate[par2],
                                  linestyle='k--', ax=axarr[1])
                axarr[1].axis([self._modellink._par_rng[par1, 0],
                               self._modellink._par_rng[par1, 1],
                               self._modellink._par_rng[par2, 0],
                               self._modellink._par_rng[par2, 1]])
                plt.colorbar(fig2, ax=axarr[1]).set_label(
                    "Line-of-Sight Depth", fontsize='large')

                # Save the figure
                plt.savefig('%s(%s-%s).png'
                            % (fig_prefix, par1_name, par2_name))

                # If show is set to True, show the figure
                if show is True:
                    f.show()
                else:
                    plt.close(f)

                # Log that this figure has been drawn
                logger.info("Finished drawing projection figure '%s-%s'."
                            % (par1_name, par2_name))

        # Log the end of the projection
        logger.info("Finished projection.")


# %% CLASS PROPERTIES
    @property
    def n_proj_samples(self):
        """
        Number of emulator evaluations used to generate the grid for the
        projection figures.

        """

        return(self._n_proj_samples)

    @property
    def n_hidden_samples(self):
        """
        Number of emulator evaluations used to generate the samples in every
        grid point for the projection figures.

        """

        return(self._n_hidden_samples)

    @property
    def impl_cut(self):
        """
        List containing all univariate implausibility cut-offs. A zero
        indicates a wildcard.

        """

        return(self._impl_cut)

    @impl_cut.setter
    # This function completes the list of implausibility cut-offs
    def impl_cut(self, impl_cut):
        """
        Generates the full list of impl_cut-offs from the incomplete, shortened
        list provided during class initialization.

        Parameters
        ----------
        impl_cut : 1D list
            Incomplete, shortened impl_cut-offs list provided during class
            initialization.

        Generates
        ---------
        impl_cut : 1D list
            Full list containing the impl_cut-offs for all data points provided
            to the emulator.
        cut_idx : int
            Index of the first impl_cut-off in the impl_cut list that is not
            0.

        """

        # Log that impl_cut-off list is being acquired
        logger = logging.getLogger('INIT')
        logger.info("Generating full implausibility cut-off list.")

        # Complete the impl_cut list
        for i in range(len(impl_cut)):
            if(impl_cut[i] == 0 and i == 0):
                pass
            elif(impl_cut[i] == 0):
                impl_cut[i] = impl_cut[i-1]
            elif(impl_cut[i-1] != 0 and impl_cut[i] > impl_cut[i-1]):
                raise ValueError("Cut-off %s is higher than cut-off %s "
                                 "(%s > %s)" % (i, i-1, impl_cut[i],
                                                impl_cut[i-1]))

        # Get the index identifying where the first real impl_cut is
        for i, impl in enumerate(impl_cut):
            if(impl != 0):
                cut_idx = i
                break

        # Save both impl_cut and cut_idx
        self._save_data('impl_cut', [impl_cut, cut_idx])

        # Log end of process
        logger.info("Finished generating implausibility cut-off list.")

    @property
    def cut_idx(self):
        """
        The list index of the first non-wildcard cut-off in impl_cut.

        """

        return(self._cut_idx)


# %% HIDDEN CLASS METHODS
    # This function reads in the impl_cut list from the PRISM parameters file
    def _get_impl_par(self):
        """
        Reads in the impl_cut list and other parameters for implausibility
        evaluations from the PRISM parameters file.

        """

        # Obtain the impl_cut and cut_idx up to this iteration from pipeline
        self._impl_cut = list(self._pipeline._impl_cut[:self._emul_i])
        self._cut_idx = list(self._pipeline._cut_idx[:self._emul_i])

        # Obtaining default pipeline parameter dict
        par_dict = self._get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._pipeline._prism_file is not None:
            pipe_par = np.genfromtxt(self._pipeline._prism_file, dtype=(str),
                                     delimiter=': ', autostrip=True)

            # Make sure that pipe_par is 2D
            pipe_par = np.array(pipe_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(pipe_par)

        # Implausibility cut-off
        impl_cut_str = str(par_dict['impl_cut']).replace(',', '').split()
        self.impl_cut = list(float(impl_cut) for impl_cut in impl_cut_str)

        # Number of samples used for implausibility evaluations
        self._save_data('n_proj_samples',
                        [int(par_dict['n_proj_samples']),
                         int(par_dict['n_hidden_samples'])])

    # This function automatically loads default pipeline parameters
    def _get_default_parameters(self):
        """
        Generates a dict containing default values for all projection
        parameters.

        Returns
        -------
        par_dict : dict
            Dict containing all default projection parameter values.

        """

        # Log this
        logger = logging.getLogger('INIT')
        logger.info("Generating default projection parameter dict.")

        # Create parameter dict with default parameters
        par_dict = {'n_proj_samples': '15',
                    'n_hidden_samples': '75',
                    'impl_cut': '0, 4.0, 3.8, 3.5'}

        # Log end
        logger.info("Finished generating default projection parameter dict.")

        # Return it
        return(par_dict)

    # This function generates a projection cube to be used for emulator
    # evaluations
    def _get_proj_hcube(self, cube_par):
        """
        Generates a hypercube containing emulator evaluation samples to be used
        in the :func:`~create_projection` function. The output of this function
        depends on the number of model parameters.

        Parameters
        ----------
        cube_par : 1D array_like of int of length {0, 1, 2}
            Array containing the internal integer identifiers of the main model
            parameters that require a projection hypercube.

        Returns
        -------
        proj_hcube : 2D :obj:`~numpy.ndarray` object
            2D Latin Hypercube of emulator evaluation samples if `par_dim` is 1
            or a 2D projection hypercube of emulator evaluation samples if
            `par_dim` is not 1.

        """

        # Log that projection hypercube is being created
        logger = logging.getLogger('PROJ_HCUBE')

        # Make abbreviations for certain variables
        nhs = self._n_hidden_samples
        nps = self._n_proj_samples
        ones = np.ones(nhs)

        # If par_dim is 2, make 1D projection cube
        if(self._modellink._par_dim == 2):
            # Identify projected parameter
            par = cube_par[0]

            # Log event
            logger.info("Creating projection hypercube '%s'."
                        % (self._modellink._par_names[par]))

            # Create empty projection hypercube array
            proj_hcube = np.zeros([nps*nhs, self._modellink._par_dim])

            # Create list that contains all the other parameters
            par_hid = list(chain(range(0, par),
                                 range(par+1, self._modellink._par_dim)))

            # Generate list with values for projected parameter
            proj_sam_set = np.linspace(self._modellink._par_rng[par, 0],
                                       self._modellink._par_rng[par, 1],
                                       nps)

            # Generate latin hypercube of the remaining parameters
            hidden_sam_set = lhd(nhs, self._modellink._par_dim-1,
                                 self._modellink._par_rng[par_hid],
                                 'fixed', self._pipeline._criterion)

            # Fill every cell in the projection cube accordingly
            for j in range(nps):
                # The projected parameter
                proj_hcube[nhs*j:nhs*(j+1), par] = ones*proj_sam_set[j]

                # The remaining hidden parameters
                proj_hcube[nhs*j:nhs*(j+1), par_hid] =\
                    np.transpose(hidden_sam_set)

            # Log that projection hypercube has been created
            logger.info("Finished creating projection hypercube '%s'."
                        % (self._modellink._par_names[par]))

            # Return proj_hcube
            return(proj_hcube)

        # If par_dim is more than 2, make 2D projection cubes like Bower fig. 4
        else:
            # Identify first projected parameter
            par1 = cube_par[0]

            # Identify second projected parameter
            par2 = cube_par[1]

            # Log event
            logger.info("Creating projection hypercube '%s-%s'."
                        % (self._modellink._par_names[par1],
                           self._modellink._par_names[par2]))

            # Create empty projection hypercube array
            proj_hcube = np.zeros([pow(nps, 2)*nhs, self._modellink._par_dim])

            # Generate list that contains all the other parameters
            par_hid = list(chain(range(0, par1), range(par1+1, par2),
                                 range(par2+1, self._modellink._par_dim)))

            # Generate list with values for first projected parameter
            proj_sam_set1 = np.linspace(self._modellink._par_rng[par1, 0],
                                        self._modellink._par_rng[par1, 1],
                                        nps)

            # Generate list with values for second projected parameter
            proj_sam_set2 = np.linspace(self._modellink._par_rng[par2, 0],
                                        self._modellink._par_rng[par2, 1],
                                        nps)

            # Generate latin hypercube of the remaining parameters
            hidden_sam_set = lhd(nhs, self._modellink._par_dim-2,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._pipeline._criterion)

            # Fill every cell in the projection cube accordingly
            for j in range(nps):
                for k in range(nps):
                    # The first projected parameter
                    proj_hcube[nhs*(j*nps+k):nhs*(j*nps+k+1), par1] =\
                        ones*proj_sam_set1[j]

                    # The second projected parameter
                    proj_hcube[nhs*(j*nps+k):nhs*(j*nps+k+1), par2] =\
                        ones*proj_sam_set2[k]

                    # The remaining hidden parameters
                    proj_hcube[nhs*(j*nps+k):nhs*(j*nps+k+1), par_hid] =\
                        hidden_sam_set

            # Log that projection hypercube has been created
            logger.info("Finished creating projection hypercube '%s-%s'."
                        % (self._modellink._par_names[par1],
                           self._modellink._par_names[par2]))

            # Return proj_hcube
            return(proj_hcube)

    # This function analyzes a projection hypercube
    def _analyze_proj_hcube(self, proj_hcube):
        """
        Analyzes an emulator projection hypercube used in the
        :func:`~create_projection` function.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.
        proj_hcube : 2D array_like
            2D projection hypercube containing all the emulator evaluation
            samples that are required to make the projection figures.

        Returns
        -------
        impl_min : 2D :obj:`~numpy.ndarray` object
            2D array containing the lowest implausibility value that can be
            reached in every single grid point on the given hypercube.
        impl_los : 2D :obj:`~numpy.ndarray` object
            2D array containing the fraction of the total amount of evaluated
            samples in every single grid point on the given hypercube, that
            still satisfied the implausibility cut-off criterion.

        """

        # Log that a projection hypercube is being analyzed
        logger = logging.getLogger('ANALYSIS')
        logger.info("Analyzing projection hypercube.")

        # Make abbreviations for certain variables
        nhs = self._n_hidden_samples

        # Obtain number of samples
        n_samples = proj_hcube.shape[0]

        # CALCULATE AND ANALYZE IMPLAUSIBILITY
        # Create empty lists of indices
        impl_idx_list = []
        impl_cut_list = []

        # Create empty list of indices for the cube
        impl_min_cube = []
        impl_los_cube = []

        # Iterate over all samples the cube
        for i in range(n_samples):
            print(i)
            for j in range(1, self._emul_i+1):
                # Obtain implausibility
                adj_val = self._emulator._evaluate(j, proj_hcube[i])
                uni_impl_val =\
                    self._pipeline._get_uni_impl(j, *adj_val)

                # Perform implausibility cut-off check
                impl_check, impl_cut_val =\
                    self._pipeline._do_impl_check(self, j, uni_impl_val)

                # If check was successful, save corresponding index
                if impl_check is False:
                    break
            else:
                impl_idx_list.append(i)

            # Save the implausibility value at the first real cut-off
            impl_cut_list.append(impl_cut_val)

            # If i has checked nhs samples, save the lowest impl and impl
            # line-of-sight
            if((i+1) % (nhs) == 0):
                # Calculate lowest impl in this grid point
                impl_min_cube.append(min(impl_cut_list))

                # Calculate impl line-of-sight in this grid point
                cube_bin = ((i+1-nhs < np.array(impl_idx_list)) &
                            (np.array(impl_idx_list) < i+1)).sum()
                impl_los_cube.append(cube_bin/nhs)

                # Clear both lists
                impl_cut_list = []
                impl_idx_list = []

        # Log that analysis has been finished
        logger.info("Finished projection hypercube analysis.")

        # Return impl_min and impl_los
        return(impl_min_cube, impl_los_cube)

    # This function saves projection data to hdf5
    def _save_data(self, keyword, data):
        """
        Saves the provided `data` for the specified data-type `keyword` at the
        given emulator iteration `emul_i` to the HDF5-file and as an data
        attribute to the current :obj:`~Projection` instance if required.

        Parameters
        ----------
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
                    % (keyword, self._emul_i))

        # Open hdf5-file
        file = self._pipeline._open_hdf5('r+')

        # 2D_PROJ_HCUBE
        if keyword in ('2D_proj_hcube'):
            file.create_dataset('%s/proj_hcube/%s/impl_min'
                                % (self._emul_i,
                                   self._modellink._par_names[data[0][0]]),
                                data=data[1])
            file.create_dataset('%s/proj_hcube/%s/impl_los'
                                % (self._emul_i,
                                   self._modellink._par_names[data[0][0]]),
                                data=data[2])

        # ND_PROJ_HCUBE
        elif keyword in ('nD_proj_hcube'):
            file.create_dataset('%s/proj_hcube/%s-%s/impl_min'
                                % (self._emul_i,
                                   self._modellink._par_names[data[0][0]],
                                   self._modellink._par_names[data[0][1]]),
                                data=data[1])
            file.create_dataset('%s/proj_hcube/%s-%s/impl_los'
                                % (self._emul_i,
                                   self._modellink._par_names[data[0][0]],
                                   self._modellink._par_names[data[0][1]]),
                                data=data[2])

        # IMPL_CUT
        elif keyword in ('impl_cut'):
            try:
                file.create_group('%s/proj_hcube' % (self._emul_i))
            except ValueError:
                pass

            self._impl_cut.append(data[0])
            self._cut_idx.append(data[1])
            file['%s/proj_hcube' % (self._emul_i)].attrs['impl_cut'] = data[0]
            file['%s/proj_hcube' % (self._emul_i)].attrs['cut_idx'] = data[1]

        # N_PROJ_SAMPLES
        elif keyword in ('n_proj_samples'):
            try:
                file.create_group('%s/proj_hcube' % (self._emul_i))
            except ValueError:
                pass

            self._n_proj_samples = data[0]
            self._n_hidden_samples = data[1]
            file['%s/proj_hcube' % (self._emul_i)].attrs['n_proj_samples'] =\
                data[0]
            file['%s/proj_hcube' % (self._emul_i)].attrs['n_hidden_samples'] =\
                data[1]

        else:
            raise ValueError("Invalid keyword argument provided!")

        # Close hdf5-file
        self._pipeline._close_hdf5(file)
