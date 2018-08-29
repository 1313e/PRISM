# -*- coding: utf-8 -*-

"""
Projection
==========
Provides the definition of PRISM's :class:`~Projection` class, that allows for
projection figures detailing a model's behavior to be created.


Available classes
-----------------
:class:`~Projection`
    Defines the :class:`~Projection` class of the PRISM package.

"""


# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
from itertools import chain, combinations
from logging import getLogger
import os
from os import path
from time import time

# Package imports
from e13tools import InputError
from e13tools.pyplot import draw_textline, suplabel
from e13tools.sampling import lhd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
# TODO: Do some research on scipy.interpolate.Rbf later
from scipy.interpolate import interp1d, interp2d
from sortedcontainers import SortedSet
from tqdm import tqdm

# PRISM imports
from ._docstrings import user_emul_i_doc
from ._internal import (PRISM_File, RequestError, check_bool, check_pos_int,
                        convert_str_seq, docstring_substitute, getCLogger,
                        rprint)

# All declaration
__all__ = ['Projection']


# %% PROJECTION CLASS DEFINITION
# TODO: Make something like Figure 4 in Bower et al
class Projection(object):
    """
    Defines the :class:`~Projection` class of the PRISM package.

    """

    def __init__(self, pipeline_obj):
        """
        Initialize an instance of the :class:`~Projection` class.

        Parameters
        ----------
        pipeline_obj : :obj:`~Pipeline` object
            Instance of the :class:`~Pipeline` class that initialized this
            class.

        """

        # Save Pipeline, Emulator and ModelLink objects as attributes
        self._pipeline = pipeline_obj
        self._emulator = self._pipeline._emulator
        self._modellink = self._pipeline._modellink

        # Copy MPI properties to this instance
        self._comm = self._pipeline._comm
        self._size = self._pipeline._size
        self._rank = self._pipeline._rank
        self._is_controller = self._pipeline._is_controller
        self._is_worker = self._pipeline._is_worker

        # Add hdf5_file attribute to PRISM_File
        PRISM_File._hdf5_file = self._pipeline._hdf5_file

    # Function that creates all projection figures
    # TODO: Allow for impl, los and line kwargs to be provided
    @docstring_substitute(emul_i=user_emul_i_doc)
    def __call__(self, emul_i=None, proj_par=None, figure=True, show=False,
                 force=False):
        """
        Analyzes the emulator iteration `emul_i` and constructs a series of
        projection figures detailing the behavior of the model parameters
        corresponding to the given `proj_par`.
        The output depends on the number of model parameters.

        Optional
        --------
        %(emul_i)s
        proj_par : 1D array_like of {int, str} or None. Default: None
            For which model parameters to construct the projection figures.
            If 1D array_like, construct projection figures for all combinations
            of active model parameters.
            If 1D array_like of str, the strings refer to the names of the
            model parameters.
            If 1D array_like of int, the integers refer to the order in which
            the model parameters are shown in the :meth:`~details` method.
            If *None*, projection figures are made for all model parameters.
        figure : bool. Default: True
            Whether or not to create the projection figures. If *False*, only
            the data required to create the figures is calculated and saved in
            the HDF5-file, but the figures themselves are not created.
        show : bool. Default: False
            If `figure` is *True*, whether or not to show a figure after it has
            been created.
        force : bool. Default: False
            Controls what to do if a projection hypercube has been calculated
            at the emulator iteration `emul_i` before.
            If *False*, it will use the previously acquired projection data to
            create the projection figure.
            If *True*, it will recalculate all the data required to create the
            projection figure.

        Generates
        ---------
        A series of projection figures detailing the behavior of the model.
        The lay-out and output of the projection figures depend on the number
        of model parameters :attr:`~ModelLink.n_par`:
            `n_par` == 2: The output will feature two figures for the two
            model parameters with two subplots each. Every figure gives details
            about the behavior of the corresponding model parameter, by showing
            the minimum implausibility value (top) and the line-of-sight depth
            (bottom) obtained at the specified parameter value, independent of
            the value of the other parameter.

            `n_par` > 2: The output will feature a figure with two subplots
            for every combination of two active model parameters that can be
            made (``n_par*(n_par-1)/2``). Every figure gives details about the
            behavior of the corresponding model parameters, as well as their
            dependency on each other. This is done by showing the minimum
            implausibility (top) and the line-of-sight depth (bottom) obtained
            at the specified parameter values, independent of the values of the
            remaining model parameters.

        """

        # Log the start of the creation of the projection
        logger = getLogger('PROJECTION')
        logger.info("Starting the projection process.")

        # Save current time
        start_time1 = time()

        # Check what kind of hdf5-file has been provided
        self._emul_i = self._emulator._get_emul_i(emul_i, True)

        # Obtain figure name prefix
        self._fig_prefix = path.join(self._pipeline._working_dir,
                                     '%s_proj_' % (self._emul_i))

        # Check if it makes sense to create a projection
        if(self._emul_i == self._emulator._emul_i):
            if not self._pipeline._n_eval_sam[self._emul_i]:
                logger.info("Requested emulator iteration %s has not been "
                            "analyzed yet. Creating projections may not be "
                            "useful." % (self._emul_i))
                print("Requested emulator iteration %s has not been analyzed "
                      "yet. Creating projections may not be useful."
                      % (self._emul_i))
            elif self._pipeline._prc:
                pass
            else:
                logger.info("Requested emulator iteration %s has no plausible "
                            "regions. Creating projections has no use."
                            % (self._emul_i))
                print("Requested emulator iteration %s has no plausible "
                      "regions. Creating projections has no use."
                      % (self._emul_i))
                return

        # Check if figure, show and force-parameters are bools
        self._figure = check_bool(figure, 'figure')
        self._show = check_bool(show, 'show')
        self._force = check_bool(force, 'force')

        # Get the impl_cut list and proj_res/proj_depth
        # TODO: Make sure that the same impl_cut is used for all figures
        # TODO: If desync is True, maybe not require force parameter?
        self._get_impl_par()

        # Obtain requested projection hypercubes
        hcube_par, create_hcube_par = self._get_req_hcubes(proj_par)

        # Save current time again
        start_time2 = time()

        # Loop over all requested projection hypercubes
        for hcube in tqdm(hcube_par, desc="Creating projections",
                          unit='hcube'):
            # Initialize impl_min and impl_los
            impl_min = None
            impl_los = None

            # Obtain name of this hypercube and save it in memory
            if(self._modellink._n_par == 2):
                self._hcube_name = ('%s'
                                    % (self._modellink._par_name[hcube[0]]))
            else:
                self._hcube_name = ('%s-%s'
                                    % (self._modellink._par_name[hcube[0]],
                                       self._modellink._par_name[hcube[1]]))

            # ANALYZE PROJECTION HYPERCUBE
            # Create projection hypercube containing all samples if required
            if hcube in create_hcube_par:
                # Log that projection data is being created
                logger.info("Calculating projection data '%s'."
                            % (self._hcube_name))

                # Obtain the corresponding hypercube
                proj_hcube = self._get_proj_hcube(hcube)

                # Analyze this proj_hcube
                impl_min, impl_los = self._analyze_proj_hcube(proj_hcube)

                # Log that projection data has been created
                logger.info("Finished calculating projection data '%s'."
                            % (self._hcube_name))

                # Save projection data to hdf5
                self._save_data({
                    'nD_proj_hcube': [self._hcube_name, impl_min, impl_los]})

            # PLOTTING
            if self._figure:
                # Determine the path of this figure
                fig_path = '%s(%s).png' % (self._fig_prefix, self._hcube_name)

                # Skip making figure if it already exists
                if path.exists(fig_path):
                    logger.info("Projection figure '%s' already exists. "
                                "Skipping figure creation."
                                % (self._hcube_name))
                    continue

                # If projection data is not already loaded, load it
                if impl_min is None and impl_los is None:
                    # Open hdf5-file
                    with PRISM_File('r', None) as file:
                        # Log that projection data is being obtained
                        logger.info("Obtaining projection data '%s'."
                                    % (self._hcube_name))

                        # Obtain data
                        impl_los = file['%s/proj_hcube/%s/impl_los'
                                        % (self._emul_i, self._hcube_name)][()]
                        impl_min = file['%s/proj_hcube/%s/impl_min'
                                        % (self._emul_i, self._hcube_name)][()]

                        # Log that projection data was obtained successfully
                        logger.info("Finished obtaining projection data '%s'."
                                    % (self._hcube_name))

                # Draw projection figure
                if(self._modellink._n_par == 2):
                    self._draw_2D_proj_fig(hcube, impl_min, impl_los)
                else:
                    self._draw_3D_proj_fig(hcube, impl_min, impl_los)

        # Log the end of the projection
        end_time = time()
        time_diff_total = end_time-start_time1
        time_diff_figs = end_time-start_time2

        logger.info("Finished projection in %.2f seconds, averaging %.2f "
                    "seconds per projection %s."
                    % (time_diff_total, time_diff_figs/len(hcube_par),
                       'figure' if figure else 'hypercube'))
        print("")


# %% CLASS PROPERTIES
    @property
    def proj_res(self):
        """
        Number of emulator evaluations used to generate the grid for the
        projection figures.

        """

        return(self._res)

    @property
    def proj_depth(self):
        """
        Number of emulator evaluations used to generate the samples in every
        grid point for the projection figures.

        """

        return(self._depth)

    @property
    def impl_cut(self):
        """
        List containing all univariate implausibility cut-offs. A zero
        indicates a wildcard.

        """

        return(self._impl_cut)

    @property
    def cut_idx(self):
        """
        The list index of the first non-wildcard cut-off in impl_cut.

        """

        return(self._cut_idx)


# %% HIDDEN CLASS METHODS
    # This function draws the 2D projection figure
    def _draw_2D_proj_fig(self, hcube, impl_min, impl_los):
        """
        Draws the 2D projection figure for the provided `hcube`, given the
        `impl_min` and `impl_los` values.

        Parameters
        ----------
        hcube : 1D array_like of int of length 1
            Array containing the internal integer identifier of the main model
            parameter that requires a projection figure.
        impl_min : 1D list
            List containing the lowest implausibility value that can be reached
            in every single grid point on the given hypercube.
        impl_los : 1D list
            List containing the fraction of the total amount of evaluated
            samples in every single grid point on the given hypercube, that
            still satisfied the implausibility cut-off criterion.

        """

        # Start logger
        logger = getLogger('PROJECTION')
        logger.info("Drawing projection figure '%s'." % (self._hcube_name))

        # Determine the path of this figure
        fig_path = '%s(%s).png' % (self._fig_prefix, self._hcube_name)

        # Get the parameter this hypercube is about
        par = hcube[0]

        # Make abbreviation for parameter name
        par_name = self._modellink._par_name[par]

        # Recreate the parameter value array used to create the hcube
        proj_sam_set = np.linspace(self._modellink._par_rng[par, 0],
                                   self._modellink._par_rng[par, 1], self._res)

        # Create parameter value array used in interpolation functions
        x = np.linspace(self._modellink._par_rng[par, 0],
                        self._modellink._par_rng[par, 1], 1000)

        # Get the interpolated functions describing the minimum
        # implausibility and line-of-sight depth obtained in every
        # point
        f_min = interp1d(proj_sam_set, impl_min, kind='cubic')
        f_los = interp1d(proj_sam_set, impl_los, kind='cubic')

        # Do plotting
        f, axarr = plt.subplots(2)
        plt.rc('text', usetex=True)
        f.suptitle(r"%s. Projection (%s)" % (self._emul_i, self._hcube_name),
                   fontsize='xx-large')

        # Plot minimum implausibility
        axarr[0].plot(x, f_min(x))
        draw_y = self._impl_cut[self._emul_i][self._cut_idx[self._emul_i]]
        draw_textline(r"$I_{\mathrm{cut-off, 1}}$", y=draw_y, ax=axarr[0],
                      line_kwargs={'color': 'g'})
        if self._modellink._par_est[par] is not None:
            draw_textline(r"", x=self._modellink._par_est[par], ax=axarr[0],
                          line_kwargs={'linestyle': '--'})
        axarr[0].set_ylabel(r"Minimum Implausibility $I_{\mathrm{min}}(%s)$"
                            % (par_name), fontsize='x-large')

        # Plot line-of-sight depth
        axarr[1].plot(x, f_los(x))
        if self._modellink._par_est[par] is not None:
            draw_textline(r"", x=self._modellink._par_est[par], ax=axarr[1],
                          line_kwargs={'linestyle': '--'})
        axarr[1].set_xlabel("Input Parameter %s" % (par_name),
                            fontsize='x-large')
        axarr[1].set_ylabel("Line-of-Sight Depth", fontsize='x-large')

        # Save the figure
        plt.savefig(fig_path)

        # If show is set to True, show the figure
        f.show() if self._show else plt.close(f)

        # Log that this hypercube has been drawn
        logger.info("Finished drawing projection figure '%s'."
                    % (self._hcube_name))

    # This function draws the 3D projection figure
    def _draw_3D_proj_fig(self, hcube, impl_min, impl_los):
        """
        Draws the #D projection figure for the provided `hcube`, given the
        `impl_min` and `impl_los` values.

        Parameters
        ----------
        hcube : 1D array_like of int of length 2
            Array containing the internal integer identifiers of the main model
            parameters that require a projection figure.
        impl_min : 1D list
            List containing the lowest implausibility value that can be reached
            in every single grid point on the given hypercube.
        impl_los : 1D list
            List containing the fraction of the total amount of evaluated
            samples in every single grid point on the given hypercube, that
            still satisfied the implausibility cut-off criterion.

        """

        # Start logger
        logger = getLogger('PROJECTION')
        logger.info("Drawing projection figure '%s'." % (self._hcube_name))

        # Determine the path of this figure
        fig_path = '%s(%s).png' % (self._fig_prefix, self._hcube_name)

        # Get the parameter on x-axis and y-axis this hcube is about
        par1 = hcube[0]
        par2 = hcube[1]

        # Make abbreviation for the parameter names
        par1_name = self._modellink._par_name[par1]
        par2_name = self._modellink._par_name[par2]

        # Recreate the parameter value arrays used to create the hcube
        proj_sam_set1 = np.linspace(self._modellink._par_rng[par1, 0],
                                    self._modellink._par_rng[par1, 1],
                                    self._res)
        proj_sam_set2 = np.linspace(self._modellink._par_rng[par2, 0],
                                    self._modellink._par_rng[par2, 1],
                                    self._res)

        # Create parameter value array used in interpolation functions
        x = np.linspace(self._modellink._par_rng[par1, 0],
                        self._modellink._par_rng[par1, 1], 1000)
        y = np.linspace(self._modellink._par_rng[par2, 0],
                        self._modellink._par_rng[par2, 1], 1000)

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
        f.suptitle(r"%s. Projection (%s)" % (self._emul_i, self._hcube_name),
                   fontsize='xx-large')

        # Plot minimum implausibility
        vmax = self._impl_cut[self._emul_i][self._cut_idx[self._emul_i]]
        fig1 = axarr[0].hexbin(x, y, z_min, gridsize, vmin=0, vmax=vmax,
                               cmap=cm.get_cmap('rainforest_r'))
        if self._modellink._par_est[par1] is not None:
            draw_textline(r"", x=self._modellink._par_est[par1], ax=axarr[0],
                          line_kwargs={'linestyle': '--', 'color': 'grey'})
        if self._modellink._par_est[par2] is not None:
            draw_textline(r"", y=self._modellink._par_est[par2], ax=axarr[0],
                          line_kwargs={'linestyle': '--', 'color': 'grey'})
        axarr[0].axis([self._modellink._par_rng[par1, 0],
                       self._modellink._par_rng[par1, 1],
                       self._modellink._par_rng[par2, 0],
                       self._modellink._par_rng[par2, 1]])
        plt.colorbar(fig1, ax=axarr[0]).set_label("Minimum Implausibility",
                                                  fontsize='large')

        # Plot line-of-sight depth
        fig2 = axarr[1].hexbin(x, y, z_los, gridsize,
                               cmap=cm.get_cmap('blaze'), vmin=0,
                               vmax=min(1, np.max(z_los)))
        if self._modellink._par_est[par1] is not None:
            draw_textline(r"", x=self._modellink._par_est[par1], ax=axarr[1],
                          line_kwargs={'linestyle': '--', 'color': 'gray'})
        if self._modellink._par_est[par2] is not None:
            draw_textline(r"", y=self._modellink._par_est[par2], ax=axarr[1],
                          line_kwargs={'linestyle': '--', 'color': 'gray'})
        axarr[1].axis([self._modellink._par_rng[par1, 0],
                       self._modellink._par_rng[par1, 1],
                       self._modellink._par_rng[par2, 0],
                       self._modellink._par_rng[par2, 1]])
        plt.colorbar(fig2, ax=axarr[1]).set_label("Line-of-Sight Depth",
                                                  fontsize='large')

        # Make super axis labels
        axarr[1].set_xlabel("%s" % (par1_name), fontsize='x-large')
        suplabel("%s" % (par2_name), axis='y', fig=f, fontsize='x-large')

        # Save the figure
        plt.savefig(fig_path)

        # If show is set to True, show the figure
        f.show() if self._show else plt.close(f)

        # Log that this hypercube has been drawn
        logger.info("Finished drawing projection figure '%s'."
                    % (self._hcube_name))

    # This function determines the projection hypercubes to be analyzed
    def _get_req_hcubes(self, proj_par):
        """
        Determines which projection hypercubes have been requested by the user.
        Also checks if these projection hypercubes have been calculated before,
        and depending on the value of `force`, either skips them or recreates
        them.

        Parameters
        ----------
        proj_par : 1D array_like of {int, str} or None
            For which model parameters to construct the projection figures.
            If 1D array_like, construct projection figures for all combinations
            of active model parameters.
            If 1D array_like of str, the strings refer to the names of the
            model parameters.
            If 1D array_like of int, the integers refer to the order in which
            the model parameters are shown in the :meth:`~details` method.
            If *None*, projection figures are made for all model parameters.

        Returns
        -------
        hcube_par : list of lists
            List containing the parameter indices of the requested projection
            hypercubes.
        create_hcube_par : list of lists
            List containing the parameter indices of the requested projection
            hypercubes that need to be created first.

        """

        # Start logger
        logger = getLogger('PROJECTION')

        # Check the proj_par that were provided
        # If none was provided, make figures for all model parameters
        if proj_par is None:
            proj_par = self._emulator._active_par[self._emul_i]

        # Else, an array of str/int must be provided
        else:
            # Convert to string, remove unwanted characters and split it up
            proj_par = convert_str_seq(proj_par)

            # Check elements if they are ints or strings, and if they are valid
            for i, val in enumerate(proj_par):
                try:
                    try:
                        par_idx = int(val)
                    except ValueError:
                        proj_par[i] = self._modellink._par_name.index(val)
                    else:
                        self._modellink._par_name[par_idx]
                        proj_par[i] = par_idx % self._modellink._n_par
                except Exception as error:
                    logger.error("Input argument 'proj_par' is invalid! (%s)"
                                 % (error))
                    raise InputError("Input argument 'proj_par' is invalid! "
                                     "(%s)" % (error))

            # If everything went without exceptions, remove duplicates and sort
            proj_par = list(SortedSet(proj_par))

            # Check which values in proj_par are also in active_par
            proj_par = np.array(
                [i for i in self._emulator._active_par[self._emul_i] if
                 i in proj_par])

            # Make sure that there are still enough values left
            if(self._modellink._n_par == 2 and len(proj_par) >= 1):
                pass
            elif(self._modellink._n_par > 2 and len(proj_par) >= 2):
                pass
            else:
                logger.error("Not enough active model parameters have been "
                             "provided to make a projection figure!")
                raise RequestError("Not enough active model parameters have "
                                   "been provided to make a projection "
                                   "figure!")

        # Obtain list of hypercube names
        if(self._modellink._n_par == 2):
            hcube_idx = list(combinations(range(len(proj_par)), 1))
        else:
            hcube_idx = list(combinations(range(len(proj_par)), 2))
        hcube_par = proj_par[np.array(hcube_idx)].tolist()

        # Create empty list holding hcube_par that needs to be created
        create_hcube_par = []

        # Open hdf5-file
        logger.info("Checking if projection data already exists.")
        with PRISM_File('r+', None) as file:
            # Check if data is already there and act accordingly
            for hcube in hcube_par:
                # Obtain name of this hypercube
                if(self._modellink._n_par == 2):
                    hcube_name = ('%s' % (self._modellink._par_name[hcube[0]]))
                else:
                    hcube_name = ('%s-%s'
                                  % (self._modellink._par_name[hcube[0]],
                                     self._modellink._par_name[hcube[1]]))

                # Check if projection data already exists
                try:
                    file['%s/proj_hcube/%s' % (self._emul_i, hcube_name)]

                # If it does not exist, add it to the creation list
                except KeyError:
                    logger.info("Projection data '%s' not found. Will be "
                                "created." % (hcube_name))
                    create_hcube_par.append(hcube)

                # If it does exist, check value of force
                else:
                    # If force is used, remove data and figure
                    if self._force:
                        # Remove data
                        del file['%s/proj_hcube/%s'
                                 % (self._emul_i, hcube_name)]
                        logger.info("Projection data '%s' already exists. "
                                    "Deleting." % (hcube_name))

                        # Try to remove figure as well
                        try:
                            os.remove('%s(%s).png'
                                      % (self._fig_prefix, hcube_name))
                        except OSError:
                            pass
                        else:
                            logger.info("Projection figure '%s' already "
                                        "exists. Deleting.")

                        # Add this hypercube to creation list
                        create_hcube_par.append(hcube)

                    # If force is not used, skip creation
                    else:
                        logger.info("Projection data '%s' already exists. "
                                    "Skipping data creation." % (hcube_name))

        # Return requested proj_hcubes and those that need to be created
        return(hcube_par, create_hcube_par)

    # This function reads in the impl_cut list from the PRISM parameters file
    def _get_impl_par(self):
        """
        Reads in the impl_cut list and other parameters for implausibility
        evaluations from the PRISM parameters file and saves them in the
        emulator iteration this class was initialized for.

        Generates
        ---------
        impl_cut : 1D :obj:`~numpy.ndarray` object
            Full list containing the impl_cut-offs for all data points provided
            to the emulator.
        cut_idx : int
            Index of the first impl_cut-off in the impl_cut list that is not
            0.
        proj_res : int
            Number of emulator evaluations used to generate the grid for the
            projection figures.
        proj_depth : int
            Number of emulator evaluations used to generate the samples in
            every grid point for the projection figures.

        """

        # Do some logging
        logger = getCLogger('INIT')
        logger.info("Obtaining implausibility analysis parameters.")

        # Obtain the impl_cut and cut_idx up to this iteration from pipeline
        self._impl_cut = list(self._pipeline._impl_cut[:self._emul_i])
        self._cut_idx = list(self._pipeline._cut_idx[:self._emul_i])

        # Obtaining default pipeline parameter dict
        par_dict = self._get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._pipeline._prism_file is not None:
            pipe_par = np.genfromtxt(self._pipeline._prism_file, dtype=(str),
                                     delimiter=':', autostrip=True)

            # Make sure that pipe_par is 2D
            pipe_par = np.array(pipe_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(pipe_par)

        # More logging
        logger.info("Checking compatibility of provided implausibility "
                    "analysis parameters.")

        # Implausibility cut-off
        # Remove all unwanted characters from the string and split it up
        impl_cut_str = convert_str_seq(par_dict['impl_cut'])

        # Convert list of strings to list of floats and perform completion
        self._pipeline._get_impl_cut(
            self, list(float(impl_cut) for impl_cut in impl_cut_str), False)

        # Number of samples used for implausibility evaluations
        proj_res = int(par_dict['proj_res'])
        proj_depth = int(par_dict['proj_depth'])
        self._save_data({
            'proj_grid': [check_pos_int(proj_res, 'proj_res'),
                          check_pos_int(proj_depth, 'proj_depth')]})

        # Finish logging
        logger.info("Finished obtaining implausibility analysis parameters.")

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
        logger = getCLogger('INIT')
        logger.info("Generating default projection parameter dict.")

        # Create parameter dict with default parameters
        par_dict = {'proj_res': '15',
                    'proj_depth': '150',
                    'impl_cut': '[0, 4.0, 3.8, 3.5]'}

        # Log end
        logger.info("Finished generating default projection parameter dict.")

        # Return it
        return(par_dict)

    # This function generates a projection hypercube to be used for emulator
    # evaluations
    def _get_proj_hcube(self, hcube):
        """
        Generates a hypercube containing emulator evaluation samples to be used
        in the :class:`~Projection` class. The output of this function depends
        on the number of model parameters.

        Parameters
        ----------
        hcube : 1D array_like of int of length {1, 2}
            Array containing the internal integer identifiers of the main model
            parameters that require a projection hypercube.

        Returns
        -------
        proj_hcube : 3D :obj:`~numpy.ndarray` object
            3D projection hypercube of emulator evaluation samples.

        """

        # Log that projection hypercube is being created
        logger = getLogger('PROJ_HCUBE')
        logger.info("Creating projection hypercube '%s'." % (self._hcube_name))

        # If n_par is 2, make 2D projection hypercube
        if(self._modellink._n_par == 2):
            # Identify projected parameter
            par = hcube[0]

            # Create empty projection hypercube array
            proj_hcube = np.zeros([self._res, self._depth, 2])

            # Create list that contains all the other parameters
            par_hid = 1 if par == 0 else 0

            # Generate list with values for projected parameter
            proj_sam_set = np.linspace(self._modellink._par_rng[par, 0],
                                       self._modellink._par_rng[par, 1],
                                       self._res)

            # Generate latin hypercube of the remaining parameters
            hidden_sam_set = lhd(self._depth, 1,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._pipeline._criterion)

            # Fill every cell in the projection hypercube accordingly
            for i in range(self._res):
                proj_hcube[i, :, par] = proj_sam_set[i]
                proj_hcube[i, :, par_hid] = hidden_sam_set

        # If n_par is more than 2, make 3D projection hypercube
        else:
            # Identify projected parameters
            par1 = hcube[0]
            par2 = hcube[1]

            # Create empty projection hypercube array
            proj_hcube = np.zeros([pow(self._res, 2), self._depth,
                                   self._modellink._n_par])

            # Generate list that contains all the other parameters
            par_hid = list(chain(range(0, par1), range(par1+1, par2),
                                 range(par2+1, self._modellink._n_par)))

            # Generate list with values for projected parameters
            proj_sam_set1 = np.linspace(self._modellink._par_rng[par1, 0],
                                        self._modellink._par_rng[par1, 1],
                                        self._res)
            proj_sam_set2 = np.linspace(self._modellink._par_rng[par2, 0],
                                        self._modellink._par_rng[par2, 1],
                                        self._res)

            # Generate Latin Hypercube of the remaining parameters
            hidden_sam_set = lhd(self._depth, self._modellink._n_par-2,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._pipeline._criterion)

            # Fill every cell in the projection hypercube accordingly
            for i in range(self._res):
                for j in range(self._res):
                    proj_hcube[i*self._res+j, :, par1] = proj_sam_set1[i]
                    proj_hcube[i*self._res+j, :, par2] = proj_sam_set2[j]
                    proj_hcube[i*self._res+j, :, par_hid] = hidden_sam_set.T

        # Log that projection hypercube has been created
        logger.info("Finished creating projection hypercube '%s'."
                    % (self._hcube_name))

        # Return proj_hcube
        return(proj_hcube)

    # This function analyzes a projection hypercube
    def _analyze_proj_hcube(self, proj_hcube):
        """
        Analyzes an emulator projection hypercube used in the
        :class:`~Projection` class.

        Parameters
        ----------
        proj_hcube : 3D array_like
            3D projection hypercube containing all the emulator evaluation
            samples that are required to make the projection figure.

        Returns
        -------
        impl_min_hcube : 1D list
            List containing the lowest implausibility value that can be reached
            in every single grid point on the given hypercube.
        impl_los_hcube : 1D list
            List containing the fraction of the total amount of evaluated
            samples in every single grid point on the given hypercube, that
            still satisfied the implausibility cut-off criterion.

        """

        # Log that a projection hypercube is being analyzed
        logger = getLogger('ANALYSIS')
        logger.info("Analyzing projection hypercube.")

        # Save current time
        start_time = time()

        # CALCULATE AND ANALYZE IMPLAUSIBILITY
        # Create empty lists for this hypercube
        impl_min_hcube = []
        impl_los_hcube = []

        # Define the various code snippets
        pre_code = compile("impl_cut = []", '<string>', 'exec')
        eval_code = compile("", '<string>', 'exec')
        anal_code = compile("impl_cut.append(impl_cut_val)", '<string>',
                            'exec')
        post_code = compile("self.results = (impl_check, impl_cut)",
                            '<string>', 'exec')
        exit_code = compile("self.results = (impl_check, impl_cut)",
                            '<string>', 'exec')

        # Combine code snippets into a tuple
        exec_code = (pre_code, eval_code, anal_code, post_code, exit_code)

        # Iterate over all samples in the hcube
        total = proj_hcube.shape[0]*self._depth
        with tqdm(desc="Analyzing hypercube ", total=proj_hcube.shape[0],
                  unit='gp') as pbar:

            # For all grid points in the hcube
            for i, grid_point in enumerate(proj_hcube):
                # Analyze grid_point
                impl_check, impl_cut =\
                    self._pipeline._analyze_sam_set(self, self._emul_i,
                                                    grid_point, *exec_code)

                # If a grid point has been checked, save lowest impl and impl
                # line-of-sight
                # Calculate lowest impl in this grid point
                impl_min_hcube.append(min(impl_cut))

                # Calculate impl line-of-sight in this grid point
                impl_los_hcube.append(sum(impl_check)/self._depth)

                # Advance progressbar
                pbar.update()
                pbar.set_postfix_str('%.2feval/s'
                                     % (self._depth/pbar.avg_time))

        # Log that analysis has been finished
        time_diff = time()-start_time
        logger.info("Finished projection hypercube analysis in %.2f seconds, "
                    "averaging %.2f emulator evaluations per second."
                    % (time_diff, total/(time_diff)))

        # Return impl_min and impl_los
        return(impl_min_hcube, impl_los_hcube)

    # This function saves projection data to hdf5
    def _save_data(self, data_dict):
        """
        Saves a given data dict ``{keyword: data}`` at the emulator iteration
        this class was initialized for, to the HDF5-file.

        Parameters
        ----------
        data_dict : dict
            Dict containing the data that needs to be saved to the HDF5-file.

        Dict Variables
        --------------
        keyword : {'impl_cut', 'proj_grid', 'nD_proj_hcube'}
            String specifying the type of data that needs to be saved.
        data : list
            The actual data that needs to be saved at data keyword `keyword`.

        Generates
        ---------
        The specified data is saved to the HDF5-file.

        """

        # Do some logging
        logger = getLogger('SAVE_DATA')

        # Open hdf5-file
        with PRISM_File('r+', None) as file:
            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %s data at iteration %s to HDF5."
                            % (keyword, self._emul_i))

                # Check what data keyword has been provided
                # IMPL_CUT
                if(keyword == 'impl_cut'):
                    # Check if projection has been created before
                    try:
                        file.create_group('%s/proj_hcube' % (self._emul_i))
                    except ValueError:
                        pass

                    self._impl_cut.append(data[0])
                    self._cut_idx.append(data[1])
                    file['%s/proj_hcube' % (self._emul_i)].attrs['impl_cut'] =\
                        data[0]
                    file['%s/proj_hcube' % (self._emul_i)].attrs['cut_idx'] =\
                        data[1]

                # PROJ_GRID
                elif(keyword == 'proj_grid'):
                    # Check if projection has been created before
                    try:
                        file.create_group('%s/proj_hcube' % (self._emul_i))
                    except ValueError:
                        pass

                    self._res = data[0]
                    self._depth = data[1]
                    file['%s/proj_hcube' % (self._emul_i)].attrs['proj_res'] =\
                        data[0]
                    file['%s/proj_hcube'
                         % (self._emul_i)].attrs['proj_depth'] = data[1]

                # ND_PROJ_HCUBE
                elif(keyword == 'nD_proj_hcube'):
                    file.create_dataset('%s/proj_hcube/%s/impl_min'
                                        % (self._emul_i, data[0]),
                                        data=data[1])
                    file.create_dataset('%s/proj_hcube/%s/impl_los'
                                        % (self._emul_i, data[0]),
                                        data=data[2])

                # INVALID KEYWORD
                else:
                    logger.error("Invalid keyword argument provided!")
                    raise ValueError("Invalid keyword argument provided!")
