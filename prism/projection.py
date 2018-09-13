# -*- coding: utf-8 -*-

"""
Projection
==========
Provides the definition of *PRISM*'s :class:`~Projection` class, a
:class:`~Pipeline` base class that allows for projection figures detailing a
model's behavior to be created.


Available classes
-----------------
:class:`~Projection`
    Defines the :class:`~Projection` class of the *PRISM* package.

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
from e13tools.core import InputError
from e13tools.pyplot import draw_textline, suplabel
from e13tools.sampling import lhd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
# TODO: Do some research on scipy.interpolate.Rbf later
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm

# PRISM imports
from ._docstrings import (def_par_doc, draw_proj_fig_doc, hcube_doc,
                          proj_par_doc_d, proj_par_doc_s, read_par_doc,
                          save_data_doc_pr, user_emul_i_doc)
from ._internal import (PRISM_File, RequestError, check_val,
                        docstring_append, docstring_substitute, getCLogger,
                        raise_error)

# All declaration
__all__ = ['Projection']


# %% PROJECTION CLASS DEFINITION
# TODO: Make something like Figure 4 in Bower et al
class Projection(object):
    """
    Defines the :class:`~Projection` class of the *PRISM* package.

    Description
    -----------
    The :class:`~Projection` class holds all specific methods that the
    :class:`~Pipeline` class needs in order to create projections of the model.

    This is a base class for the :class:`~Pipeline` class and cannot be used on
    its own.

    """

    # Determine the fig_prefix
    _fig_prefix = path.join('{0}', '{1}_proj_')

    # Create __init__ method to warn if this class is every initialized
    def __init__(self, *args, **kwargs):
        logger = getCLogger('PROJECTION')
        err_msg = ("The Projection class is a base class for the Pipeline "
                   "class and cannot be used on its own!")
        raise_error(RequestError, err_msg, logger)

    # Function that creates all projection figures
    @docstring_substitute(emul_i=user_emul_i_doc, proj_par=proj_par_doc_d)
    def project(self, *args, **kwargs):
        """
        Analyzes the emulator iteration `emul_i` and constructs a series of
        projection figures detailing the behavior of the model parameters
        corresponding to the given `proj_par`.
        The input and output depend on the number of model parameters
        :attr:`~ModelLink.n_par`.

        Positional/keyword arguments
        ----------------------------
        %(emul_i)s
        %(proj_par)s

        Keyword arguments
        -----------------
        figure : bool. Default: True
            Whether or not to create the projection figures. If *False*, only
            the data required to create the figures is calculated and saved in
            the HDF5-file, but the figures themselves are not made.
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
        fig_kwargs : dict. Default: {'figsize': (6.4, 4.8), 'dpi': 100}
            Dict of keyword arguments to be used when creating the subplots
            figure. It takes all arguments that can be provided to the
            :func:`~matplotlib.pyplot.subplots` function.
        impl_kwargs : dict. Default: {} or {'cmap': 'rainforest_r'}
            Dict of keyword arguments to be used for making the minimum
            implausibility (top) plot. It takes all arguments that can be
            provided to the :func:`~matplotlib.pyplot.plot` or
            :func:`~matplotlib.pyplot.hexbin` function.
        los_kwargs : dict. Default: {} or {'cmap': 'blaze'}
            Dict of keyword arguments to be used for making the line-of-sight
            (bottom) plot. It takes all arguments that can be provided to the
            :func:`~matplotlib.pyplot.plot` or
            :func:`~matplotlib.pyplot.hexbin` function.
        line_kwargs : dict. Default: {'linestyle': '--', 'color': 'grey'}
            Dict of keyword arguments to be used for drawing the parameter
            estimate lines in both plots. It takes all arguments that can be
            provided to the :func:`~matplotlib.pyplot.draw` function.

        Generates
        ---------
        A series of projection figures detailing the behavior of the model.
        The lay-out and output of the projection figures depend on the number
        of model parameters :attr:`~ModelLink.n_par`:
            :attr:`~ModelLink.n_par` == 2: The output will feature two figures
            for the two model parameters with two subplots each. Every figure
            gives details about the behavior of the corresponding model
            parameter, by showing the minimum implausibility value (top) and
            the line-of-sight depth (bottom) obtained at the specified
            parameter value, independent of the value of the other parameter.

            :attr:`~ModelLink.n_par` > 2: The output will feature a figure with
            two subplots for every combination of two active model parameters
            that can be made (``n_par*(n_par-1)/2``). Every figure gives
            details about the behavior of the corresponding model parameters,
            as well as their dependency on each other. This is done by showing
            the minimum implausibility (top) and the line-of-sight depth
            (bottom) obtained at the specified parameter values, independent of
            the values of the remaining model parameters.

        Notes
        -----
        If given emulator iteration `emul_i` has been analyzed before, the
        implausibility parameters of the last analysis are used. If not, then
        the values are used that were read in when the emulator was loaded.

        """

        # Log the start of the creation of the projection
        logger = getCLogger('PROJECTION')
        logger.info("Starting the projection process.")

        # Save current time
        start_time1 = time()

        # Prepare for making projections
        self.__prepare_projections(*args, **kwargs)

        # Save current time again
        start_time2 = time()

        # Loop over all requested projection hypercubes
        if self._is_controller:
            hcubes_bar = tqdm(self.__hcubes, desc="Creating projections",
                              unit='hcube')
        else:
            hcubes_bar = self.__hcubes
        for hcube in hcubes_bar:
            # Initialize impl_min and impl_los
            impl_min = None
            impl_los = None

            # Obtain name of this hypercube
            hcube_name = self.__get_hcube_name(hcube)

            # ANALYZE PROJECTION HYPERCUBE
            # Analyze projection hypercube containing all samples if required
            if hcube in self.__create_hcubes:
                # Log that projection data is being created
                logger.info("Calculating projection data '%s'." % (hcube_name))

                # Analyze this proj_hcube
                impl_min, impl_los = self.__analyze_proj_hcube(hcube)

            # PLOTTING (CONTROLLER ONLY)
            # Draw projection figure if requested
            if self._is_controller and self.__figure:
                # Determine the path of this figure
                self.__fig_path = '%s(%s).png' % (self.__fig_prefix,
                                                  hcube_name)

                # Skip making figure if it already exists
                if path.exists(self.__fig_path):
                    logger.info("Projection figure '%s' already exists. "
                                "Skipping figure creation." % (hcube_name))
                    self._comm.Barrier()
                    continue

                # If projection data is not already loaded, load it
                if impl_min is None and impl_los is None:
                    # Open hdf5-file
                    with PRISM_File('r', None) as file:
                        # Log that projection data is being obtained
                        logger.info("Obtaining projection data '%s'."
                                    % (hcube_name))

                        # Obtain data
                        data_set = file['%s/proj_hcube' % (self.__emul_i)]
                        impl_los = data_set['%s/impl_los' % (hcube_name)][()]
                        impl_min = data_set['%s/impl_min' % (hcube_name)][()]

                        # Log that projection data was obtained successfully
                        logger.info("Finished obtaining projection data '%s'."
                                    % (hcube_name))

                # Draw projection figure
                if(self._modellink._n_par == 2):
                    self.__draw_2D_proj_fig(hcube, impl_min, impl_los)
                else:
                    self.__draw_3D_proj_fig(hcube, impl_min, impl_los)

            # MPI Barrier
            self._comm.Barrier()

        # Controller logging end of the projection
        if self._is_controller:
            end_time = time()
            time_diff1 = end_time-start_time1
            time_diff2 = end_time-start_time2
            logger.info("Finished projection in %.2f seconds, averaging %.2f "
                        "seconds per projection %s."
                        % (time_diff1, time_diff2/len(self.__hcubes),
                           'figure' if self.__figure else 'hypercube'))
            print("")

        # Show details
        self.details()

    # %% CLASS PROPERTIES
    @property
    def proj_res(self):
        """
        Number of emulator evaluations used to generate the grid for the
        projection figures.

        """

        return(self.__res)

    @property
    def proj_depth(self):
        """
        Number of emulator evaluations used to generate the samples in every
        grid point for the projection figures.

        """

        return(self.__depth)

    # %% HIDDEN CLASS METHODS
    # This function draws the 2D projection figure
    @docstring_append(draw_proj_fig_doc.format("2D", "1"))
    def __draw_2D_proj_fig(self, hcube, impl_min, impl_los):
        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Start logger
        logger = getLogger('PROJECTION')
        logger.info("Drawing projection figure '%s'." % (hcube_name))

        # Get the parameter this hypercube is about
        par = hcube[0]

        # Make abbreviation for parameter name
        par_name = self._modellink._par_name[par]

        # Recreate the parameter value array used to create the hcube
        proj_sam_set = np.linspace(self._modellink._par_rng[par, 0],
                                   self._modellink._par_rng[par, 1],
                                   self.__res)

        # Create parameter value array used in interpolation functions
        x = np.linspace(self._modellink._par_rng[par, 0],
                        self._modellink._par_rng[par, 1], 1000)

        # Get the interpolated functions describing the minimum
        # implausibility and line-of-sight depth obtained in every
        # point
        f_min = interp1d(proj_sam_set, impl_min, kind='cubic')
        f_los = interp1d(proj_sam_set, impl_los, kind='cubic')

        # Do plotting
        f, axarr = plt.subplots(2, **self.__fig_kwargs)
        f.suptitle(r"%s. Projection (%s)" % (self.__emul_i, hcube_name),
                   fontsize='xx-large')

        # Plot minimum implausibility
        axarr[0].plot(x, f_min(x), **self.__impl_kwargs)
        draw_y = self._impl_cut[self.__emul_i][self._cut_idx[self.__emul_i]]
        axarr[0].axis([self._modellink._par_rng[par, 0],
                       self._modellink._par_rng[par, 1],
                       0, 1.5*draw_y])
        if self._modellink._par_est[par] is not None:
            draw_textline(r"", x=self._modellink._par_est[par], ax=axarr[0],
                          line_kwargs=self.__line_kwargs)
        draw_textline(r"", y=draw_y, ax=axarr[0], line_kwargs={'color': 'g'})
        axarr[0].set_ylabel("Minimum Implausibility", fontsize='large')

        # Plot line-of-sight depth
        axarr[1].plot(x, f_los(x), **self.__los_kwargs)
        axarr[1].axis([self._modellink._par_rng[par, 0],
                       self._modellink._par_rng[par, 1],
                       0, min(1, np.max(impl_los))])
        if self._modellink._par_est[par] is not None:
            draw_textline(r"", x=self._modellink._par_est[par], ax=axarr[1],
                          line_kwargs=self.__line_kwargs)
        axarr[1].set_xlabel("Input Parameter %s" % (par_name),
                            fontsize='x-large')
        axarr[1].set_ylabel("Line-of-Sight Depth", fontsize='large')

        # Save the figure
        plt.savefig(self.__fig_path)

        # If show is set to True, show the figure
        f.show() if self.__show else plt.close(f)

        # Log that this hypercube has been drawn
        logger.info("Finished drawing projection figure '%s'." % (hcube_name))

    # This function draws the 3D projection figure
    @docstring_append(draw_proj_fig_doc.format("3D", "2"))
    def __draw_3D_proj_fig(self, hcube, impl_min, impl_los):
        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Start logger
        logger = getLogger('PROJECTION')
        logger.info("Drawing projection figure '%s'." % (hcube_name))

        # Get the parameter on x-axis and y-axis this hcube is about
        par1 = hcube[0]
        par2 = hcube[1]

        # Make abbreviation for the parameter names
        par1_name = self._modellink._par_name[par1]
        par2_name = self._modellink._par_name[par2]

        # Recreate the parameter value arrays used to create the hcube
        proj_sam_set1 = np.linspace(self._modellink._par_rng[par1, 0],
                                    self._modellink._par_rng[par1, 1],
                                    self.__res)
        proj_sam_set2 = np.linspace(self._modellink._par_rng[par2, 0],
                                    self._modellink._par_rng[par2, 1],
                                    self.__res)

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
        f, axarr = plt.subplots(2, **self.__fig_kwargs)
        f.suptitle(r"%s. Projection (%s)" % (self.__emul_i, hcube_name),
                   fontsize='xx-large')

        # Plot minimum implausibility
        vmax = self._impl_cut[self.__emul_i][self._cut_idx[self.__emul_i]]
        fig1 = axarr[0].hexbin(x, y, z_min, gridsize, vmin=0, vmax=vmax,
                               **self.__impl_kwargs)
        if self._modellink._par_est[par1] is not None:
            draw_textline(r"", x=self._modellink._par_est[par1], ax=axarr[0],
                          line_kwargs=self.__line_kwargs)
        if self._modellink._par_est[par2] is not None:
            draw_textline(r"", y=self._modellink._par_est[par2], ax=axarr[0],
                          line_kwargs=self.__line_kwargs)
        axarr[0].axis([self._modellink._par_rng[par1, 0],
                       self._modellink._par_rng[par1, 1],
                       self._modellink._par_rng[par2, 0],
                       self._modellink._par_rng[par2, 1]])
        plt.colorbar(fig1, ax=axarr[0]).set_label("Minimum Implausibility",
                                                  fontsize='large')

        # Plot line-of-sight depth
        fig2 = axarr[1].hexbin(x, y, z_los, gridsize, vmin=0,
                               vmax=min(1, np.max(z_los)), **self.__los_kwargs)
        if self._modellink._par_est[par1] is not None:
            draw_textline(r"", x=self._modellink._par_est[par1], ax=axarr[1],
                          line_kwargs=self.__line_kwargs)
        if self._modellink._par_est[par2] is not None:
            draw_textline(r"", y=self._modellink._par_est[par2], ax=axarr[1],
                          line_kwargs=self.__line_kwargs)
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
        plt.savefig(self.__fig_path)

        # If show is set to True, show the figure
        f.show() if self.__show else plt.close(f)

        # Log that this hypercube has been drawn
        logger.info("Finished drawing projection figure '%s'." % (hcube_name))

    # This function determines the projection hypercubes to be analyzed
    @docstring_substitute(proj_par=proj_par_doc_s)
    def __get_req_hcubes(self, proj_par):
        """
        Determines which projection hypercubes have been requested by the user.
        Also checks if these projection hypercubes have been calculated before,
        and depending on the value of `force`, either skips them or recreates
        them.

        Parameters
        ----------
        %(proj_par)s

        Generates
        ---------
        hcubes : list of lists
            List containing the parameter indices of the requested projection
            hypercubes.
        create_hcubes : list of lists
            List containing the parameter indices of the requested projection
            hypercubes that need to be created first.

        """

        # Start logger
        logger = getCLogger('PROJECTION')

        # Controller determining which proj_hcubes are going to be made
        if self._is_controller:
            # Check the proj_par that were provided
            # If none were provided, make figs for all active model parameters
            if proj_par is None:
                proj_par = self._emulator._active_par[self.__emul_i]

            # Else, a sequence of str/int must be provided
            else:
                proj_par = self._get_model_par_seq(proj_par, 'proj_par')

                # Check which values in proj_par are also in active_par
                proj_par = np.array(
                    [i for i in self._emulator._active_par[self.__emul_i] if
                     i in proj_par])

                # Make sure that there are still enough values left
                if(self._modellink._n_par == 2 and len(proj_par) >= 1):
                    pass
                elif(self._modellink._n_par > 2 and len(proj_par) >= 2):
                    pass
                else:
                    err_msg = ("Not enough active model parameters have been "
                               "provided to make a projection figure!")
                    raise_error(RequestError, err_msg, logger)

            # Obtain list of hypercube names
            if(self._modellink._n_par == 2):
                hcube_idx = list(combinations(range(len(proj_par)), 1))
            else:
                hcube_idx = list(combinations(range(len(proj_par)), 2))
            hcubes = proj_par[np.array(hcube_idx)].tolist()

            # Create empty list holding hcube_par that needs to be created
            create_hcubes = []

            # Open hdf5-file
            logger.info("Checking if projection data already exists.")
            with PRISM_File('r+', None) as file:
                # Check if data is already there and act accordingly
                for hcube in hcubes:
                    # Obtain name of this hypercube
                    hcube_name = self.__get_hcube_name(hcube)

                    # Check if projection data already exists
                    try:
                        file['%s/proj_hcube/%s' % (self.__emul_i, hcube_name)]

                    # If it does not exist, add it to the creation list
                    except KeyError:
                        logger.info("Projection data '%s' not found. Will be "
                                    "created." % (hcube_name))
                        create_hcubes.append(hcube)

                    # If it does exist, check value of force
                    else:
                        # If force is used, remove data and figure
                        if self.__force:
                            # Remove data
                            del file['%s/proj_hcube/%s'
                                     % (self.__emul_i, hcube_name)]
                            logger.info("Projection data '%s' already exists. "
                                        "Deleting." % (hcube_name))

                            # Try to remove figure as well
                            if path.exists('%s(%s).png'
                                           % (self.__fig_prefix, hcube_name)):
                                logger.info("Projection figure '%s' already "
                                            "exists. Deleting." % (hcube_name))
                                os.remove('%s(%s).png'
                                          % (self.__fig_prefix, hcube_name))

                            # Add this hypercube to creation list
                            create_hcubes.append(hcube)

                        # If force is not used, skip creation
                        else:
                            logger.info("Projection data '%s' already exists. "
                                        "Skipping data creation."
                                        % (hcube_name))

        # Workers getting dummy hypercubes
        else:
            hcubes = []
            create_hcubes = []

        # Broadcast hypercubes to workers
        self.__hcubes = self._comm.bcast(hcubes, 0)
        self.__create_hcubes = self._comm.bcast(create_hcubes, 0)

    # This function returns the name of a proj_hcube when given a hcube
    @docstring_substitute(hcube=hcube_doc)
    def __get_hcube_name(self, hcube):
        """
        Determines the name of a provided projection hypercube `hcube` and
        returns it.

        Parameters
        ----------
        %(hcube)s

        Returns
        -------
        hcube_name : str
            The name of this projection hypercube.

        """

        if(self._modellink._n_par == 2):
            return('%s' % (self._modellink._par_name[hcube[0]]))
        else:
            return('%s-%s' % (self._modellink._par_name[hcube[0]],
                              self._modellink._par_name[hcube[1]]))

    # This function returns default projection parameters
    @docstring_append(def_par_doc.format('projection'))
    def __get_default_parameters(self):
        # Create parameter dict with default parameters
        par_dict = {'proj_res': '15',
                    'proj_depth': '150'}

        # Return it
        return(par_dict)

    # This function returns default projection input arguments
    def __get_default_input_arguments(self):
        """
        Generates a dict containing default values for all input arguments.

        Returns
        -------
        kwargs_dict : dict
            Dict containing all default input argument values.

        """

        # Create input argument dicts with default figure parameters
        fig_kwargs = {'figsize': (6.4, 4.8),
                      'dpi': 100}
        impl_kwargs = {'cmap': 'rainforest_r'}
        los_kwargs = {'cmap': 'blaze'}
        line_kwargs = {'linestyle': '--',
                       'color': 'grey'}

        # Create input argument dict with default projection parameters
        kwargs_dict = {'emul_i': None,
                       'proj_par': None,
                       'figure': True,
                       'show': False,
                       'force': False,
                       'fig_kwargs': fig_kwargs,
                       'impl_kwargs': impl_kwargs,
                       'los_kwargs': los_kwargs,
                       'line_kwargs': line_kwargs}

        # Return it
        return(kwargs_dict)

    # This function reads in the impl_cut list from the PRISM parameters file
    @docstring_append(read_par_doc.format("projection", "Pipeline"))
    def __read_parameters(self):
        # Log that the PRISM parameter file is being read
        logger = getCLogger('INIT')
        logger.info("Reading projection parameters.")

        # Obtaining default projection parameter dict
        par_dict = self.__get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._prism_file is not None:
            pipe_par = np.genfromtxt(self._prism_file, dtype=(str),
                                     delimiter=':', autostrip=True)

            # Make sure that pipe_par is 2D
            pipe_par = np.array(pipe_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(pipe_par)

        # More logging
        logger.info("Checking compatibility of provided projection "
                    "parameters.")

        # Save impl_data to proj_hcube
        self.__save_data({
            'impl_cut': [self._impl_cut[self.__emul_i],
                         self._cut_idx[self.__emul_i]]})

        # Number of samples used for implausibility evaluations
        proj_res = int(par_dict['proj_res'])
        proj_depth = int(par_dict['proj_depth'])
        self.__save_data({
            'proj_grid': [check_val(proj_res, 'proj_res', 'pos'),
                          check_val(proj_depth, 'proj_depth', 'pos')]})

        # Finish logging
        logger.info("Finished reading projection parameters.")

    # This function generates a projection hypercube to be used for emulator
    # evaluations
    @docstring_substitute(hcube=hcube_doc)
    def __get_proj_hcube(self, hcube):
        """
        Generates a projection hypercube `hcube` containing emulator evaluation
        samples The output of this function depends on the number of model
        parameters.

        Parameters
        ----------
        %(hcube)s

        Returns
        -------
        proj_hcube : 3D :obj:`~numpy.ndarray` object
            3D projection hypercube of emulator evaluation samples.

        """

        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Log that projection hypercube is being created
        logger = getCLogger('PROJ_HCUBE')
        logger.info("Creating projection hypercube '%s'." % (hcube_name))

        # If n_par is 2, make 2D projection hypercube on controller
        if(self._is_controller and self._modellink._n_par == 2):
            # Identify projected parameter
            par = hcube[0]

            # Create empty projection hypercube array
            proj_hcube = np.zeros([self.__res, self.__depth, 2])

            # Create list that contains all the other parameters
            par_hid = 1 if par == 0 else 0

            # Generate list with values for projected parameter
            proj_sam_set = np.linspace(self._modellink._par_rng[par, 0],
                                       self._modellink._par_rng[par, 1],
                                       self.__res)

            # Generate latin hypercube of the remaining parameters
            hidden_sam_set = lhd(self.__depth, 1,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._criterion)[:, 0]

            # Fill every cell in the projection hypercube accordingly
            for i in range(self.__res):
                proj_hcube[i, :, par] = proj_sam_set[i]
                proj_hcube[i, :, par_hid] = hidden_sam_set

        # If n_par is more than 2, make 3D projection hypercube on controller
        elif self._is_controller:
            # Identify projected parameters
            par1 = hcube[0]
            par2 = hcube[1]

            # Create empty projection hypercube array
            proj_hcube = np.zeros([pow(self.__res, 2), self.__depth,
                                   self._modellink._n_par])

            # Generate list that contains all the other parameters
            par_hid = list(chain(range(0, par1), range(par1+1, par2),
                                 range(par2+1, self._modellink._n_par)))

            # Generate list with values for projected parameters
            proj_sam_set1 = np.linspace(self._modellink._par_rng[par1, 0],
                                        self._modellink._par_rng[par1, 1],
                                        self.__res)
            proj_sam_set2 = np.linspace(self._modellink._par_rng[par2, 0],
                                        self._modellink._par_rng[par2, 1],
                                        self.__res)

            # Generate Latin Hypercube of the remaining parameters
            hidden_sam_set = lhd(self.__depth, self._modellink._n_par-2,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._criterion)

            # Fill every cell in the projection hypercube accordingly
            for i in range(self.__res):
                for j in range(self.__res):
                    proj_hcube[i*self.__res+j, :, par1] = proj_sam_set1[i]
                    proj_hcube[i*self.__res+j, :, par2] = proj_sam_set2[j]
                    proj_hcube[i*self.__res+j, :, par_hid] = hidden_sam_set.T

        # Workers get dummy proj_hcube
        else:
            proj_hcube = []

        # Broadcast proj_hcube to workers
        proj_hcube = self._comm.bcast(proj_hcube, 0)

        # Log that projection hypercube has been created
        logger.info("Finished creating projection hypercube '%s'."
                    % (hcube_name))

        # Return proj_hcube
        return(proj_hcube)

    # This function analyzes a projection hypercube
    @docstring_substitute(hcube=hcube_doc)
    def __analyze_proj_hcube(self, hcube):
        """
        Analyzes an emulator projection hypercube `hcube`.

        Parameters
        ----------
        %(hcube)s

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

        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Log that a projection hypercube is being analyzed
        logger = getCLogger('ANALYSIS')
        logger.info("Analyzing projection hypercube '%s'." % (hcube_name))

        # Obtain the corresponding hypercube
        proj_hcube = self.__get_proj_hcube(hcube)

        # CALCULATE AND ANALYZE IMPLAUSIBILITY
        # Create empty lists for this hypercube
        impl_min_hcube = []
        impl_los_hcube = []

        # Define the various code snippets
        pre_code = compile("impl_cut = np.zeros([n_sam])", '<string>', 'exec')
        eval_code = compile("", '<string>', 'exec')
        anal_code = compile("impl_cut[sam_idx[j]] = impl_cut_val", '<string>',
                            'exec')
        post_code = compile("", '<string>', 'exec')
        exit_code = compile("self.results = (impl_check, impl_cut)",
                            '<string>', 'exec')

        # Combine code snippets into a tuple
        exec_code = (pre_code, eval_code, anal_code, post_code, exit_code)

        # For now, manually flatten the first two dimensions of proj_hcube
        gridsize = proj_hcube.shape[0]
        depth = proj_hcube.shape[1]
        proj_hcube = proj_hcube.reshape(gridsize*depth, self._modellink._n_par)

        # Save current time
        start_time = time()

        # Analyze all samples in proj_hcube
        results = self._analyze_sam_set(self.__emul_i, proj_hcube, *exec_code)

        # Controller only
        if self._is_controller:
            # Retrieve results
            impl_check, impl_cut = results

            # Unflatten the received results
            impl_check = impl_check.reshape(gridsize, self.__depth)
            impl_cut = impl_cut.reshape(gridsize, self.__depth)

            # Loop over all grid point results and save lowest impl and los
            for check_grid, cut_grid in zip(impl_check, impl_cut):
                # Calculate lowest impl in this grid point
                impl_min_hcube.append(min(cut_grid))

                # Calculate impl line-of-sight in this grid point
                impl_los_hcube.append(sum(check_grid)/self.__depth)

            # Log that analysis has been finished
            time_diff = time()-start_time
            total = np.size(impl_check)
            logger.info("Finished projection hypercube analysis in %.2f "
                        "seconds, averaging %.2f emulator evaluations per "
                        "second." % (time_diff, total/(time_diff)))

            # Log that projection data has been created
            logger.info("Finished calculating projection data '%s'."
                        % (hcube_name))

            # Save projection data to hdf5
            self.__save_data({
                'nD_proj_hcube': [hcube_name, impl_min_hcube, impl_los_hcube]})

        # Return the results for this proj_hcube
        return(impl_min_hcube, impl_los_hcube)

    # This function processes the input arguments of project
    def __process_input_arguments(self, *args, **kwargs):
        """
        Processes the input arguments given to the :meth:`~project` method.

        Parameters
        ----------
        args : list
            List of positional arguments that were provided to
            :meth:`~project`.
        kwargs : dict
            Dict of keyword arguments that were provided to :meth:`~project`.

        Generates
        ---------
        All default and provided `args` and `kwargs` are saved to their
        respective properties.

        """

        # Make a logger
        logger = getCLogger('PROJ_INIT')
        logger.info("Processing input arguments.")

        # Make dictionary with default argument values
        kwargs_dict = self.__get_default_input_arguments()

        # Make list with forbidden figure and plot kwargs
        pop_fig_kwargs = ['nrows', 'ncols', 'sharex', 'sharey']
        pop_plt_kwargs = ['x', 'y', 'C', 'gridsize', 'vmin', 'vmax']

        # Check if not more than two args have been provided
        if(len(args) > 2):
            err_msg = ("The project()-method takes a maximum of 2 positional "
                       "arguments, but %s have been provided!" % (len(args)))
            raise_error(InputError, err_msg, logger)

        # Update emul_i and proj_par by given args
        try:
            kwargs_dict['emul_i'] = args[0]
            kwargs_dict['proj_par'] = args[1]
        except IndexError:
            pass

        # Update kwargs_dict with given kwargs
        for key, value in kwargs.items():
            if key in ('fig_kwargs', 'impl_kwargs', 'los_kwargs',
                       'line_kwargs'):
                if not isinstance(value, dict):
                    err_msg = ("Input argument '%s' is not of type 'dict'!"
                               % (key))
                    raise_error(TypeError, err_msg, logger)
                else:
                    kwargs_dict[key].update(value)
            else:
                kwargs_dict[key] = value
        kwargs = kwargs_dict

        # Get emul_i
        self.__emul_i = self._emulator._get_emul_i(kwargs['emul_i'], True)

        # Controller checking all other kwargs
        if self._is_controller:
            # Check if figure, show and force-parameters are bools
            self.__figure = check_val(kwargs['figure'], 'figure', 'bool')
            self.__show = check_val(kwargs['show'], 'show', 'bool')
            self.__force = check_val(kwargs['force'], 'force', 'bool')

            # Pop all specific kwargs dicts from kwargs
            fig_kwargs = kwargs['fig_kwargs']
            impl_kwargs = kwargs['impl_kwargs']
            los_kwargs = kwargs['los_kwargs']
            line_kwargs = kwargs['line_kwargs']

            # FIG_KWARGS
            # Check if any forbidden kwargs are given and remove them
            fig_keys = list(fig_kwargs.keys())
            for key in fig_keys:
                if key in pop_fig_kwargs:
                    fig_kwargs.pop(key)

            # IMPL_KWARGS
            # Check if provided cmap is an actual cmap
            try:
                impl_kwargs['cmap'] = cm.get_cmap(impl_kwargs['cmap'])
            except Exception as error:
                err_msg = ("Input argument 'impl_kwargs/cmap' is invalid (%s)!"
                           % (error))
                raise_error(InputError, err_msg, logger)

            # Check if any forbidden kwargs are given and remove them
            impl_keys = list(impl_kwargs.keys())
            for key in impl_keys:
                if key in pop_plt_kwargs:
                    impl_kwargs.pop(key)
            if(self._modellink._n_par == 2):
                impl_kwargs.pop('cmap')

            # LOS_KWARGS
            # Check if provided cmap is an actual cmap
            try:
                los_kwargs['cmap'] = cm.get_cmap(los_kwargs['cmap'])
            except Exception as error:
                err_msg = ("Input argument 'los_kwargs/cmap' is invalid (%s)!"
                           % (error))
                raise_error(InputError, err_msg, logger)

            # Check if any forbidden kwargs are given and remove them
            los_keys = list(los_kwargs.keys())
            for key in los_keys:
                if key in pop_plt_kwargs:
                    los_kwargs.pop(key)
            if(self._modellink._n_par == 2):
                los_kwargs.pop('cmap')

            # Save kwargs dicts to memory
            self.__fig_kwargs = fig_kwargs
            self.__impl_kwargs = impl_kwargs
            self.__los_kwargs = los_kwargs
            self.__line_kwargs = line_kwargs

        # MPI Barrier
        self._comm.Barrier()

        # Log again
        logger.info("Finished processing input arguments.")

        # Return proj_par
        return(kwargs['proj_par'])

    # This function prepares for projections to be made
    def __prepare_projections(self, *args, **kwargs):
        """
        Prepares the pipeline for the creation of the requested projections.

        Parameters
        ----------
        args : list
            List of positional arguments that were provided to
            :meth:`~project`.
        kwargs : dict
            Dict of keyword arguments that were provided to :meth:`~project`.

        """

        # Create logger
        logger = getCLogger('PROJ_INIT')

        # Combine received args and kwargs with default ones
        proj_par = self.__process_input_arguments(*args, **kwargs)

        # Controller doing some preparations
        if self._is_controller:
            # Check if it makes sense to create a projection
            if(self.__emul_i == self._emulator._emul_i):
                if not self._n_eval_sam[self.__emul_i]:
                    msg = ("Requested emulator iteration %s has not been "
                           "analyzed yet. Creating projections may not be "
                           "useful." % (self.__emul_i))
                    logger.info(msg)
                    print(msg)
                elif self._prc:
                    pass
                else:
                    err_msg = ("Requested emulator iteration %s has no "
                               "plausible regions. Creating projections has no"
                               " use." % (self.__emul_i))
                    raise_error(RequestError, err_msg, logger)

            # Check if projection has been created before
            with PRISM_File('r+', None) as file:
                try:
                    file.create_group('%s/proj_hcube' % (self.__emul_i))
                except ValueError:
                    pass

            # Obtain figure name prefix
            self.__fig_prefix = self._fig_prefix.format(self._working_dir,
                                                        self.__emul_i)

            # Read in projection parameters
            self.__read_parameters()

        # Obtain requested projection hypercubes
        self.__get_req_hcubes(proj_par)

    # This function saves projection data to hdf5
    @docstring_substitute(save_data=save_data_doc_pr)
    def __save_data(self, data_dict):
        """
        Saves a given data dict ``{keyword: data}`` at the emulator iteration
        this class was initialized for, to the HDF5-file.

        %(save_data)s

        """

        # Do some logging
        logger = getLogger('SAVE_DATA')

        # Open hdf5-file
        with PRISM_File('r+', None) as file:
            # Obtain the dataset this data needs to be saved to
            data_set = file['%s/proj_hcube' % (self.__emul_i)]

            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %s data at iteration %s to HDF5."
                            % (keyword, self.__emul_i))

                # Check what data keyword has been provided
                # IMPL_CUT
                if(keyword == 'impl_cut'):
                    # Save the used implausibility parameters to file
                    data_set.attrs['impl_cut'] = data[0]
                    data_set.attrs['cut_idx'] = data[1]

                # PROJ_GRID
                elif(keyword == 'proj_grid'):
                    # Save projection resolution and depth to file and memory
                    self.__res = data[0]
                    self.__depth = data[1]
                    data_set.attrs['proj_res'] = data[0]
                    data_set.attrs['proj_depth'] = data[1]

                # ND_PROJ_HCUBE
                elif(keyword == 'nD_proj_hcube'):
                    # Save the projection data to file
                    data_set.create_dataset('%s/impl_min' % (data[0]),
                                            data=data[1])
                    data_set.create_dataset('%s/impl_los' % (data[0]),
                                            data=data[2])

                # INVALID KEYWORD
                else:
                    err_msg = "Invalid keyword argument provided!"
                    raise_error(ValueError, err_msg, logger)
