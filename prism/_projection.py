# -*- coding: utf-8 -*-

"""
Projection
==========
Provides the definition of *PRISM*'s :class:`~Projection` class, a
:class:`~prism.Pipeline` base class that allows for projection figures
detailing a model's behavior to be created.

"""


# %% IMPORTS
# Built-in imports
from itertools import chain, combinations
import os
from os import path
from time import time

# Package imports
from e13tools import InputError
from e13tools.pyplot import draw_textline
from e13tools.sampling import lhd
from e13tools.utils import (convert_str_seq, docstring_append,
                            docstring_substitute, raise_error, raise_warning)
from matplotlib import cm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from sortedcontainers import SortedDict as sdict
from tqdm import tqdm

# PRISM imports
from prism._docstrings import (def_par_doc, draw_proj_fig_doc, get_emul_i_doc,
                               hcube_doc, proj_data_doc, proj_par_doc_d,
                               proj_par_doc_s, save_data_doc_pr, set_par_doc,
                               user_emul_i_doc)
from prism._internal import (RequestError, RequestWarning, check_vals,
                             getCLogger, np_array)

# All declaration
__all__ = ['Projection']


# %% PROJECTION CLASS DEFINITION
class Projection(object):
    """
    Defines the :class:`~Projection` class of the *PRISM* package.

    Description
    -----------
    The :class:`~Projection` class holds all specific methods that the
    :class:`~prism.Pipeline` class needs in order to create
    projections of the model.

    This is a base class for the :class:`~prism.Pipeline` class and
    cannot be used on its own.

    """

    # Create __init__ method to warn if this class is ever initialized
    def __init__(self, *args, **kwargs):
        logger = getCLogger('PROJECTION')
        err_msg = ("The Projection class is a base class for the Pipeline "
                   "class and cannot be used on its own!")
        raise_error(err_msg, RequestError, logger)

    # Function that creates all projection figures
    # TODO: Allow for projection figures to be zoomed? (Cut off all black)
    @docstring_substitute(emul_i=user_emul_i_doc, proj_par=proj_par_doc_d)
    def project(self, emul_i=None, proj_par=None, **kwargs):
        """
        Analyzes the emulator iteration `emul_i` and constructs a series of
        projection figures detailing the behavior of the model parameters
        corresponding to the given `proj_par`.
        The input and output depend on the number of model parameters
        :attr:`~prism.modellink.ModelLink.n_par`.

        All optional keyword arguments (except `force`) control various aspects
        of drawing the projection figures and do not affect the projection data
        that is saved to HDF5. This is instead influenced by the
        :attr:`~proj_res` and :attr:`~proj_depth` properties.

        Parameters
        ----------
        %(emul_i)s
        %(proj_par)s

        Keyword arguments
        -----------------
        proj_type : {'2D', '3D', 'both'}. Default: '2D' (2D), 'both' (nD)
            String indicating which projection type to create for all supplied
            active parameters.
            If :attr:`~prism.modellink.ModelLink.n_par` == 2, this is always
            '2D' (and cannot be modified).
        figure : bool. Default: True
            Whether or not to create the projection figures. If *True*, the
            figures are calculated, drawn and saved. If *False*, the figures
            are calculated and their data is returned in a dict.
        align : {'row'/'horizontal', 'col'/'column'/'vertical'}. Default: 'col'
            If `figure` is *True*, string indicating how to position the two
            subplots.
            If 'row'/'horizontal', the subplots are positioned on a single row.
            If 'col'/'column'/'vertical', the subplots are positioned on a
            single column.
        show_cuts : bool. Default: False
            If `figure` is *True* and `proj_type` is not '3D', whether to show
            all implausibility cut-offs in the 2D projections (*True*) or only
            the first cut-off (*False*).
        smooth : bool. Default: False
            Controls what to do if a grid point contains no plausible samples,
            but does contain a minimum implausibility value below the first
            non-wildcard cut-off.
            If *False*, these values are kept, which can show up as
            artifact-like features in the projection figure.
            If *True*, these values are set to the first cut-off, removing them
            from the projection figure. Doing this may also remove interesting
            features. This does not affect the projection data saved to HDF5.
            Smoothed figures have an '_s' string appended to their filenames.
        force : bool. Default: False
            Controls what to do if a projection hypercube has been calculated
            at the emulator iteration `emul_i` before.
            If *False*, it will use the previously acquired projection data to
            create the projection figure.
            If *True*, it will recalculate all the data required to create the
            projection figure. Note that this will also delete all associated
            projection figures.
        fig_kwargs : dict. Default: {'figsize': (6.4, 4.8), 'dpi': 100}
            Dict of keyword arguments to be used when creating the subplots
            figure. It takes all arguments that can be provided to the
            :func:`~matplotlib.pyplot.figure` function.
        impl_kwargs_2D : dict. Default: {}
            Dict of keyword arguments to be used for making the minimum
            implausibility (top/left) plot in the 2D projection figures. It
            takes all arguments that can be provided to the
            :func:`~matplotlib.pyplot.plot` function.
        impl_kwargs_3D : dict. Default: {'cmap': 'rainforest_r'}
            Dict of keyword arguments to be used for making the minimum
            implausibility (top/left) plot in the 3D projection figures. It
            takes all arguments that can be provided to the
            :func:`~matplotlib.pyplot.hexbin` function.
        los_kwargs_2D : dict. Default: {}
            Dict of keyword arguments to be used for making the line-of-sight
            (bottom/right) plot in the 2D projection figures. It takes all
            arguments that can be provided to the
            :func:`~matplotlib.pyplot.plot` function.
        los_kwargs_3D : dict. Default: {'cmap': 'blaze'}
            Dict of keyword arguments to be used for making the line-of-sight
            (bottom/right) plot in the 3D projection figures. It takes all
            arguments that can be provided to the
            :func:`~matplotlib.pyplot.hexbin` function.
        line_kwargs_est : dict. Default: {'linestyle': '--', 'color': 'grey'}
            Dict of keyword arguments to be used for drawing the parameter
            estimate lines in both plots. It takes all arguments that can be
            provided to the :func:`~matplotlib.pyplot.draw` function.
        line_kwargs_cut : dict. Default: {'color': 'r'}
            Dict of keyword arguments to be used for drawing the implausibility
            cut-off line(s) in the top/left plot in the 2D projection figures.
            It takes all arguments that can be provided to the
            :func:`~matplotlib.pyplot.draw` function.

        Returns (if `figure` is *False*)
        --------------------------------
        fig_data : dict of dicts
            Dict containing the data for every requested projection figure,
            split up into the 'impl_min' and 'impl_los' dicts. For 2D
            projections, every dict contains a list with the x and y values.
            For 3D projections, it contains the x, y and z values.
            Note that due to the figures being interpolations, the y/z values
            can be below zero or the line-of-sight values being above unity.

        Generates (if `figure` is *True*)
        ---------------------------------
        A series of projection figures detailing the behavior of the model.
        The lay-out and output of the projection figures depend on the type of
        figure:

            2D projection figure: The output will feature a figure with two
            subplots for every active model parameter (``n_par``). Every figure
            gives details about the behavior of the corresponding model
            parameter, by showing the minimum implausibility value (top/left)
            and the line-of-sight depth (bottom/right) obtained at the
            specified parameter value, independent of the values of the other
            parameters.

            3D projection figure (only if
            :attr:`~prism.modellink.ModelLink.n_par` > 2): The output
            will feature a figure with two subplots for every combination of
            two active model parameters that can be made
            (``n_par*(n_par-1)/2``). Every figure gives details about the
            behavior of the corresponding model parameters, as well as their
            dependency on each other. This is done by showing the minimum
            implausibility (top/left) and the line-of-sight depth
            (bottom/right) obtained at the specified parameter values,
            independent of the values of the remaining model parameters.

        Notes
        -----
        If given emulator iteration `emul_i` has been analyzed before, the
        implausibility parameters of the last analysis are used. If not, then
        the values are used that were read in when the emulator was loaded or
        that have been set by the user.

        """

        # Log the start of the creation of the projection
        logger = getCLogger('PROJECTION')
        logger.info("Starting the projection process.")

        # Save current time
        start_time1 = time()

        # Prepare for making projections
        self.__prepare_projections(emul_i, proj_par, **kwargs)

        # Save current time again
        start_time2 = time()

        # Loop over all requested projection hypercubes
        if self._is_controller and self._do_logging:
            hcubes_bar = tqdm(self.__hcubes, desc="Creating projections",
                              unit='hcube', dynamic_ncols=True)
        else:
            hcubes_bar = self.__hcubes
        for hcube in hcubes_bar:
            # Initialize impl_min and impl_los
            impl_min = None
            impl_los = None

            # Obtain name of this hypercube
            hcube_name = self.__get_hcube_name(hcube)

            # ANALYZE PROJECTION HYPERCUBE
            # Create and analyze projection hypercube if required
            if hcube in self.__create_hcubes:
                # Log that projection data is being created
                logger.info("Calculating projection data %r." % (hcube_name))

                # Analyze this proj_hcube
                impl_min, impl_los = self.__analyze_proj_hcube(hcube)

            # PLOTTING (CONTROLLER ONLY)
            # Create projection figure
            if self._is_controller:
                # Skip making figure if it already exists and figure is True
                if(path.exists(self.__get_fig_path(hcube)[self.__smooth]) and
                   self.__figure):
                    logger.info("Projection figure %r already exists. "
                                "Skipping figure creation." % (hcube_name))
                    self._comm.Barrier()
                    continue

                # If projection data is not already loaded, load it
                if impl_min is None and impl_los is None:
                    impl_min, impl_los, proj_res, _ =\
                        self.__get_proj_data(hcube)
                # Otherwise, the used resolution is the current resolution
                else:
                    proj_res = self._res

                # Draw projection figure
                if(len(hcube) == 1):
                    self.__draw_2D_proj_fig(hcube, impl_min, impl_los,
                                            proj_res)
                else:
                    self.__draw_3D_proj_fig(hcube, impl_min, impl_los,
                                            proj_res)

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
        self.details(self.__emul_i)

        # If figure is False, return figure data on controller
        if self._is_controller and not self.__figure:
            return(self.__fig_data)

    # %% CLASS PROPERTIES
    @property
    def proj_res(self):
        """
        int: Number of emulator evaluations that will be used to generate the
        grid for the projection figures.

        """

        return(getattr(self, '_res', None))

    @proj_res.setter
    def proj_res(self, proj_res):
        self._res = check_vals(proj_res, 'proj_res', 'int', 'pos')

    @property
    def proj_depth(self):
        """
        int: Number of emulator evaluations that will be used to generate the
        samples in every grid point for the projection figures.
        Note that when making 2D projections of nD models, the used depth will
        be this number multiplied by :attr:`~proj_res`.

        """

        return(getattr(self, '_depth', None))

    @proj_depth.setter
    def proj_depth(self, proj_depth):
        self._depth = check_vals(proj_depth, 'proj_depth', 'int', 'pos')

    # %% HIDDEN CLASS METHODS
    # This function draws the 2D projection figure
    @docstring_append(draw_proj_fig_doc.format("2D", "1"))
    def __draw_2D_proj_fig(self, hcube, impl_min, impl_los, proj_res):
        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Make abbreviation for implausibility cut-off values
        impl_cut = self._impl_cut[self.__emul_i][0]
        impl_cuts = self._impl_cut[self.__emul_i]

        # Start logger
        logger = getCLogger('PROJECTION')
        logger.info("Calculating projection figure %r." % (hcube_name))

        # Get the parameter this hypercube is about
        par = hcube[0]

        # Make abbreviation for parameter name
        par_name = self._modellink._par_name[par]

        # Create the normalized parameter value array used to create the hcube
        # Normalization is necessary for avoiding interpolation errors
        x_proj = np.linspace(0, 1, proj_res)

        # Get the interpolated functions describing the minimum
        # implausibility and line-of-sight depth obtained in every
        # point
        # TODO: Allow user to set smooth parameter for Rbf function
        # This probably means that smoothed figures have to be renamed
        f_min = Rbf(x_proj, impl_min)
        f_los = Rbf(x_proj, impl_los)

        # Set the size of the grid
        gridsize = max(10*self.__fig_kwargs['dpi'], proj_res)

        # Create normalized parameter value array for interpolation functions
        x = np.linspace(0, 1, gridsize)

        # Calculate y_min and y_los
        y_min = f_min(x)
        y_los = f_los(x)

        # Check for every interpolated grid point if the corresponding y_los is
        # non-zero due to interpolation artifacts
        for i, xi in enumerate(x):
            # Obtain the 1D indices of the two corner projected grid points
            corner_1D = [
                max(x_proj.searchsorted(xi)-1, 0),
                min(x_proj.searchsorted(xi, side='right'), proj_res-1)]

            # If both corner points are zero, a point in between should be zero
            if not (impl_los[corner_1D] > 0).any():
                y_los[i] = 0

        # Check if y_min is requested to be smoothed
        if self.__smooth:
            # Loop over all grid points
            y_min[y_los <= 0] = impl_cut

        # Create plotted parameter value array
        x = np.linspace(*self._modellink._par_rng[par], gridsize)

        # Create figure object if the figure is requested
        if self.__figure:
            if(self.__align == 'row'):
                f = plt.figure(constrained_layout=True, **self.__fig_kwargs)
                w_pad, h_pad, wspace, hspace = f.get_constrained_layout_pads()

                # Create GridSpec objects including a dummy Axes object
                gsarr = gs.GridSpec(2, 2, figure=f, height_ratios=[1, 0.00001])
                ax0 = f.add_subplot(gsarr[0, 0])
                ax1 = f.add_subplot(gsarr[0, 1])
                label_ax = f.add_subplot(gsarr[1, :])

                # Set padding to the bare minimum
                f.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad/2,
                                              wspace=wspace, hspace=0)
            else:
                f, (ax0, ax1) = plt.subplots(2, constrained_layout=True,
                                             **self.__fig_kwargs)
                w_pad, h_pad, wspace, hspace = f.get_constrained_layout_pads()

                # Set padding to the bare minimum
                f.set_constrained_layout_pads(w_pad=w_pad/2, h_pad=h_pad,
                                              wspace=0, hspace=hspace)

            # Set super title
            f.suptitle(r"%s. Projection (%s)" % (self.__emul_i, hcube_name),
                       fontsize='xx-large')

            # MINIMUM IMPLAUSIBILITY PLOT
            # Plot minimum implausibility
            ax0.plot(x, y_min, **self.__impl_kwargs_2D)
            ax0_rng = [*self._modellink._par_rng[par], 0, 1.5*impl_cut]
            ax0.axis(ax0_rng)

            # Draw parameter estimate line
            if self._modellink._par_est[par] is not None:
                draw_textline(r"", x=self._modellink._par_est[par], ax=ax0,
                              line_kwargs=self.__line_kwargs_est)

            # Draw implausibility cut-off line(s)
            if self.__show_cuts:
                # If all lines are requested, draw them
                for cut in impl_cuts:
                    draw_textline(r"", y=cut, ax=ax0,
                                  line_kwargs=self.__line_kwargs_cut)
            else:
                # Else, draw the first cut-off line
                draw_textline(r"", y=impl_cut, ax=ax0,
                              line_kwargs=self.__line_kwargs_cut)

            # Set axes and label
            ax0.axis(ax0_rng)
            ax0.set_ylabel("Min. Implausibility", fontsize='large')

            # LINE-OF-SIGHT DEPTH PLOT
            # Plot line-of-sight depth
            ax1.plot(x, y_los, **self.__los_kwargs_2D)
            ax1_rng = [*self._modellink._par_rng[par],
                       0, min(1, np.max(y_los))]
            ax1.axis(ax1_rng)

            # Draw parameter estimate line
            if self._modellink._par_est[par] is not None:
                draw_textline(r"", x=self._modellink._par_est[par], ax=ax1,
                              line_kwargs=self.__line_kwargs_est)

            # Set axes and label
            ax1.axis(ax1_rng)
            ax1.set_ylabel("Line-of-Sight Depth", fontsize='large')

            # Make super axis label using dummy Axes object as an empty plot
            if(self.__align == 'row'):
                label_ax.set_frame_on(False)
                label_ax.get_xaxis().set_ticks([])
                label_ax.get_yaxis().set_ticks([])
                label_ax.autoscale(tight=True)
                label_ax.set_xlabel(par_name, fontsize='x-large', labelpad=0)
            else:
                ax1.set_xlabel(par_name, fontsize='x-large')

            # Save the figure
            plt.savefig(self.__get_fig_path(hcube)[self.__smooth])

            # Close the figure
            plt.close(f)

            # Log that this hypercube has been drawn
            logger.info("Finished calculating and drawing projection figure "
                        "%r." % (hcube_name))

        # If the figure data has been requested instead
        else:
            self.__fig_data[hcube_name] = {
                'impl_min': [x, y_min],
                'impl_los': [x, y_los]}
            logger.info("Finished calculating projection figure %r."
                        % (hcube_name))

    # This function draws the 3D projection figure
    @docstring_append(draw_proj_fig_doc.format("3D", "2"))
    def __draw_3D_proj_fig(self, hcube, impl_min, impl_los, proj_res):
        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Make abbreviation for first implausibility cut-off value
        impl_cut = self._impl_cut[self.__emul_i][0]

        # Start logger
        logger = getCLogger('PROJECTION')
        logger.info("Calculating projection figure %r." % (hcube_name))

        # Get the parameter on x-axis and y-axis this hcube is about
        par1 = hcube[0]
        par2 = hcube[1]

        # Make abbreviation for the parameter names
        par1_name = self._modellink._par_name[par1]
        par2_name = self._modellink._par_name[par2]

        # Create the normalized parameter value grid used to create the hcube
        # Normalization is necessary for avoiding interpolation errors
        x_proj = np.linspace(0, 1, proj_res)
        y_proj = np.linspace(0, 1, proj_res)
        X_proj, Y_proj = np.meshgrid(x_proj, y_proj, indexing='ij')

        # Get the interpolated functions describing the minimum
        # implausibility and line-of-sight depth obtained in every
        # grid point
        # TODO: Allow user to set smooth parameter for Rbf function
        # This probably means that smoothed figures have to be renamed
        f_min = Rbf(X_proj.ravel(), Y_proj.ravel(), impl_min)
        f_los = Rbf(X_proj.ravel(), Y_proj.ravel(), impl_los)

        # Set the size of the hexbin grid
        gridsize = max(10*self.__fig_kwargs['dpi'], proj_res)

        # Create normalized parameter value grid for interpolation functions
        x = np.linspace(0, 1, gridsize)
        y = np.linspace(0, 1, gridsize)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Calculate impl_min and impl_los for X, Y
        Z_min = np.zeros([gridsize, gridsize])
        Z_los = np.zeros([gridsize, gridsize])
        for i, (xi, yi) in enumerate(zip(X, Y)):
            Z_min[i] = f_min(xi, yi)
            Z_los[i] = f_los(xi, yi)

        # Flatten the mesh grids
        x = X.ravel()
        y = Y.ravel()
        z_min = Z_min.ravel()
        z_los = Z_los.ravel()

        # Check for every interpolated grid point if the corresponding z_los is
        # non-zero due to interpolation artifacts
        for i, (xi, yi) in enumerate(zip(x, y)):
            # Obtain the 2D indices of the four corner projected grid points
            corner_2D = [
                [max(x_proj.searchsorted(xi)-1, 0),
                 min(x_proj.searchsorted(xi, side='right'), proj_res-1),
                 max(x_proj.searchsorted(xi)-1, 0),
                 min(x_proj.searchsorted(xi, side='right'), proj_res-1)],
                [max(y_proj.searchsorted(yi)-1, 0),
                 min(y_proj.searchsorted(yi, side='right'), proj_res-1),
                 min(y_proj.searchsorted(yi, side='right'), proj_res-1),
                 max(y_proj.searchsorted(yi)-1, 0)]]

            # Convert 2D indices to 1D indices
            corner_1D = np.ravel_multi_index(corner_2D, [proj_res, proj_res])

            # If all corner points are zero, a point in between should be zero
            if not (impl_los[corner_1D] > 0).any():
                z_los[i] = 0

        # Check if z_min is requested to be smoothed
        if self.__smooth:
            # Calculate the highest z_los that corresponds to 0 in color
            # Matplotlib uses 256 segments in every colormap
            # Therefore, max(z_los)/256 gives the color for 0
            min_los = min(1, np.max(z_los))/256

            # Loop over all grid points
            z_min[z_los <= min_los] = impl_cut

        # Create plotted parameter value grid
        x = np.linspace(*self._modellink._par_rng[par1], gridsize)
        y = np.linspace(*self._modellink._par_rng[par2], gridsize)
        X, Y = np.meshgrid(x, y, indexing='ij')
        x = X.ravel()
        y = Y.ravel()

        # Create figure object if the figure is requested
        if self.__figure:
            f = plt.figure(constrained_layout=True, **self.__fig_kwargs)
            w_pad, h_pad, wspace, hspace = f.get_constrained_layout_pads()

            # Create GridSpec objects including a dummy Axes object
            if(self.__align == 'row'):
                gsarr = gs.GridSpec(2, 2, figure=f, height_ratios=[1, 0.00001])
                ax0 = f.add_subplot(gsarr[0, 0])
                ax1 = f.add_subplot(gsarr[0, 1])
                label_ax = f.add_subplot(gsarr[1, :])

                # Set padding to the bare minimum
                f.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad/2,
                                              wspace=wspace, hspace=0)
            else:
                gsarr = gs.GridSpec(2, 2, figure=f, width_ratios=[0.00001, 1])
                label_ax = f.add_subplot(gsarr[:, 0])
                ax0 = f.add_subplot(gsarr[0, 1])
                ax1 = f.add_subplot(gsarr[1, 1])

                # Set padding to the bare minimum
                f.set_constrained_layout_pads(w_pad=w_pad/2, h_pad=h_pad,
                                              wspace=0, hspace=hspace)

            # Set super title
            f.suptitle(r"%s. Projection (%s)" % (self.__emul_i, hcube_name),
                       fontsize='xx-large')

            # MINIMUM IMPLAUSIBILITY PLOT
            # Plot minimum implausibility
            fig1 = ax0.hexbin(x, y, z_min, gridsize, vmin=0, vmax=impl_cut,
                              **self.__impl_kwargs_3D)

            # Draw parameter estimate lines
            if self._modellink._par_est[par1] is not None:
                draw_textline(r"", x=self._modellink._par_est[par1], ax=ax0,
                              line_kwargs=self.__line_kwargs_est)
            if self._modellink._par_est[par2] is not None:
                draw_textline(r"", y=self._modellink._par_est[par2], ax=ax0,
                              line_kwargs=self.__line_kwargs_est)

            # Set axes and labels
            ax0.axis([*self._modellink._par_rng[par1],
                      *self._modellink._par_rng[par2]])
            plt.colorbar(fig1, ax=ax0, extend='max').set_label(
                "Min. Implausibility", fontsize='large')

            # LINE-OF-SIGHT DEPTH PLOT
            # Plot line-of-sight depth
            fig2 = ax1.hexbin(x, y, z_los, gridsize, vmin=0,
                              vmax=min(1, np.max(z_los)),
                              **self.__los_kwargs_3D)

            # Draw parameter estimate lines
            if self._modellink._par_est[par1] is not None:
                draw_textline(r"", x=self._modellink._par_est[par1], ax=ax1,
                              line_kwargs=self.__line_kwargs_est)
            if self._modellink._par_est[par2] is not None:
                draw_textline(r"", y=self._modellink._par_est[par2], ax=ax1,
                              line_kwargs=self.__line_kwargs_est)

            # Set axes and label
            ax1.axis([*self._modellink._par_rng[par1],
                      *self._modellink._par_rng[par2]])
            plt.colorbar(fig2, ax=ax1).set_label("Line-of-Sight Depth",
                                                 fontsize='large')

            # Make super axis labels using dummy Axes object as an empty plot
            if(self.__align == 'row'):
                ax0.set_ylabel(par2_name, fontsize='x-large')
                label_ax.set_frame_on(False)
                label_ax.get_xaxis().set_ticks([])
                label_ax.get_yaxis().set_ticks([])
                label_ax.autoscale(tight=True)
                label_ax.set_xlabel(par1_name, fontsize='x-large', labelpad=0)
            else:
                ax1.set_xlabel(par1_name, fontsize='x-large')
                label_ax.set_frame_on(False)
                label_ax.get_xaxis().set_ticks([])
                label_ax.get_yaxis().set_ticks([])
                label_ax.autoscale(tight=True)
                label_ax.set_ylabel(par2_name, fontsize='x-large', labelpad=0)

            # Save the figure
            plt.savefig(self.__get_fig_path(hcube)[self.__smooth])

            # Close the figure
            plt.close(f)

            # Log that this hypercube has been drawn
            logger.info("Finished calculating and drawing projection figure"
                        "%r." % (hcube_name))

        # If the figure data has been requested instead
        else:
            self.__fig_data[hcube_name] = {
                'impl_min': [x, y, z_min],
                'impl_los': [x, y, z_los]}
            logger.info("Finished calculating projection figure %r."
                        % (hcube_name))

    # This function returns the projection data belonging to a proj_hcube
    @docstring_substitute(hcube=hcube_doc, proj_data=proj_data_doc)
    def __get_proj_data(self, hcube):
        """
        Returns the projection data belonging to the provided hypercube
        `hcube`.

        Parameters
        ----------
        %(hcube)s

        Returns
        -------
        %(proj_data)s
        proj_res : int
            Number of emulator evaluations used to generate the grid for the
            given hypercube.
        proj_depth : int
            Number of emulator evaluations used to generate the samples in
            every grid point for the given hypercube.

        """

        # Make logger
        logger = getCLogger('PROJECTION')

        # Obtain hcube name
        hcube_name = self.__get_hcube_name(hcube)

        # Open hdf5-file
        with self._File('r', None) as file:
            # Log that projection data is being obtained
            logger.info("Obtaining projection data %r." % (hcube_name))

            # Obtain data
            data_set = file['%i/proj_hcube/%s' % (self.__emul_i, hcube_name)]
            impl_min_hcube = data_set['impl_min'][()]
            impl_los_hcube = data_set['impl_los'][()]
            res_hcube = data_set.attrs['proj_res']
            depth_hcube = data_set.attrs['proj_depth']

            # Log that projection data was obtained successfully
            logger.info("Finished obtaining projection data %r."
                        % (hcube_name))

        # Return it
        return(impl_min_hcube, impl_los_hcube, res_hcube, depth_hcube)

    # This function determines the projection hypercubes to be analyzed
    @docstring_substitute(proj_par=proj_par_doc_s)
    def __get_req_hcubes(self, proj_par):
        """
        Determines which projection hypercubes have been requested by the user.
        Also checks if these projection hypercubes have been calculated before,
        and depending on the value of :attr:`~force`, either skips them or
        recreates them.

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
                proj_par = self._modellink._get_model_par_seq(proj_par,
                                                              'proj_par')

                # Check which values in proj_par are also in active_par
                proj_par = np_array(
                    [i for i in self._emulator._active_par[self.__emul_i] if
                     i in proj_par])

                # Make sure that there are still enough values left
                if(self.__proj_2D and len(proj_par) >= 1):
                    pass
                elif(self.__proj_3D and len(proj_par) >= 2):
                    pass
                else:
                    err_msg = ("Not enough active model parameters have been "
                               "provided to make a projection figure!")
                    raise_error(err_msg, RequestError, logger)

            # Obtain list of hypercube names
            hcubes = []
            if self.__proj_2D:
                hcube_idx = list(combinations(range(len(proj_par)), 1))
                hcubes.extend(proj_par[np_array(hcube_idx)].tolist())
            if self.__proj_3D:
                hcube_idx = list(combinations(range(len(proj_par)), 2))
                if len(hcube_idx):
                    hcubes.extend(proj_par[np_array(hcube_idx)].tolist())

            # Create empty list holding hcube_par that needs to be created
            create_hcubes = []

            # Open hdf5-file
            logger.info("Checking if projection data already exists.")
            with self._File('r+', None) as file:
                # Check if data is already there and act accordingly
                for hcube in hcubes:
                    # Obtain name of this hypercube
                    hcube_name = self.__get_hcube_name(hcube)

                    # Check if projection data already exists
                    try:
                        file['%i/proj_hcube/%s' % (self.__emul_i, hcube_name)]

                    # If it does not exist, add it to the creation list
                    except KeyError:
                        logger.info("Projection data %r not found. Will be "
                                    "created." % (hcube_name))
                        create_hcubes.append(hcube)

                    # If it does exist, check value of force
                    else:
                        # If force is used, remove data and figure
                        if self.__force:
                            # Remove data
                            del file['%i/proj_hcube/%s'
                                     % (self.__emul_i, hcube_name)]
                            logger.info("Projection data %r already exists. "
                                        "Deleting." % (hcube_name))

                            # Try to remove figures as well
                            fig_path, fig_path_s = self.__get_fig_path(hcube)
                            if path.exists(fig_path):
                                logger.info("Projection figure %r already "
                                            "exists. Deleting." % (hcube_name))
                                os.remove(fig_path)
                            if path.exists(fig_path_s):
                                logger.info("Projection figure %r already "
                                            "exists. Deleting." % (hcube_name))
                                os.remove(fig_path_s)

                            # Add this hypercube to creation list
                            create_hcubes.append(hcube)

                        # If force is not used, skip creation
                        else:
                            logger.info("Projection data %r already exists. "
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

        if(len(hcube) == 1):
            return('%s' % (self._modellink._par_name[hcube[0]]))
        else:
            return('%s-%s' % (self._modellink._par_name[hcube[0]],
                              self._modellink._par_name[hcube[1]]))

    # This function returns the full path of a figure when given a hcube
    @docstring_substitute(hcube=hcube_doc, emul_i=user_emul_i_doc)
    def __get_fig_path(self, hcube, emul_i=None):
        """
        Determines the absolute path of a projection figure corresponding to a
        provided projection hypercube `hcube` in emulator iteration `emul_i`
        and returns it.

        Parameters
        ----------
        hcube : 1D array_like of int of length {1, 2} or str
            Array containing the internal integer identifiers of the main model
            parameters that require a projection hypercube.
            If str, the name of `hcube` instead
            (:meth:`~_Projection__get_hcube_name`).

        Optional
        --------
        %(emul_i)s

        Returns
        -------
        fig_path : str
            The absolute path to the requested projection figure.
        fig_path_s : str
            The absolute path to the smoothed version.

        """

        # If emul_i is None, set it to default
        if emul_i is None:
            emul_i = self.__emul_i

        # Determine the fig prefix
        fig_prefix = '%i_proj_' % (emul_i)
        fig_prefix = path.join(self._working_dir, fig_prefix)

        # Obtain name of this projection hypercube
        if isinstance(hcube, str):
            hcube_name = hcube
        else:
            hcube_name = self.__get_hcube_name(hcube)

        # Determine fig_path and fig_path_s
        fig_path = '%s(%s).png' % (fig_prefix, hcube_name)
        fig_path_s = '%s(%s)_s.png' % (fig_prefix, hcube_name)

        # Return both
        return(fig_path, fig_path_s)

    # This function returns default projection parameters
    @docstring_append(def_par_doc.format('projection'))
    def __get_default_parameters(self):
        # Create parameter dict with default parameters
        par_dict = {'proj_res': '25',
                    'proj_depth': '250'}

        # Return it
        return(sdict(par_dict))

    # This function returns default projection input arguments
    def __get_default_input_arguments(self):
        """
        Generates a dict containing default values for all input arguments.

        Returns
        -------
        kwargs_dict : dict
            Dict containing all default input argument values.

        """

        # Define variable figsizes
        figsize_c = (6.4, 4.8)
        figsize_r = (12.8, 2.4)

        # Create input argument dicts with default figure parameters
        fig_kwargs = {'dpi': 100}
        impl_kwargs_2D = {}
        impl_kwargs_3D = {'cmap': 'rainforest_r'}
        los_kwargs_2D = {}
        los_kwargs_3D = {'cmap': 'blaze'}
        line_kwargs_est = {'linestyle': '--',
                           'color': 'grey'}
        line_kwargs_cut = {'color': 'r'}

        # Create input argument dict with default projection parameters
        kwargs_dict = {'proj_type': '2D' if(self.__n_par == 2) else 'both',
                       'figure': 1,
                       'align': 'col',
                       'show_cuts': 0,
                       'smooth': 0,
                       'force': 0,
                       'fig_kwargs': fig_kwargs,
                       'impl_kwargs_2D': impl_kwargs_2D,
                       'impl_kwargs_3D': impl_kwargs_3D,
                       'los_kwargs_2D': los_kwargs_2D,
                       'los_kwargs_3D': los_kwargs_3D,
                       'line_kwargs_est': line_kwargs_est,
                       'line_kwargs_cut': line_kwargs_cut,
                       'figsize_c': figsize_c,
                       'figsize_r': figsize_r}

        # Return it
        return(sdict(kwargs_dict))

    # Set the parameters that were read in from the provided parameter file
    @docstring_append(set_par_doc.format("Projection"))
    def __set_parameters(self):
        # Log that the projection parameters are being set
        logger = getCLogger('INIT')
        logger.info("Setting projection parameters.")

        # Obtaining default projection parameter dict
        par_dict = self.__get_default_parameters()

        # Add the read-in prism dict to it
        par_dict.update(self._prism_dict)

        # More logging
        logger.info("Checking compatibility of provided projection "
                    "parameters.")

        # Number of samples used for implausibility evaluations
        if not hasattr(self, '_res'):
            self.proj_res = convert_str_seq(par_dict['proj_res'])[0]
        if not hasattr(self, '_depth'):
            self.proj_depth = convert_str_seq(par_dict['proj_depth'])[0]

        # Finish logging
        logger.info("Finished setting projection parameters.")

    # This function generates a projection hypercube to be used for emulator
    # evaluations
    @docstring_substitute(hcube=hcube_doc)
    def __get_proj_hcube(self, hcube):
        """
        Generates a projection hypercube `hcube` containing emulator evaluation
        samples The output of this function depends on the requested projection
        type.

        Parameters
        ----------
        %(hcube)s

        Returns
        -------
        proj_hcube : 3D :obj:`~numpy.ndarray` object
            3D projection hypercube of emulator evaluation samples.
            For 3D projections, the grid uses matrix indexing (second parameter
            varies the fastest).

        """

        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Log that projection hypercube is being created
        logger = getCLogger('PROJ_HCUBE')
        logger.info("Creating projection hypercube %r." % (hcube_name))

        # If hcube has 1 parameter, make 2D projection hypercube on controller
        if(self._is_controller and len(hcube) == 1):
            # Identify projected parameter
            par = hcube[0]

            # Calculate the actual depth
            if(self.__n_par == 2):
                # If n_par == 2, use normal depth
                depth = self._depth
            else:
                # If n_par > 2, multiply depth by res to have same n_sam as 3D
                depth = self._depth*self._res

            # Create empty projection hypercube array
            proj_hcube = np.zeros([self._res, depth, self.__n_par])

            # Create list that contains all the other parameters
            par_hid = list(chain(range(0, par), range(par+1, self.__n_par)))

            # Generate list with values for projected parameter
            proj_sam_set = np.linspace(*self._modellink._par_rng[par],
                                       self._res)

            # Generate latin hypercube of the remaining parameters
            hidden_sam_set = lhd(depth, self.__n_par-1,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._criterion)

            # Fill every cell in the projection hypercube accordingly
            for i in range(self._res):
                proj_hcube[i, :, par] = proj_sam_set[i]
                proj_hcube[i, :, par_hid] = hidden_sam_set.T

        # If hcube has 2 parameters, make 3D projection hypercube on controller
        elif self._is_controller:
            # Identify projected parameters
            par1 = hcube[0]
            par2 = hcube[1]

            # Create empty projection hypercube array
            proj_hcube = np.zeros([pow(self._res, 2), self._depth,
                                   self.__n_par])

            # Generate list that contains all the other parameters
            par_hid = list(chain(range(0, par1), range(par1+1, par2),
                                 range(par2+1, self.__n_par)))

            # Generate list with values for projected parameters
            proj_sam_set1 = np.linspace(*self._modellink._par_rng[par1],
                                        self._res)
            proj_sam_set2 = np.linspace(*self._modellink._par_rng[par2],
                                        self._res)

            # Generate Latin Hypercube of the remaining parameters
            hidden_sam_set = lhd(self._depth, self.__n_par-2,
                                 self._modellink._par_rng[par_hid], 'fixed',
                                 self._criterion)

            # Fill every cell in the projection hypercube accordingly
            for i in range(self._res):
                for j in range(self._res):
                    proj_hcube[i*self._res+j, :, par1] = proj_sam_set1[i]
                    proj_hcube[i*self._res+j, :, par2] = proj_sam_set2[j]
                    proj_hcube[i*self._res+j, :, par_hid] = hidden_sam_set.T

        # Workers get dummy proj_hcube
        else:
            proj_hcube = []

        # Broadcast proj_hcube to workers
        proj_hcube = self._comm.bcast(proj_hcube, 0)

        # Log that projection hypercube has been created
        logger.info("Finished creating projection hypercube %r."
                    % (hcube_name))

        # Return proj_hcube
        return(proj_hcube)

    # This function analyzes a projection hypercube
    @docstring_substitute(hcube=hcube_doc, proj_data=proj_data_doc)
    def __analyze_proj_hcube(self, hcube):
        """
        Analyzes an emulator projection hypercube `hcube`.

        Parameters
        ----------
        %(hcube)s

        Returns
        -------
        %(proj_data)s

        """

        # Obtain name of this projection hypercube
        hcube_name = self.__get_hcube_name(hcube)

        # Log that a projection hypercube is being analyzed
        logger = getCLogger('ANALYSIS')
        logger.info("Analyzing projection hypercube %r." % (hcube_name))

        # Obtain the corresponding hypercube
        proj_hcube = self.__get_proj_hcube(hcube)

        # CALCULATE AND ANALYZE IMPLAUSIBILITY
        # Create empty lists for this hypercube
        impl_min_hcube = []
        impl_los_hcube = []

        # For now, manually flatten the first two dimensions of proj_hcube
        gridsize = proj_hcube.shape[0]
        depth = proj_hcube.shape[1]
        proj_hcube = proj_hcube.reshape(gridsize*depth, self.__n_par)

        # Save current time
        start_time = time()

        # Analyze all samples in proj_hcube
        results = self._evaluate_sam_set(self.__emul_i, proj_hcube, 'project')

        # Controller only
        if self._is_controller:
            # Retrieve results
            impl_check, impl_cut = results

            # Unflatten the received results
            impl_check = impl_check.reshape(gridsize, depth)
            impl_cut = impl_cut.reshape(gridsize, depth)

            # Loop over all grid point results and save lowest impl and los
            for check_grid, cut_grid in zip(impl_check, impl_cut):
                # Calculate lowest impl in this grid point
                impl_min_hcube.append(min(cut_grid))

                # Calculate impl line-of-sight in this grid point
                impl_los_hcube.append(sum(check_grid)/depth)

            # Log that analysis has been finished
            time_diff = time()-start_time
            total = np.size(impl_check)
            logger.info("Finished projection hypercube analysis in %.2f "
                        "seconds, averaging %.2f emulator evaluations per "
                        "second." % (time_diff, total/(time_diff)))

            # Log that projection data has been created
            logger.info("Finished calculating projection data %r."
                        % (hcube_name))

            # Save projection data to hdf5
            self.__save_data({
                'nD_proj_hcube': {
                    'hcube_name': hcube_name,
                    'impl_min': impl_min_hcube,
                    'impl_los': impl_los_hcube,
                    'proj_depth': depth}})

        # Return the results for this proj_hcube
        return(np_array(impl_min_hcube), np_array(impl_los_hcube))

    # This function processes the input arguments of project
    @docstring_substitute(emul_i=get_emul_i_doc)
    def __process_input_arguments(self, emul_i, **kwargs):
        """
        Processes the input arguments given to the :meth:`~project` method.

        Parameters
        ----------
        %(emul_i)s
        kwargs : dict
            Dict of keyword arguments that were provided to :meth:`~project`.

        Generates
        ---------
        All default and provided arguments are saved to their respective
        properties.

        """

        # Make a logger
        logger = getCLogger('PROJ_INIT')
        logger.info("Processing input arguments.")

        # Make dictionary with default argument values
        kwargs_dict = self.__get_default_input_arguments()

        # Make list with forbidden figure and plot kwargs
        pop_fig_kwargs = ['num', 'ncols', 'nrows', 'sharex', 'sharey',
                          'constrained_layout']
        pop_plt_kwargs = ['x', 'y', 'C', 'gridsize', 'vmin', 'vmax']

        # Update kwargs_dict with given kwargs
        for key, value in kwargs.items():
            if key in ('fig_kwargs', 'impl_kwargs_2D', 'impl_kwargs_3D',
                       'los_kwargs_2D', 'los_kwargs_3D', 'line_kwargs_est',
                       'line_kwargs_cut'):
                if not isinstance(value, dict):
                    err_msg = ("Input argument %r is not of type 'dict'!"
                               % (key))
                    raise_error(err_msg, TypeError, logger)
                else:
                    kwargs_dict[key].update(value)

            elif(self.__n_par == 2 and key == 'proj_type'):
                pass
            else:
                kwargs_dict[key] = value
        kwargs = kwargs_dict

        # Get emul_i
        self.__emul_i = self._emulator._get_emul_i(emul_i, True)

        # Controller checking all other kwargs
        if self._is_controller:
            # Check if several parameters are bools
            self.__figure = check_vals(kwargs.pop('figure'), 'figure', 'bool')
            self.__show_cuts = check_vals(kwargs.pop('show_cuts'), 'show_cuts',
                                          'bool')
            self.__smooth = check_vals(kwargs.pop('smooth'), 'smooth', 'bool')
            self.__force = check_vals(kwargs.pop('force'), 'force', 'bool')

            # Check if proj_type parameter is a valid string
            proj_type =\
                str(kwargs.pop('proj_type')).replace("'", '').replace('"', '')
            if proj_type.lower() in ('2d', '1', 'one'):
                self.__proj_2D = 1
                self.__proj_3D = 0
            elif proj_type.lower() in ('3d', '2', 'two'):
                self.__proj_2D = 0
                self.__proj_3D = 1
            elif proj_type.lower() in ('nd', 'both'):
                self.__proj_2D = 1
                self.__proj_3D = 1
            else:
                err_msg = ("Input argument 'proj_type' is invalid (%r)!"
                           % (proj_type))
                raise_error(err_msg, ValueError, logger)

            # Check if align parameter is a valid string
            align = str(kwargs.pop('align')).replace("'", '').replace('"', '')
            if align.lower() in ('r', 'row', 'h', 'horizontal'):
                self.__align = 'row'
                kwargs['fig_kwargs']['figsize'] =\
                    kwargs['fig_kwargs'].pop('figsize', kwargs['figsize_r'])
            elif align.lower() in ('c', 'col', 'column', 'v', 'vertical'):
                self.__align = 'col'
                kwargs['fig_kwargs']['figsize'] =\
                    kwargs['fig_kwargs'].pop('figsize', kwargs['figsize_c'])
            else:
                err_msg = ("Input argument 'align' is invalid (%r)!"
                           % (align))
                raise_error(err_msg, ValueError, logger)

            # Pop all specific kwargs dicts from kwargs
            fig_kwargs = kwargs.pop('fig_kwargs')
            impl_kwargs_2D = kwargs.pop('impl_kwargs_2D')
            impl_kwargs_3D = kwargs.pop('impl_kwargs_3D')
            los_kwargs_2D = kwargs.pop('los_kwargs_2D')
            los_kwargs_3D = kwargs.pop('los_kwargs_3D')
            line_kwargs_est = kwargs.pop('line_kwargs_est')
            line_kwargs_cut = kwargs.pop('line_kwargs_cut')

            # FIG_KWARGS
            # Check if any forbidden kwargs are given and remove them
            fig_keys = list(fig_kwargs.keys())
            for key in fig_keys:
                if key in pop_fig_kwargs:
                    fig_kwargs.pop(key)

            # IMPL_KWARGS
            # Check if provided cmap is an actual cmap
            try:
                impl_kwargs_3D['cmap'] = cm.get_cmap(impl_kwargs_3D['cmap'])
            except Exception as error:
                err_msg = ("Input argument 'impl_kwargs_3D/cmap' is invalid! "
                           "(%s)" % (error))
                raise_error(err_msg, InputError, logger)

            # Check if any forbidden kwargs are given and remove them
            impl_keys = list(impl_kwargs_2D.keys())
            for key in impl_keys:
                if key in pop_plt_kwargs or (key == 'cmap'):
                    impl_kwargs_2D.pop(key)
            impl_keys = list(impl_kwargs_3D.keys())
            for key in impl_keys:
                if key in pop_plt_kwargs:
                    impl_kwargs_3D.pop(key)

            # LOS_KWARGS
            # Check if provided cmap is an actual cmap
            try:
                los_kwargs_3D['cmap'] = cm.get_cmap(los_kwargs_3D['cmap'])
            except Exception as error:
                err_msg = ("Input argument 'los_kwargs_3D/cmap' is invalid! "
                           "(%s)" % (error))
                raise_error(err_msg, InputError, logger)

            # Check if any forbidden kwargs are given and remove them
            los_keys = list(los_kwargs_2D.keys())
            for key in los_keys:
                if key in pop_plt_kwargs or (key == 'cmap'):
                    los_kwargs_2D.pop(key)
            los_keys = list(los_kwargs_3D.keys())
            for key in los_keys:
                if key in pop_plt_kwargs:
                    los_kwargs_3D.pop(key)

            # Save kwargs dicts to memory
            self.__fig_kwargs = fig_kwargs
            self.__impl_kwargs_2D = impl_kwargs_2D
            self.__impl_kwargs_3D = impl_kwargs_3D
            self.__los_kwargs_2D = los_kwargs_2D
            self.__los_kwargs_3D = los_kwargs_3D
            self.__line_kwargs_est = line_kwargs_est
            self.__line_kwargs_cut = line_kwargs_cut

        # MPI Barrier
        self._comm.Barrier()

        # Log again
        logger.info("Finished processing input arguments.")

    # This function prepares for projections to be made
    @docstring_substitute(emul_i=get_emul_i_doc, proj_par=proj_par_doc_s)
    def __prepare_projections(self, emul_i, proj_par, **kwargs):
        """
        Prepares the pipeline for the creation of the requested projections.

        Parameters
        ----------
        %(emul_i)s
        %(proj_par)s
        kwargs : dict
            Dict of keyword arguments that were provided to :meth:`~project`.

        """

        # Create logger
        logger = getCLogger('PROJ_INIT')

        # Save number of parameters as an attribute
        self.__n_par = self._modellink._n_par

        # Combine received args and kwargs with default ones
        self.__process_input_arguments(emul_i, **kwargs)

        # Controller doing some preparations
        if self._is_controller:
            # Check if it makes sense to create a projection
            if(self.__emul_i == self._emulator._emul_i):
                if not self._n_eval_sam[self.__emul_i]:
                    warn_msg = ("Requested emulator iteration %i has not been "
                                "analyzed yet. Creating projections may not be"
                                " useful." % (self.__emul_i))
                    raise_warning(warn_msg, RequestWarning, logger, 2)
                elif not self._n_impl_sam[self.__emul_i]:
                    err_msg = ("Requested emulator iteration %i has no "
                               "plausible regions. Creating projections has no"
                               " use." % (self.__emul_i))
                    raise_error(err_msg, RequestError, logger)

            # Check if projection has been created before
            with self._File('r+', None) as file:
                try:
                    file.create_group('%i/proj_hcube' % (self.__emul_i))
                except ValueError:
                    pass

            # If projection data has been requested, initialize dict
            if not self.__figure:
                self.__fig_data = sdict()

            # Set projection parameters
            self.__set_parameters()

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
        logger = getCLogger('SAVE_DATA')

        # Open hdf5-file
        with self._File('r+', None) as file:
            # Obtain the group this data needs to be saved to
            group = file['%i/proj_hcube' % (self.__emul_i)]

            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %r data at iteration %i to HDF5."
                            % (keyword, self.__emul_i))

                # Check what data keyword has been provided
                # ND_PROJ_HCUBE
                if(keyword == 'nD_proj_hcube'):
                    # Get the data set of this projection hypercube
                    data_set = group.create_group(data['hcube_name'])

                    # Save the projection data to file
                    data_set.create_dataset('impl_min', data=data['impl_min'])
                    data_set.create_dataset('impl_los', data=data['impl_los'])
                    data_set.attrs['impl_cut'] = self._impl_cut[self.__emul_i]
                    data_set.attrs['cut_idx'] = self._cut_idx[self.__emul_i]
                    data_set.attrs['proj_res'] = self._res
                    data_set.attrs['proj_depth'] = data['proj_depth']

                # INVALID KEYWORD
                else:
                    err_msg = "Invalid keyword argument provided!"
                    raise_error(err_msg, ValueError, logger)
