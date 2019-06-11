# -*- coding: utf-8 -*-

"""
Emulator
========
Provides the definition of the base class holding the emulator of the *PRISM*
package, the :class:`~Emulator` class.

"""


# %% IMPORTS
# Built-in imports
from collections import Counter
import os
from os import path
from time import time
from struct import calcsize

# Package imports
from e13tools import InputError
from e13tools.math import diff, nearest_PD
from e13tools.utils import (check_instance, convert_str_seq, delist,
                            docstring_append, docstring_substitute,
                            raise_error, raise_warning)
import h5py
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mpi4pyd import MPI
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline as Pipeline_sk
from sklearn.preprocessing import PolynomialFeatures as PF
from sortedcontainers import SortedDict as sdict, SortedSet as sset

# PRISM imports
from prism import __version__
from prism._docstrings import (adj_exp_doc, adj_var_doc, def_par_doc,
                               emul_s_seq_doc, eval_doc, full_cov_doc,
                               get_emul_i_doc, regr_cov_doc, save_data_doc_e,
                               set_par_doc, std_emul_i_doc)
from prism._internal import (RequestError, RequestWarning, check_compatibility,
                             check_vals, getCLogger, getRLogger, np_array)
from prism.modellink import ModelLink

# All declaration
__all__ = ['Emulator']

# Windows 32-bit/64-bit compatibility
int_size = 'int%i' % (calcsize('P')*8)


# %% EMULATOR CLASS DEFINITION
class Emulator(object):
    """
    Defines the :class:`~Emulator` base class of the *PRISM* package.

    Description
    -----------
    The :class:`~Emulator` class is the backbone of the *PRISM* package,
    holding all tools necessary to construct, load, save and evaluate the
    emulator of a model. It performs many checks to see if the provided
    :obj:`~prism.modellink.ModelLink` object is compatible with the
    current emulator, advises the user on alternatives when certain operations
    are requested, automatically takes care of distributing emulator systems
    over MPI ranks and more.

    Even though the purpose of the :class:`~Emulator` class is to hold only
    information about the emulator and therefore does not require any details
    about the provided :obj:`~prism.modellink.ModelLink` object, it
    will keep track of changes made to it. This is to allow the user to modify
    the properties of the :class:`~prism.modellink.ModelLink`
    subclass without causing any desynchronization problems by accident.

    The :class:`~Emulator` class requires to be linked to an instance of the
    :class:`~prism.Pipeline` class and will automatically attempt to
    do so when initialized. By default, this class should only be initialized
    from within a :obj:`~prism.Pipeline` object.

    """

    # Identify this class as being a default emulator
    _emul_type = 'default'

    def __init__(self, pipeline_obj, modellink_obj):
        """
        Initialize an instance of the :class:`~Emulator` class.

        Parameters
        ----------
        pipeline_obj : :obj:`~prism.Pipeline` object
            The :obj:`~prism.Pipeline` instance this :obj:`~Emulator` instance
            should be linked to.
        modellink_obj : :obj:`~prism.modellink.ModelLink` object
            The :obj:`~prism.modellink.ModelLink` instance that should be
            linked to `pipeline_obj`.

        """

        # Save the provided Pipeline object
        self._pipeline = pipeline_obj

        # Make pointers to MPI properties
        self._comm = self._pipeline._comm
        self._size = self._pipeline._size
        self._rank = self._pipeline._rank
        self._is_controller = self._pipeline._is_controller
        self._is_worker = self._pipeline._is_worker
        self._worker_mode = self._pipeline._worker_mode

        # Make pointer to File property
        self._File = self._pipeline._File

        # Make pointer to prism_dict property
        self._prism_dict = self._pipeline._prism_dict

        # Load the emulator and data
        self._load_emulator(modellink_obj)

        # Bind this Emulator instance to the supplied Pipeline object
        self._pipeline._emulator = self

    # %% CLASS PROPERTIES
    # General details
    @property
    def emul_type(self):
        """
        str: The type of emulator that is currently loaded. This determines the
        way in which the :obj:`~prism.Pipeline` instance will treat this
        :obj:`~Emulator` instance.

        """

        return(self._emul_type)

    @property
    def emul_load(self):
        """
        bool: Whether or not a previously constructed emulator is currently
        loaded.

        """

        return(bool(self._emul_load))

    @property
    def emul_i(self):
        """
        int: The last emulator iteration that is fully constructed for all
        emulator systems on this MPI rank.

        """

        return(self._emul_i)

    @property
    def ccheck(self):
        """
        list of str: The emulator system components that are still required to
        complete the construction of an emulator iteration on this MPI rank.
        The controller rank additionally lists the required components that are
        emulator iteration specific ('mod_real_set' and 'active_par').

        """

        return(self._ccheck)

    @property
    def n_sam(self):
        """
        int: Number of model evaluation samples that have been/will be used to
        construct an emulator iteration.

        """

        return(self._n_sam)

    @property
    def n_emul_s(self):
        """
        int: Number of emulator systems assigned to this MPI rank.

        """

        return(self._n_emul_s)

    @property
    def n_emul_s_tot(self):
        """
        int: Total number of emulator systems assigned to all MPI ranks
        combined. Only available on the controller rank.

        """

        return(self._n_emul_s_tot)

    @property
    def method(self):
        """
        str: The emulation method to use for constructing the emulator.
        Possible are 'gaussian', 'regression' and 'full'.

        """

        return(self._method)

    @property
    def use_mock(self):
        """
        bool: Whether or not mock data has been used for the construction of
        this emulator instead of actual data. If *True*, changes made to the
        data in the provided :obj:`~prism.modellink.ModelLink` object
        are ignored.

        """

        return(bool(self._use_mock))

    @property
    def use_regr_cov(self):
        """
        bool: Whether or not to take into account the regression covariance
        when calculating the covariance of the emulator, in addition to the
        Gaussian covariance.
        If :attr:`~method` == 'gaussian', this bool is not required.
        If :attr:`~method` == 'regression', this bool is always set to *True*.

        """

        return(bool(self._use_regr_cov))

    @property
    def poly_order(self):
        """
        int: Polynomial order that is considered for the regression process.
        If :attr:`~method` == 'gaussian' and
        :attr:`~prism.Pipeline.do_active_anal` is *False*, this number is not
        required.

        """

        return(self._poly_order)

    @property
    def n_cross_val(self):
        """
        int: Number of (k-fold) cross-validations that are used for determining
        the quality of the regression process. It is set to zero if
        cross-validations are not used.
        If :attr:`~method` == 'gaussian' and
        :attr:`~prism.Pipeline.do_active_anal` is *False*, this number is not
        required.

        """

        return(self._n_cross_val)

    @property
    def active_emul_s(self):
        """
        list of int: The indices of the emulator systems on this MPI rank that
        are active.

        """

        return(self._active_emul_s)

    @property
    def emul_s(self):
        """
        list of int: The indices of the emulator systems that are assigned to
        this MPI rank.

        """

        return(self._emul_s)

    @property
    def emul_s_to_core(self):
        """
        list of lists: List of the indices of the emulator systems that are
        assigned to every MPI rank. Only available on the controller rank.

        """

        return(self._emul_s_to_core)

    @property
    def data_idx_to_core(self):
        """
        list of lists: List of the data identifiers that were assigned to the
        emulator systems listed in :attr:`~emul_s_to_core`. Only available on
        the controller rank.

        """

        return(self._data_idx_to_core)

    # Active Parameters
    @property
    def active_par(self):
        """
        list of str: The model parameter names that are considered active.
        Only available on the controller rank.

        """

        return([[self._modellink._par_name[par] for
                 par in active_par] for
                active_par in self._active_par])

    @property
    def active_par_data(self):
        """
        list of str: The model parameter names that are considered active for
        every emulator system on this MPI rank.

        """

        return([[[self._modellink._par_name[par] for
                  par in active_par] for
                 active_par in active_par_data] for
                active_par_data in self._active_par_data])

    # Regression
    @property
    def rsdl_var(self):
        """
        list of float: The residual variances for every emulator system on this
        MPI rank.
        Obtained from regression process and replaces the Gaussian sigma.
        Empty if :attr:`~method` == 'gaussian'.

        """

        return(self._rsdl_var)

    @property
    def poly_coef(self):
        """
        list of :obj:`~numpy.ndarray`: The non-zero coefficients for the
        polynomial terms in the regression function for every emulator system
        on this MPI rank.
        Empty if :attr:`~method` == 'gaussian'.

        """

        return(self._poly_coef)

    @property
    def poly_coef_cov(self):
        """
        list of :obj:`~numpy.ndarray`: The covariances for all coefficients in
        :attr:`~poly_coef` for every emulator system on this MPI rank.
        Empty if :attr:`~method` == 'gaussian' or :attr:`~use_regr_cov` is
        *False*.

        """

        return(self._poly_coef_cov)

    @property
    def poly_powers(self):
        """
        list of :obj:`~numpy.ndarray`: The powers for all polynomial terms with
        non-zero coefficients in the regression function for every emulator
        system on this MPI rank.
        Empty if :attr:`~method` == 'gaussian'.

        """

        return(self._poly_powers)

    @property
    def poly_idx(self):
        """
        list of :obj:`~numpy.ndarray`: The indices for all polynomial terms
        with non-zero coefficients in the regression function for every
        emulator system on this MPI rank.
        Empty if :attr:`~method` == 'gaussian'.

        """

        return(self._poly_idx)

    # Emulator Data
    @property
    def sam_set(self):
        """
        :obj:`~numpy.ndarray`: The model evaluation samples that have been/will
        be used to construct the specified emulator iteration.

        """

        return(self._sam_set)

    @property
    def mod_set(self):
        """
        list of :obj:`~numpy.ndarray`: The model outputs corresponding to the
        samples in :attr:`~sam_set` for every emulator system on this MPI rank.

        """

        return(self._mod_set)

    @property
    def cov_mat_inv(self):
        """
        list of :obj:`~numpy.ndarray`: The inverses of the covariance matrices
        for every emulator system on this MPI rank.

        """

        return(self._cov_mat_inv)

    @property
    def exp_dot_term(self):
        """
        list of :obj:`~numpy.ndarray`: The second expectation adjustment
        dot-term values of all model evaluation samples for every emulator
        system on this MPI rank.

        """

        return(self._exp_dot_term)

    # Covariances
    @property
    def sigma(self):
        """
        float: Value of the Gaussian sigma.
        If :attr:`~method` != 'gaussian', this value is not required, since it
        is obtained from the regression process instead.

        """

        return(self._sigma)

    @property
    def l_corr(self):
        """
        :obj:`~numpy.ndarray`: The Gaussian correlation lengths for all model
        parameters, which is defined as the maximum distance between two values
        of a specific model parameter within which the Gaussian contribution to
        the correlation between the values is still significant.

        """

        return(self._l_corr)

    # %% GENERAL CLASS METHODS
    # This function checks if provided emul_i can be requested
    @docstring_substitute(emul_i=get_emul_i_doc)
    def _get_emul_i(self, emul_i, cur_iter):
        """
        Checks if the provided emulator iteration `emul_i` can be requested or
        replaces it if *None* was provided.
        This method requires all MPI ranks to call it simultaneously.

        Parameters
        ----------
        %(emul_i)s
        cur_iter : bool
            Bool determining whether the current (*True*) or the next (*False*)
            emulator iteration is requested.

        Returns
        -------
        emul_i : int
            The requested emulator iteration that passed the check.

        """

        # Log that emul_i is being selected
        logger = getCLogger('INIT')
        logger.info("Checking requested emulator iteration %s." % (emul_i))

        # Determine the emul_i that is constructed on all ranks
        global_emul_i = self._comm.allreduce(self._emul_i, op=MPI.MIN)

        # Check if provided emul_i is correct/allowed
        # If the current iteration is requested
        if cur_iter:
            if(emul_i == 0 or self._emul_load == 0 or global_emul_i == 0):
                err_msg = "Emulator is not built yet!"
                raise_error(err_msg, RequestError, logger)
            elif emul_i is None:
                emul_i = global_emul_i
            elif not(1 <= emul_i <= global_emul_i):
                err_msg = ("Requested emulator iteration %i does not exist!"
                           % (emul_i))
                raise_error(err_msg, RequestError, logger)
            else:
                emul_i = check_vals(emul_i, 'emul_i', 'pos', 'int')

        # If the next iteration is requested
        else:
            if emul_i is None:
                emul_i = global_emul_i+1
            elif not(1 <= emul_i <= global_emul_i+1):
                err_msg = ("Requested emulator iteration %i cannot be "
                           "requested!" % (emul_i))
                raise_error(err_msg, RequestError, logger)
            else:
                emul_i = check_vals(emul_i, 'emul_i', 'pos', 'int')

        # Do some logging
        logger.info("Requested emulator iteration set to %i." % (emul_i))

        # Return correct emul_i
        return(emul_i)

    # Creates a new emulator file and writes all information to it
    def _create_new_emulator(self):
        """
        Creates a new master HDF5-file that holds all the information of a new
        emulator and writes all important emulator details to it.
        Afterwards, resets all loaded emulator data and prepares the HDF5-file
        and emulator for the construction of the first emulator iteration.

        Generates
        ---------
        A new master HDF5-file 'prism.hdf5' contained in the working directory
        specified in the :obj:`~prism.Pipeline` instance, holding all
        information required to construct the first iteration of the emulator.

        """

        # Start logger
        logger = getCLogger('INIT')
        logger.info("Creating a new emulator in working directory %r."
                    % (self._pipeline._working_dir))

        # Clean-up all emulator system files
        self._cleanup_emul_files(1)

        # Set parameters from provided parameter file
        mock_par = self._set_parameters()

        # Set emul_load to 0
        self._emul_load = 0

        # Controller creating the master file
        if self._is_controller:
            # Create master hdf5-file
            with self._File('w', None) as file:
                # Save all relevant emulator parameters to hdf5
                file.attrs['sigma'] = self._sigma
                file.attrs['l_corr'] = self._l_corr
                file.attrs['method'] = self._method.encode('ascii', 'ignore')
                file.attrs['use_regr_cov'] = bool(self._use_regr_cov)
                file.attrs['poly_order'] = self._poly_order
                file.attrs['n_cross_val'] = self._n_cross_val
                file.attrs['modellink_name'] =\
                    self._modellink._name.encode('ascii', 'ignore')
                file.attrs['prism_version'] =\
                    __version__.encode('ascii', 'ignore')
                file.attrs['emul_type'] = self._emul_type.encode('ascii',
                                                                 'ignore')
                file.attrs['use_mock'] = bool(self._use_mock)

        # Check if mock data is requested
        if self._use_mock:
            # If so, temporarily save ModelLink properties as Emulator props
            # This is to make sure that one version of get_md_var() is required
            self._n_data[0] = self._modellink._n_data
            self._data_val[0] = self._modellink._data_val
            self._data_err[0] = self._modellink._data_err
            self._data_spc[0] = self._modellink._data_spc
            self._data_idx[0] = self._modellink._data_idx

            # Call get_mock_data()
            self._pipeline._get_mock_data(mock_par)

            # Controller only
            if self._is_controller:
                # Open master hdf5-file
                with self._File('r+', None) as file:
                    # Save mock_data to hdf5
                    file.attrs['mock_par'] = self._modellink._par_est

        # Prepare first emulator iteration for construction
        self._prepare_new_iteration(1)

        # Logging again
        logger.info("Finished creating new emulator.")

    # This function cleans up all the emulator files
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _cleanup_emul_files(self, emul_i):
        """
        Opens all emulator HDF5-files and removes the provided emulator
        iteration `emul_i` and subsequent iterations from them. Also removes
        any related projection figures that have default names.
        If `emul_i` == 1, all emulator HDF5-files are removed instead.

        Parameters
        ----------
        %(emul_i)s

        """

        # Do some logging
        logger = getCLogger('CLEAN-UP')
        logger.info("Cleaning up emulator HDF5-files, starting at emulator"
                    " iteration %i." % (emul_i))

        # Workers wait for controller to finish clean-up
        if self._is_worker:
            # MPI Barrier
            self._comm.Barrier()

            # Set emul_i to the last iteration still present in files
            self._emul_i = emul_i-1

            # Return
            return

        # Loop over all emulator system HDF5-files
        for s in range(0, self._n_emul_s_tot):
            # Open emulator system HDF5-file
            with self._File('r+', s) as file:
                # Save filename
                filename = file.filename

                # Loop over all requested iterations to be removed
                for i in range(emul_i, self._emul_i+2):
                    # Try to remove it, skip if not possible
                    try:
                        del file['%i' % (i)]
                    except KeyError:
                        pass

            # If emul_i is 1, remove the entire emulator system HDF5-file
            if(emul_i == 1):
                os.remove(filename)

        # Open emulator master HDF5-file if it exists
        if self._n_emul_s_tot:
            with self._File('r+', None) as file:
                # Loop over all requested iterations to be removed
                for i in range(emul_i, self._emul_i+2):
                    # Check if proj_hcube exists
                    try:
                        file['%i/proj_hcube' % (i)]
                    except KeyError:
                        pass
                    else:
                        # If so, get names of available hcubes
                        hcube_names = list(file['%i/proj_hcube' % (i)].keys())

                        # Try to remove figures for which data is available
                        for hcube in hcube_names:
                            fig_path, fig_path_s =\
                                self._pipeline._Projection__get_fig_path(hcube,
                                                                         i)
                            if path.exists(fig_path):
                                os.remove(fig_path)
                            if path.exists(fig_path_s):
                                os.remove(fig_path_s)

                    # Try to remove the iteration, skip if not possible
                    try:
                        del file['%i' % (i)]
                    except KeyError:
                        pass

            # If emul_i is 1, remove the entire emulator master HDF5-file
            if(emul_i == 1):
                os.remove(self._pipeline._hdf5_file)

        # MPI Barrier
        self._comm.Barrier()

        # Set emul_i to the last iteration still present in files
        self._emul_i = emul_i-1

        # Do more logging
        if(emul_i == 1):
            logger.info("Finished removing emulator files.")
        else:
            logger.info("Finished cleaning up emulator files.")

    # This function reads in data_idx parts, combines them and returns it
    def _read_data_idx(self, emul_s_group):
        """
        Reads in and combines the parts of the data point identifier that is
        assigned to the provided `emul_s_group`.

        Parameters
        ----------
        emul_s_group : :obj:`~h5py.Group` object
            The HDF5-group from which the data point identifier needs to be
            read in.

        Returns
        -------
        data_idx : tuple of {int, float, str}
            The combined data point identifier.

        """

        # Obtain list of attribute keys that contain data_idx parts
        idx_keys = [key for key in emul_s_group.attrs.keys()
                    if key[:8] == 'data_idx']

        # Determine the number of parts
        idx_len = len(idx_keys)

        # If there is a single part, save it instead of a tuple
        if(idx_len == 1):
            # If part is an encoded string, decode and save it
            if isinstance(emul_s_group.attrs['data_idx'], bytes):
                data_idx = emul_s_group.attrs['data_idx'].decode('utf-8')
            # Else, save it normally
            else:
                data_idx = emul_s_group.attrs['data_idx']

        # If there are multiple parts, add all of them to a list
        else:
            # Sort parts on their index number
            idx_keys = sorted(idx_keys, key=lambda x: int(x[9:]))

            # Initialize empty data_idx list
            data_idx = []

            # Loop over all parts
            for key in idx_keys:
                # If part is an encoded string, decode and save it
                if isinstance(emul_s_group.attrs[key], bytes):
                    idx_str = emul_s_group.attrs[key].decode('utf-8')
                    data_idx.append(idx_str)
                # Else, save it normally
                else:
                    data_idx.append(emul_s_group.attrs[key])

            # Convert data_idx from list to tuple
            data_idx = tuple(data_idx)

        # Return data_idx
        return(data_idx)

    # This function splits data_idx into parts and writes it to HDF5
    def _write_data_idx(self, emul_s_group, data_idx):
        """
        Splits a given `data_idx` up into individual parts and saves it as an
        attribute to the provided `emul_s_group`.

        Parameters
        ----------
        emul_s_group : :obj:`~h5py.Group` object
            The HDF5-group to which the data point identifier needs to be
            saved.
        data_idx : tuple of {int, float, str}
            The data point identifier to be saved.

        """

        # If data_idx contains multiple parts
        if isinstance(data_idx, tuple):
            # Obtain list of attribute keys required for the data_idx parts
            idx_keys = ['data_idx_%i' % (i) for i in range(len(data_idx))]

            # Loop over all parts
            for key, idx in zip(idx_keys, data_idx):
                # If part is a string, encode and save it
                if isinstance(idx, str):
                    emul_s_group.attrs[key] = idx.encode('ascii', 'ignore')
                # Else, save it normally
                else:
                    emul_s_group.attrs[key] = idx

        # If data_idx contains a single part, save it
        else:
            # If part is a string, encode and save it
            if isinstance(data_idx, str):
                emul_s_group.attrs['data_idx'] =\
                    data_idx.encode('ascii', 'ignore')
            # Else, save it normally
            else:
                emul_s_group.attrs['data_idx'] = data_idx

    # This function matches data points with those in a previous iteration
    # TODO: Write this function and _assign_emul_s simpler and dependent
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _assign_data_idx(self, emul_i):
        """
        Determines the emulator system each data point in the provided emulator
        iteration `emul_i` should be assigned to, in order to make sure that
        recurring data points have the same emulator system index as in the
        previous emulator iteration. If multiple options are possible, data
        points are assigned such to spread them as much as possible.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        data_to_emul_s : list of int
            The index of the emulator system that each data point should be
            assigned to.
        n_emul_s : int
            The total number of active and passive emulator systems there will
            be in the provided emulator iteration.

        Examples
        --------
        If the number of data points is less than the previous iteration:

            >>> emul_i = 2
            >>> self._data_idx[emul_i-1]
            ['A', 'B', 'C', 'D', 'E']
            >>> self._modellink._data_idx
            ['B', 'F', 'G', 'E']
            >>> self._assign_data_idx(emul_i)
            ([1, 3, 2, 4], 5)


        If the number of data points is more than the previous iteration:

            >>> emul_i = 2
            >>> self._data_idx[emul_i-1]
            ['A', 'B', 'C', 'D', 'E']
            >>> self._modellink._data_idx
            ['B', 'F', 'G', 'E', 'A', 'C']
            >>> self._assign_data_idx(emul_i)
            ([1, 5, 3, 4, 0, 2], 6)


        If there is no previous iteration:

            >>> emul_i = 1
            >>> self._data_idx[emul_i-1]
            []
            >>> self._modellink._data_idx
            ['B', 'F', 'G', 'E', 'A', 'C']
            >>> self._assign_data_idx(emul_i)
            ([5, 4, 3, 2, 1, 0], 6)

        """

        # Do some logging
        logger = getCLogger('INIT')
        logger.info("Assigning model comparison data points to emulator "
                    "systems for emulator iteration %i." % (emul_i))

        # Create empty Counter for number of emulator system occurances
        emul_s_counter = Counter()

        # Calculate the total number of active and passive emulator systems
        n_emul_s = max(self._modellink._n_data, self._n_emul_s_tot)

        # Create some empty lists
        active_emul_s_list = [[]]
        data_idx_list = []

        # Open hdf5-file
        with self._File('r', None) as file:
            # Loop over all previous iterations
            for i in range(1, emul_i):
                # Obtain the active emulator systems for this iteration
                active_emul_s_list.append(
                    [int(key[5:]) for key in file['%i' % (i)].keys() if
                     key[:5] == 'emul_'])

            # Loop over all active emulator systems in the last iteration
            for emul_s in active_emul_s_list[-1]:
                data_set = file['%i/emul_%i' % (emul_i-1, emul_s)]

                # Read in all data_idx parts and combine them
                data_idx_list.append(self._read_data_idx(data_set))

        # Set number of occurances for all emulator systems to 0
        for emul_s in range(n_emul_s):
            emul_s_counter[emul_s] = 0

        # Count how many times each emulator system has already been active
        for active_emul_s in active_emul_s_list:
            emul_s_counter.update(active_emul_s)

        # Create empty data_to_emul_s list
        data_to_emul_s = [[] for _ in range(self._modellink._n_data)]

        # Assign all recurring data points to the correct emulator systems
        for i, data_idx in enumerate(self._modellink._data_idx):
            # Check for every data_idx if it existed in the previous iteration
            try:
                emul_s = active_emul_s_list[-1][data_idx_list.index(data_idx)]
            except ValueError:
                pass
            else:
                # If it existed, assign data_idx to corresponding emul_s
                data_to_emul_s[i] = emul_s

                # Also remove emul_s from emul_s_counter
                emul_s_counter.pop(emul_s)

        # Assign all 'new' data points to the empty emulator systems
        for i in range(self._modellink._n_data):
            # Check if this data point has already been assigned
            if(data_to_emul_s[i] == []):
                # If not, check which emulator system is the least common
                emul_s = emul_s_counter.most_common()[-1][0]

                # Assign data point to this emulator system and remove it
                data_to_emul_s[i] = emul_s
                emul_s_counter.pop(emul_s)

        # More logging
        logger.info("Finished assigning data points to emulator systems.")

        # Return data_to_emul_s and n_emul_s
        return(data_to_emul_s, n_emul_s)

    # This function determines how to assign emulator systems to MPI ranks
    # TODO: Might want to include the size (n_sam) of every system as well
    # TODO: May also want to include low-level MPI distribution
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _assign_emul_s(self, emul_i):
        """
        Determines which emulator systems (files) should be assigned to which
        MPI rank in order to balance the number of active emulator systems on
        every rank for every iteration up to the provided emulator iteration
        `emul_i`. If multiple choices can achieve this, the emulator systems
        are automatically spread out such that the total number of active
        emulator systems on a single rank is also balanced as much as possible.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        emul_s_to_core : list of lists
            A list containing the emulator systems that have been assigned to
            the corresponding MPI rank by the controller.

        Notes
        -----
        Currently, this function only uses high-level MPI. Additional speed can
        be obtained by also implementing low-level MPI, which will potentially
        be done in the future.

        """

        # Start logging
        logger = getCLogger('INIT')
        logger.info("Determining emulator system assignments up to emulator "
                    "iteration %i for available MPI ranks." % (emul_i))

        # Create empty list of active emulator systems
        active_emul_s_list = [[]]

        # Open hdf5-file
        with self._File('r', None) as file:
            logger.info("Determining active emulator systems in every "
                        "emulator iteration.")

            # Determine the active emulator systems in every iteration
            for i in range(1, emul_i+1):
                active_emul_s_list.append(
                    [int(key[5:]) for key in file['%i' % (i)].keys() if
                     key[:5] == 'emul_'])

        # Determine number of active emulator systems in each iteration
        n_active_emul_s = [[i, len(active_emul_s)] for i, active_emul_s in
                           enumerate(active_emul_s_list[:emul_i+1])]
        iter_size = sorted(n_active_emul_s, key=lambda x: x[1])

        # Create empty emul_s_to_core list
        emul_s_to_core = [[] for _ in range(self._size)]

        # Create empty Counter for total number of assigned systems
        core_cntr = Counter()

        # Create empty Counter for total number of system occurances
        emul_s_cntr = Counter()

        # Determine how many times a specific emulator system is active
        for active_emul_s in active_emul_s_list:
            emul_s_cntr.update(active_emul_s)

        # Set the number of assigned emulator systems for each core to zero
        for rank in range(self._size):
            core_cntr[rank] = 0

        # Create empty list holding all assigned emulator systems
        emul_s_chosen = []

        # Loop over all iterations, from smallest to largest
        for i, size in iter_size:
            # Set the number of assigned systems in this iteration to zero
            iter_core_cntr = [0 for _ in range(self._size)]

            # Create empty Counter for number of system occurances that are
            # also in this iteration
            iter_emul_s_cntr = Counter()

            # Fill that counter with systems that are not assigned yet
            for emul_s in active_emul_s_list[i]:
                # If this system is not assigned yet, copy its size
                if emul_s not in emul_s_chosen:
                    iter_emul_s_cntr[emul_s] = emul_s_cntr[emul_s]

                # Check if certain systems have already been assigned
                for j, emul_s_list in enumerate(emul_s_to_core):
                    iter_core_cntr[j] += emul_s_list.count(emul_s)

            # Set the minimum number of assigned systems for a core to 0
            min_count = 0

            # While not all emulator systems in this iteration are assigned
            while(sum(iter_core_cntr) != size):
                # Determine cores that have minimum number of assignments
                min_cores = [j for j, num in enumerate(iter_core_cntr) if(
                    num == min_count)]

                # If no core has this minimum size, increase it by 1
                if(len(min_cores) == 0):
                    min_count += 1

                # If one core has this number, assign system with lowest
                # number of occurances to it and remove it from the list
                elif(len(min_cores) == 1):
                    core = min_cores[0]
                    emul_s, emul_size = iter_emul_s_cntr.most_common()[-1]
                    core_cntr[core] += emul_size
                    emul_s_chosen.append(emul_s)
                    emul_s_to_core[core].append(emul_s)
                    iter_core_cntr[core] += 1
                    iter_emul_s_cntr.pop(emul_s)
                    min_count += 1

                # If more than one core has this number, determine the core
                # that has the lowest total number of assigned systems and
                # assign the system with the highest number of occurances
                # to it
                else:
                    emul_s, emul_size = iter_emul_s_cntr.most_common()[0]
                    core_sizes = [[j, core_cntr[j]] for j in min_cores]
                    core_lowest = min(core_sizes, key=lambda x: x[1])[0]
                    core_cntr[core_lowest] += emul_size
                    emul_s_chosen.append(emul_s)
                    emul_s_to_core[core_lowest].append(emul_s)
                    iter_core_cntr[core_lowest] += 1
                    iter_emul_s_cntr.pop(emul_s)

        # Log that assignments have been determined
        logger.info("Finished determining emulator system assignments.")

        # Return emul_s_to_core
        return(emul_s_to_core)

    # Prepares the emulator for a new iteration
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _prepare_new_iteration(self, emul_i):
        """
        Prepares the emulator for the construction of a new iteration `emul_i`.
        Checks if this iteration can be prepared or if it has been prepared
        before, and acts accordingly.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        reload : bool
            Bool indicating whether or not the controller rank of the
            :obj:`~prism.Pipeline` instance needs to reload its data.

        Generates
        ---------
        A new group in the master HDF5-file with the emulator iteration as its
        name, containing subgroups corresponding to all emulator systems that
        will be used in this iteration.

        Notes
        -----
        Preparing an iteration that has been prepared before, causes that and
        all subsequent iterations of the emulator to be deleted.
        A check is carried out to see if it was necessary to reprepare the
        requested iteration and a warning is given if this check fails.

        """

        # Logger
        logger = getCLogger('EMUL_PREP')
        logger.info("Preparing emulator iteration %i for construction."
                    % (emul_i))

        # Check if new iteration can be constructed
        logger.info("Checking if emulator iteration can be prepared.")
        if(emul_i == 1):
            # Set reload flag to 1
            reload = 1
        elif not(1 <= emul_i-1 <= self._emul_i):
            err_msg = ("Preparation of emulator iteration %i is only available"
                       " when all previous iterations exist!" % (emul_i))
            raise_error(err_msg, RequestError, logger)
        elif(emul_i-1 == self._emul_i):
            # Set reload flag to 0
            reload = 0
        else:
            logger.info("Emulator iteration %i already exists." % (emul_i))

            # Check if repreparation was actually necessary
            diff_flag = 1
            for i, idx in enumerate(self._modellink._data_idx):
                try:
                    j = self._data_idx[emul_i].index(idx)
                except ValueError:
                    break
                else:
                    if(self._data_val[emul_i][j] !=
                       self._modellink._data_val[i]):
                        break
                    if(self._data_err[emul_i][j] !=
                       self._modellink._data_err[i]):
                        break
                    if(self._data_spc[emul_i][j] !=
                       self._modellink._data_spc[i]):
                        break
            # If not, set diff_flag to 0
            else:
                diff_flag = 0

            # Gather the diff_flags on the controller
            diff_flag = np.any(self._comm.gather(diff_flag, 0))

            # If all diff_flags were 0, give out a warning
            if self._is_controller and not diff_flag:
                warn_msg = ("No differences in model comparison data detected."
                            "\nUnless this repreparation was intentional, "
                            "using the 'analyze' method of the Pipeline class "
                            "is much faster for reanalyzing the emulator with "
                            "new implausibility analysis parameters.")
                raise_warning(warn_msg, RequestWarning, logger, 3)

            # Set reload flag to 1
            reload = 1

        # Clean-up all emulator files if emul_i is not 1
        if(emul_i != 1):
            self._cleanup_emul_files(emul_i)

        # Controller preparing the emulator iteration
        if self._is_controller:
            # Assign data points to emulator systems
            data_to_emul_s, n_emul_s = self._assign_data_idx(emul_i)

            # Open hdf5-file
            with self._File('r+', None) as file:
                # Make group for emulator iteration
                group = file.create_group('%i' % (emul_i))

                # Save the number of data points
                group.attrs['n_data'] = self._modellink._n_data

                # Create an empty data set for statistics as attributes
                group.create_dataset('statistics', data=h5py.Empty(float))

                # Save the total number of active and passive emulator systems
                group.attrs['n_emul_s'] = n_emul_s

                # Create groups for all data points
                logger.info("Preparing emulator system files.")
                for i, emul_s in enumerate(data_to_emul_s):
                    with self._File('a', emul_s) as file_i:
                        # Make iteration group for this emulator system
                        data_set = file_i.create_group('%i' % (emul_i))

                        # Save data value, errors and space to this system
                        data_set.attrs['data_val'] =\
                            self._modellink._data_val[i]
                        data_set.attrs['data_err'] =\
                            self._modellink._data_err[i]
                        data_set.attrs['data_spc'] =\
                            self._modellink._data_spc[i].encode('ascii',
                                                                'ignore')

                        # Save data_idx in portions to make it HDF5-compatible
                        self._write_data_idx(data_set,
                                             self._modellink._data_idx[i])

                        # Create external link between file_i and master file
                        group['emul_%i' % (emul_s)] = h5py.ExternalLink(
                            path.basename(file_i.filename), '%i' % (emul_i))

        # MPI Barrier
        self._comm.Barrier()

        # All ranks reload the emulator systems to allow for reassignments
        self._emul_i = emul_i
        self._emul_load = 1
        self._load_data(emul_i)

        # Logging
        logger.info("Finished preparing emulator iteration.")

        # Return the result
        return(reload)

    # This function constructs the emulator iteration emul_i
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _construct_iteration(self, emul_i):
        """
        Constructs the emulator iteration corresponding to the provided
        `emul_i`, by performing the given emulation method and pre-calculating
        the prior expectation and variance values of the used model evaluation
        samples.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        All data sets that are required to evaluate the emulator at the
        constructed iteration.

        """

        # Save current time on controller
        if self._is_controller:
            start_time = time()

        # Get the emul_s_seq
        emul_s_seq = self._active_emul_s[emul_i]

        # Determine active parameters
        ccheck_active_par = [emul_s for emul_s in emul_s_seq if
                             'active_par_data' in self._ccheck[emul_i][emul_s]]
        if len(ccheck_active_par):
            self._get_active_par(emul_i, ccheck_active_par)

        # Check if regression is required
        if(self._method.lower() in ('regression', 'full')):
            # Perform regression
            ccheck_regression = [emul_s for emul_s in emul_s_seq if
                                 'regression' in self._ccheck[emul_i][emul_s]]
            if len(ccheck_regression):
                self._do_regression(emul_i, ccheck_regression)

        # Calculate the covariance matrices of sam_set
        # TODO: Implement system that, if needed, calculates cov_mat in seq
        # This is to avoid requiring massive amounts of RAM during calculation
        ccheck_cov_mat = [emul_s for emul_s in emul_s_seq if
                          'cov_mat' in self._ccheck[emul_i][emul_s]]
        if len(ccheck_cov_mat):
            self._get_cov_matrix(emul_i, ccheck_cov_mat)

        # Calculate the second dot-term for the adjusted expectation
        ccheck_exp_dot_term = [emul_s for emul_s in emul_s_seq if
                               'exp_dot_term' in self._ccheck[emul_i][emul_s]]
        if len(ccheck_exp_dot_term):
            self._get_exp_dot_term(emul_i, ccheck_exp_dot_term)

        # If a worker is finished, set current emul_i to constructed emul_i
        if self._is_worker:
            self._emul_i = emul_i

        # MPI Barrier
        self._comm.Barrier()

        # If everything is done, gather the total set of active parameters
        active_par_data = self._comm.gather(self._active_par_data[emul_i], 0)

        # Allow the controller to save them
        if self._is_controller and 'active_par' in self._ccheck[emul_i]:
            active_par = sset()
            for active_par_rank in active_par_data:
                active_par.update(*active_par_rank)
            self._save_data(emul_i, None, {
                'active_par': np_array(active_par)})

            # Set current emul_i to constructed emul_i
            self._emul_i = emul_i

            # Save time difference and communicator size
            self._pipeline._save_statistics(emul_i, {
                'emul_construct_time': ['%.2f' % (time()-start_time), 's'],
                'MPI_comm_size_cons': ['%i' % (self._size), '']})

        # MPI Barrier
        self._comm.Barrier()

    # This is function 'E_D(f(x'))'
    # This function gives the adjusted emulator expectation value back
    @docstring_append(adj_exp_doc)
    def _get_adj_exp(self, emul_i, emul_s_seq, par_set, cov_vec):
        # Obtain prior expectation value of par_set
        prior_exp_par_set = self._get_prior_exp(emul_i, emul_s_seq, par_set)

        # Create empty adj_exp_val
        adj_exp_val = np.zeros(len(emul_s_seq))

        # Calculate the adjusted emulator expectation value at given par_set
        for i, emul_s in enumerate(emul_s_seq):
            adj_exp_val[i] = prior_exp_par_set[i] +\
                cov_vec[i].T @ self._exp_dot_term[emul_i][emul_s]

        # Return it
        return(adj_exp_val)

    # This is function 'Var_D(f(x'))'
    # This function gives the adjusted emulator variance value back
    # TODO: Find out why many adj_var have a very low value for GaussianLink
    @docstring_append(adj_var_doc)
    def _get_adj_var(self, emul_i, emul_s_seq, par_set, cov_vec):
        # Obtain prior variance value of par_set
        prior_var_par_set = self._get_cov(emul_i, emul_s_seq, par_set, par_set)

        # Create empty adj_var_val
        adj_var_val = np.zeros(len(emul_s_seq))

        # Calculate the adjusted emulator variance value at given par_set
        for i, emul_s in enumerate(emul_s_seq):
            adj_var_val[i] = prior_var_par_set[i] -\
                cov_vec[i].T @ self._cov_mat_inv[emul_i][emul_s] @ cov_vec[i]

        # Return it
        return(adj_var_val)

    # This function evaluates the emulator at a given emul_i and par_set and
    # returns the adjusted expectation and variance values
    # TODO: Take sam_set instead of par_set?
    @docstring_append(eval_doc)
    def _evaluate(self, emul_i, par_set):
        # Obtain active emulator systems for this iteration
        emul_s_seq = self._active_emul_s[emul_i]

        # Calculate the covariance vector for this par_set
        cov_vec = self._get_cov(emul_i, emul_s_seq, par_set, None)

        # Calculate the adjusted expectation and variance values
        adj_exp_val = self._get_adj_exp(emul_i, emul_s_seq, par_set, cov_vec)
        adj_var_val = self._get_adj_var(emul_i, emul_s_seq, par_set, cov_vec)

        # Make sure that adj_var_val cannot drop below zero
        adj_var_val[adj_var_val < 0] = 0

        # Return adj_exp_val and adj_var_val
        return(adj_exp_val, adj_var_val)

    # This function extracts the set of active parameters
    # TODO: Write code cleaner, if possible
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_active_par(self, emul_i, emul_s_seq):
        """
        Determines the active parameters to be used for every emulator system
        listed in `emul_s_seq` in the provided emulator iteration `emul_i`.
        Uses backwards stepwise elimination to determine the set of active
        parameters. The polynomial order that is used in the stepwise
        elimination depends on :attr:`~poly_order`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates (for every emulator system)
        -------------------------------------
        active_par_data : 1D :obj:`~numpy.ndarray` object
            Array containing the indices of all the parameters that are active
            in the emulator iteration `emul_i`.

        """

        # Log that active parameters are being determined
        logger = getRLogger('ACTIVE_PAR')
        logger.info("Determining active parameters.")

        # Loop over all emulator systems and determine active parameters
        for emul_s in emul_s_seq:
            # Initialize active parameters data set
            active_par_data = sset()

            # Check if previously active parameters must be active again
            if(self._pipeline._freeze_active_par and
               emul_s in self._active_emul_s[emul_i-1] and
               self._data_idx[emul_i-1][emul_s] ==
               self._data_idx[emul_i][emul_s]):
                active_par_data.update(self._active_par_data[emul_i-1][emul_s])

            # Check if active parameters analysis has been requested
            if not self._pipeline._do_active_anal:
                # If not requested, then save all potentially active parameters
                active_par_data.update(self._pipeline._pot_active_par)

            # If requested, perform a sequential backward stepwise regression
            else:
                # Obtain frozen+potentially active parameters list
                frz_pot_par = sset(active_par_data)
                frz_pot_par.update(self._pipeline._pot_active_par)
                frz_pot_par = list(frz_pot_par)
                frz_pot_idx = list(range(len(frz_pot_par)))

                # Obtain non-frozen potentially active parameters
                non_frz_par = [par for par in self._pipeline._pot_active_par
                               if par not in active_par_data]
                non_frz_idx = [frz_pot_par.index(par) for par in non_frz_par]

                # If non_frz_par has at least 1 element, carry out analysis
                if len(non_frz_par):
                    # Create SequentialFeatureSelector object
                    sfs_obj = SFS(LR(), k_features='parsimonious',
                                  forward=False, floating=False, scoring='r2',
                                  cv=self._n_cross_val)

                    # Obtain sam_set of frz_pot_par
                    frz_pot_sam_set = self._sam_set[emul_i][:, frz_pot_par]

                    # Obtain polynomial terms of frz_pot_sam_set
                    pf_obj = PF(self._poly_order, include_bias=False)
                    frz_pot_poly_terms = pf_obj.fit_transform(frz_pot_sam_set)

                    # Perform linear regression with linear terms only
                    sfs_obj.fit(frz_pot_sam_set, self._mod_set[emul_i][emul_s])

                    # Extract active parameters due to linear significance
                    act_idx_lin = list(sfs_obj.k_feature_idx_)

                    # Get passive non-frozen parameters in linear significance
                    pas_idx_lin = [i for i in non_frz_idx if
                                   i not in act_idx_lin]

                    # Make sure frozen parameters are considered active
                    act_idx_lin = [i for i in frz_pot_idx if
                                   i not in pas_idx_lin]
                    act_idx = list(act_idx_lin)

                    # Do n-order polynomial regression for every passive par
                    for i in pas_idx_lin:
                        # Check which polynomial terms involve this passive par
                        poly_idx = pf_obj.powers_[:, i] != 0

                        # Add the active linear terms as well
                        poly_idx[act_idx_lin] = 1

                        # Convert poly_idx to an array of indices
                        poly_idx = np.arange(len(poly_idx))[poly_idx]

                        # Obtain polynomial terms for this passive parameter
                        poly_terms = frz_pot_poly_terms[:, poly_idx]

                        # Perform linear regression with addition of poly terms
                        sfs_obj.fit(poly_terms, self._mod_set[emul_i][emul_s])

                        # Extract indices of active polynomial terms
                        act_idx_poly = poly_idx[list(sfs_obj.k_feature_idx_)]

                        # Check if any additional polynomial terms survived
                        # Add i to act_idx if this is the case
                        if np.any([j not in act_idx_lin for
                                   j in act_idx_poly]):
                            act_idx.append(i)

                    # Update the active parameters for this emulator system
                    active_par_data.update(np_array(frz_pot_par)[act_idx])

            # Log the resulting active parameters
            logger.info("Active parameters for emulator system %i: %s"
                        % (self._emul_s[emul_s],
                           [self._modellink._par_name[par]
                            for par in active_par_data]))

            # Convert active_par_data to a NumPy array and save
            self._save_data(emul_i, emul_s, {
                'active_par_data': np_array(active_par_data)})

        # Log that active parameter determination is finished
        logger.info("Finished determining active parameters.")

    # This function performs a forward stepwise regression on sam_set
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _do_regression(self, emul_i, emul_s_seq):
        """
        Performs a forward stepwise linear regression for all requested
        emulator systems `emul_s_seq` in the provided emulator iteration
        `emul_i`. Calculates what the expectation values of all polynomial
        coefficients are. The polynomial order that is used in the regression
        depends on :attr:`~poly_order`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates (for every emulator system)
        -------------------------------------
        rsdl_var : float
            Residual variance of the regression function.
        regr_score : float
            Fit-score of the regression function.
        poly_coef : 1D :obj:`~numpy.ndarray` object
            Array containing the expectation values of the non-zero polynomial
            coefficients.
        poly_powers : 2D :obj:`~numpy.ndarray` object
            Array containing the powers of the non-zero polynomial terms in the
            regression function.
        poly_idx : 1D :obj:`~numpy.ndarray` object
            Array containing the indices of the non-zero polynomial terms in
            the regression function.
        poly_coef_cov : 1D :obj:`~numpy.ndarray` object (if \
            :attr:`~use_regr_cov` is *True*)
            Array containing the covariance values of the non-zero polynomial
            coefficients.

        """

        # Create logger
        logger = getRLogger('REGRESSION')
        logger.info("Performing regression.")

        # Create SequentialFeatureSelector object
        sfs_obj = SFS(LR(), k_features='best', forward=True, floating=False,
                      scoring='neg_mean_squared_error',
                      cv=self._n_cross_val)

        # Create Scikit-learn Pipeline object
        # The bias/intercept/constant-term is not included in the SFS object to
        # ensure that it is taken into account in the linear regression, since
        # it is required for getting the residual variance. It also ensures
        # that the SFS does not focus on the constant-term in its calculations.
        pipe = Pipeline_sk([('poly', PF(self._poly_order, include_bias=False)),
                            ('SFS', sfs_obj),
                            ('linear', LR())])

        # Loop over all emulator systems and perform a regression on them
        for emul_s in emul_s_seq:
            # Extract active_sam_set
            active_sam_set = self._sam_set[emul_i][
                :, self._active_par_data[emul_i][emul_s]]

            # Perform regression for this emulator system
            pipe.fit(active_sam_set, self._mod_set[emul_i][emul_s])

            # Obtain the corresponding polynomial indices
            poly_idx = np_array(pipe.named_steps['SFS'].k_feature_idx_)

            # Extract sam_set_poly
            sam_set_poly = pipe.named_steps['poly'].transform(
                active_sam_set)[:, poly_idx]

            # Extract the residual variance
            rsdl_var = mse(self._mod_set[emul_i][emul_s],
                           pipe.named_steps['linear'].predict(sam_set_poly))

            # Log the score of the regression process
            regr_score = pipe.named_steps['linear'].score(
                sam_set_poly, self._mod_set[emul_i][emul_s])
            logger.info("Regression score for emulator system %i: %f."
                        % (self._emul_s[emul_s], regr_score))

            # Obtain polynomial powers and include intercept term
            poly_powers = np.insert(pipe.named_steps['poly'].powers_[poly_idx],
                                    0, 0, 0)

            # Obtain polynomial coefficients and include intercept term
            poly_coef = np.insert(pipe.named_steps['linear'].coef_, 0,
                                  pipe.named_steps['linear'].intercept_, 0)

            # Check every polynomial coefficient if it is significant enough
            poly_sign = ~np.isclose(poly_coef, 0)

            # Only include significant polynomial terms unless none are
            poly_powers = poly_powers[poly_sign if sum(poly_sign) else ()]
            poly_coef = poly_coef[poly_sign if sum(poly_sign) else ()]

            # Redetermine the active parameters, poly_powers and poly_idx
            new_active_par_idx = [np.any(powers) for powers in poly_powers.T]
            poly_powers = poly_powers[:, new_active_par_idx]
            new_active_par =\
                self._active_par_data[emul_i][emul_s][new_active_par_idx]
            new_pf_obj = PF(self._poly_order).fit([[0]*poly_powers.shape[1]])
            new_powers = new_pf_obj.powers_
            new_powers_list = new_powers.tolist()
            poly_powers_list = poly_powers.tolist()
            poly_idx = np_array([i for i, powers in enumerate(new_powers_list)
                                 if powers in poly_powers_list])

            # If regression covariances are requested, calculate them
            if self._use_regr_cov:
                # Redetermine the active sam_set_poly
                active_sam_set = self._sam_set[emul_i][:, new_active_par]
                sam_set_poly =\
                    new_pf_obj.fit_transform(active_sam_set)[:, poly_idx]

                # Calculate the poly_coef covariances
                poly_coef_cov =\
                    rsdl_var*inv(sam_set_poly.T @ sam_set_poly).flatten()

            # Create regression data dict
            regr_data_dict = {
                'active_par': new_active_par,
                'rsdl_var': rsdl_var,
                'regr_score': regr_score,
                'poly_coef': poly_coef,
                'poly_powers': poly_powers,
                'poly_idx': poly_idx}

            # If regression covariance is used, add it to regr_data_dict
            if self._use_regr_cov:
                regr_data_dict['poly_coef_cov'] = poly_coef_cov

            # Save everything to hdf5
            self._save_data(emul_i, emul_s, {
                'regression': regr_data_dict})

        # Log that this is finished
        logger.info("Finished performing regression.")

    # This function gives the prior expectation value
    # This is function 'E(f(x'))' or 'u(x')'
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_prior_exp(self, emul_i, emul_s_seq, par_set):
        """
        Calculates the prior expectation value for requested emulator systems
        `emul_s_seq` at a given emulator iteration `emul_i` for specified
        parameter set `par_set`. This expectation depends on :attr:`~method`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
        par_set : 1D :obj:`~numpy.ndarray` object or None
            If *None*, calculate the prior expectation values of sam_set.
            If not *None*, calculate the prior expectation value for the given
            model parameter value set.

        Returns
        -------
        prior_exp : 1D or 2D :obj:`~numpy.ndarray` object
            Prior expectation values for either sam_set or `par_set` for
            requested emulator systems.

        """

        # If prior_exp of sam_set is requested (exp_dot_term)
        if par_set is None:
            # Initialize empty prior expectation
            prior_exp = np.zeros([len(emul_s_seq), self._n_sam[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Gaussian prior expectation is equal to the mean, which is 0
                prior_exp += 0
            if self._method.lower() in ('regression', 'full'):
                for i, emul_s in enumerate(emul_s_seq):
                    # Initialize PF object
                    pf_obj = PF(self._poly_order)

                    # Obtain the polynomial terms
                    poly_terms = pf_obj.fit_transform(
                        self._sam_set[emul_i][
                            :, self._active_par_data[emul_i][emul_s]])[
                                :, self._poly_idx[emul_i][emul_s]]
                    prior_exp[i] += np.sum(
                        self._poly_coef[emul_i][emul_s]*poly_terms, axis=-1)

        # If prior_exp of par_set is requested (adj_exp)
        else:
            # Initialize empty prior expectation
            prior_exp = np.zeros(len(emul_s_seq))

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Gaussian prior expectation is equal to the mean, which is 0
                prior_exp += 0
            if self._method.lower() in ('regression', 'full'):
                for i, emul_s in enumerate(emul_s_seq):
                    poly_terms = np.product(pow(
                        par_set[self._active_par_data[emul_i][emul_s]],
                        self._poly_powers[emul_i][emul_s]), axis=-1)
                    prior_exp[i] += np.sum(
                        self._poly_coef[emul_i][emul_s]*poly_terms, axis=-1)

        # Return it
        return(prior_exp)

    # This function pre-calculates the second adjustment dot-term
    # This is function 'Var(D)^-1*(D-E(D))'
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_exp_dot_term(self, emul_i, emul_s_seq):
        """
        Pre-calculates the second expectation adjustment dot-term for requested
        emulator systems `emul_s_seq` at a given emulator iteration `emul_i`
        for all model evaluation samples and saves it for later use.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates
        ---------
        exp_dot_term : 2D :obj:`~numpy.ndarray` object
            2D array containing the pre-calculated values for the second
            adjustment dot-term of the adjusted expectation for requested
            emulator systems.

        """

        # Create logger
        logger = getRLogger('DOT_TERM')
        logger.info("Pre-calculating second expectation adjustment dot-term "
                    "for known samples at emulator iteration %i." % (emul_i))

        # Obtain prior expectation value of sam_set
        prior_exp_sam_set = self._get_prior_exp(emul_i, emul_s_seq, None)

        # Calculate the exp_dot_term values and save it to hdf5
        for i, emul_s in enumerate(emul_s_seq):
            exp_dot_term = self._cov_mat_inv[emul_i][emul_s] @\
                (self._mod_set[emul_i][emul_s]-prior_exp_sam_set[i])
            self._save_data(emul_i, emul_s, {
                'exp_dot_term': {
                    'prior_exp_sam_set': prior_exp_sam_set[i],
                    'exp_dot_term': exp_dot_term}})

        # Log again
        logger.info("Finished pre-calculating second adjustment dot-term "
                    "values.")

    # This function calculates the covariance between parameter sets
    # This is function 'Cov(f(x), f(x'))' or 'c(x,x')
    # TODO: Improve Gaussian-only method by making sigma data point dependent
    @docstring_substitute(full_cov=full_cov_doc)
    def _get_cov(self, emul_i, emul_s_seq, par_set1, par_set2):
        """
        Calculates the full emulator covariances for requested emulator systems
        `emul_s_seq` at emulator iteration `emul_i` for given parameter sets
        `par_set1` and `par_set2`. The contributions to these covariances
        depend on :attr:`~method`.

        %(full_cov)s

        """

        # Value for fraction of residual variance for variety in passive pars
        weight = [1-len(active_par)/self._modellink._n_par
                  for active_par in self._active_par_data[emul_i]]

        # Determine which residual variance should be used
        if self._method.lower() in ('regression', 'full'):
            rsdl_var = self._rsdl_var[emul_i]
        elif(self.method.lower() == 'gaussian'):
            rsdl_var = [self._sigma**2]*len(emul_s_seq)

        # If cov of sam_set with sam_set is requested (cov_mat)
        if par_set1 is None:
            # Calculate covariance between sam_set and sam_set
            cov = np.zeros([len(emul_s_seq), self._n_sam[emul_i],
                            self._n_sam[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Obtain the difference between sam_set and sam_set
                diff_sam_set = diff(self._sam_set[emul_i], flatten=False)

                # If Gaussian needs to be taken into account
                for i, emul_s in enumerate(emul_s_seq):
                    # Get active_par
                    active_par = self._active_par_data[emul_i][emul_s]

                    # Gaussian variance
                    cov[i] += (1-weight[emul_s])*rsdl_var[emul_s] *\
                        np.exp(-1*np.sum(diff_sam_set[:, :, active_par]**2,
                                         axis=-1) /
                               np.sum(self._l_corr[active_par]**2))

                    # Passive parameter variety
                    cov[i] += weight[emul_s]*rsdl_var[emul_s] *\
                        np.eye(self._n_sam[emul_i])

            if(self._method.lower() in ('regression', 'full') and
               self._use_regr_cov):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, emul_s_seq, None, None)

        # If cov of par_set1 with sam_set is requested (cov_vec)
        elif par_set2 is None:
            # Calculate covariance between par_set1 and sam_set
            cov = np.zeros([len(emul_s_seq), self._n_sam[emul_i]])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Obtain the difference between par_set1 and sam_set
                diff_sam_set = par_set1-self._sam_set[emul_i]

                # If Gaussian needs to be taken into account
                for i, emul_s in enumerate(emul_s_seq):
                    # Get active_par
                    active_par = self._active_par_data[emul_i][emul_s]

                    # Gaussian variance
                    cov[i] += (1-weight[emul_s])*rsdl_var[emul_s] *\
                        np.exp(-1*np.sum(diff_sam_set[:, active_par]**2,
                                         axis=-1) /
                               np.sum(self._l_corr[active_par]**2))

                    # Passive parameter variety
                    cov[i] += weight[emul_s]*rsdl_var[emul_s] *\
                        (par_set1 == self._sam_set[emul_i]).all(axis=-1)

            if(self._method.lower() in ('regression', 'full') and
               self._use_regr_cov):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, emul_s_seq, par_set1, None)

        # If cov of par_set1 with par_set2 is requested (cov)
        else:
            # Calculate covariance between par_set1 and par_set2
            cov = np.zeros([len(emul_s_seq)])

            # Check what 'method' is given
            if self._method.lower() in ('gaussian', 'full'):
                # Obtain the difference between par_set1 and par_set2
                diff_sam_set = par_set1-par_set2

                # If Gaussian needs to be taken into account
                for i, emul_s in enumerate(emul_s_seq):
                    # Get active_par
                    active_par = self._active_par_data[emul_i][emul_s]

                    # Gaussian variance
                    cov[i] += (1-weight[emul_s])*rsdl_var[emul_s] *\
                        np.exp(-1*np.sum(diff_sam_set[active_par]**2,
                                         axis=-1) /
                               np.sum(self._l_corr[active_par]**2))

                    # Passive parameter variety
                    cov[i] += weight[emul_s]*rsdl_var[emul_s] *\
                        (par_set1 == par_set2).all()
            if(self._method.lower() in ('regression', 'full') and
               self._use_regr_cov):
                # If regression needs to be taken into account
                cov += self._get_regr_cov(emul_i, emul_s_seq, par_set1,
                                          par_set2)

        # Return it
        return(cov)

    # This function calculates the regression covariance between parameter sets
    # This is function 'Cov(r(x), r(x'))'
    # OPTIMIZE: Takes roughly 45-50% of total evaluation time
    @docstring_substitute(regr_cov=regr_cov_doc)
    def _get_regr_cov(self, emul_i, emul_s_seq, par_set1, par_set2):
        """
        Calculates the covariances of the regression function for requested
        emulator systems `emul_s_seq` at emulator iteration `emul_i` for given
        parameter sets `par_set1` and `par_set2`.

        %(regr_cov)s

        """

        # If regr_cov of sam_set is requested (cov_mat)
        if par_set1 is None:
            # Make empty array
            regr_cov = np.zeros([len(emul_s_seq), self._n_sam[emul_i],
                                 self._n_sam[emul_i]])

            for i, emul_s in enumerate(emul_s_seq):
                # Initialize PF object
                pf_obj = PF(self._poly_order)

                # Obtain the polynomial terms for both parameter sets
                poly_terms1 = pf_obj.fit_transform(
                    self._sam_set[emul_i][
                        :, self._active_par_data[emul_i][emul_s]])[
                            :, self._poly_idx[emul_i][emul_s]]
                poly_terms2 = poly_terms1

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] =\
                    np.sum(self._poly_coef_cov[emul_i][emul_s]*prod_terms,
                           axis=-1).reshape([self._n_sam[emul_i],
                                             self._n_sam[emul_i]])

        # If regr_cov of par_set with sam_set is requested (cov_vec)
        elif par_set2 is None:
            # Make empty array
            regr_cov = np.zeros([len(emul_s_seq), self._n_sam[emul_i]])

            for i, emul_s in enumerate(emul_s_seq):
                # Initialize PF object
                pf_obj = PF(self._poly_order)

                # Obtain the polynomial terms for both parameter sets
                poly_terms1 =\
                    np.product(pow(
                        par_set1[self._active_par_data[emul_i][emul_s]],
                        self._poly_powers[emul_i][emul_s]), axis=-1)
                poly_terms2 = pf_obj.fit_transform(
                    self._sam_set[emul_i][
                        :, self._active_par_data[emul_i][emul_s]])[
                            :, self._poly_idx[emul_i][emul_s]]

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] =\
                    np.sum(self._poly_coef_cov[emul_i][emul_s]*prod_terms,
                           axis=-1)

        # If regr_cov of par_set1 with par_set2 is requested (cov)
        else:
            # Make empty array
            regr_cov = np.zeros([len(emul_s_seq)])

            for i, emul_s in enumerate(emul_s_seq):
                # Obtain the polynomial terms for both parameter sets
                poly_terms1 =\
                    np.product(pow(
                        par_set1[self._active_par_data[emul_i][emul_s]],
                        self._poly_powers[emul_i][emul_s]), axis=-1)
                poly_terms2 =\
                    np.product(pow(
                        par_set2[self._active_par_data[emul_i][emul_s]],
                        self._poly_powers[emul_i][emul_s]), axis=-1)

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] =\
                    np.sum(self._poly_coef_cov[emul_i][emul_s]*prod_terms,
                           axis=-1)

        # Return it
        return(regr_cov)

    # This function calculates the covariance matrix
    # This is function 'Var(D)' or 'A'
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_cov_matrix(self, emul_i, emul_s_seq):
        """
        Calculates the (inverse) matrix of covariances between known model
        evaluation samples for requested emulator systems `emul_s_seq` at
        emulator iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates
        ---------
        cov_mat : 3D :obj:`~numpy.ndarray` object
            Matrix containing the covariances between all known model
            evaluation samples for requested emulator systems.
        cov_mat_inv : 3D :obj:`~numpy.ndarray` object
            Inverse of covariance matrix for requested emulator systems.

        """

        # Log the creation of the covariance matrix
        logger = getRLogger('COV_MAT')
        logger.info("Calculating covariance matrix for emulator iteration %i."
                    % (emul_i))

        # Calculate covariance matrix
        # Since this calculation can cause memory issues, catch error and try
        # slower but less memory-intensive method
        try:
            cov_mat = self._get_cov(emul_i, emul_s_seq, None, None)
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
            cov_mat = np.zeros([len(emul_s_seq), self._n_sam[emul_i],
                                self._n_sam[emul_i]])
            for i in range(self._n_sam[emul_i]):
                cov_mat[:, i] = self._get_cov(emul_i, emul_s_seq,
                                              self._sam_set[emul_i][i], None)

        # Loop over all emulator systems
        for i, emul_s in enumerate(emul_s_seq):
            # Make sure that cov_mat is symmetric positive-definite by
            # finding the nearest one
            cov_mat[i] = nearest_PD(cov_mat[i])

            # Calculate the inverse of the covariance matrix
            logger.info("Calculating inverse of covariance matrix %i."
                        % (self._emul_s[emul_s]))

            # TODO: Maybe I should put an error catch for memory overflow here?
            cov_mat_inv = self._get_inv_matrix(cov_mat[i])

            # Save the covariance matrix and inverse to hdf5
            self._save_data(emul_i, emul_s, {
                'cov_mat': {
                    'cov_mat': cov_mat[i],
                    'cov_mat_inv': cov_mat_inv}})

        # Log that calculation has been finished
        logger.info("Finished calculating covariance matrix.")

    # This function calculates the inverse of a given matrix
    # TODO: Improve the inverse calculation
    # OPTIMIZE: Use pre-conditioners and linear systems?
    def _get_inv_matrix(self, matrix):
        """
        Calculates the inverse of a given `matrix`.
        Right now only uses the :func:`~numpy.linalg.inv` function.

        Parameters
        ----------
        matrix : 2D array_like
            Matrix to be inverted.

        Returns
        -------
        matrix_inv : 2D :obj:`~numpy.ndarray` object
            Inverse of the given `matrix`.

        """

        # Calculate the inverse of the given matrix
        matrix_inv = inv(matrix)

        # Return it
        return(matrix_inv)

    # Load the emulator
    def _load_emulator(self, modellink_obj):
        """
        Checks if the provided working directory contains a constructed
        emulator and loads in the emulator data accordingly.

        Parameters
        ----------
        modellink_obj : :obj:`~prism.modellink.ModelLink` object
            Instance of the :class:`~prism.modellink.ModelLink` class that
            links the emulated model to this :obj:`~prism.Pipeline` object.

        """

        # Make logger
        logger = getCLogger('EMUL_LOAD')
        logger.info("Loading emulator.")

        # Check if an existing hdf5-file is provided
        try:
            logger.info("Checking if provided working directory %r contains "
                        "a constructed emulator."
                        % (self._pipeline._working_dir))
            with self._File('r', None) as file:
                # Log that an hdf5-file has been found
                logger.info("Provided working directory contains a constructed"
                            " emulator. Checking validity.")

                # Obtain the number of emulator iterations constructed
                emul_i = len(file.keys())

                # Check if the hdf5-file contains solely groups made by PRISM
                req_keys = [str(i) for i in range(1, emul_i+1)]
                if(req_keys != list(file.keys())):
                    err_msg = ("Found master HDF5-file contains invalid data "
                               "groups!")
                    raise_error(err_msg, InputError, logger)

                # Log that valid emulator was found
                logger.info("Found master HDF5-file is valid.")
                self._emul_load = 1

                # Save number of emulator iterations constructed
                self._emul_i = emul_i

        except (OSError, IOError):
            # No existing emulator was provided
            logger.info("Provided working directory does not contain a "
                        "constructed emulator.")
            self._emul_load = 0
            self._emul_i = 0

            # No emulator provided, so no loaded modellink either
            modellink_loaded = None
        else:
            # Read all emulator parameters from the hdf5-file
            modellink_loaded = self._retrieve_parameters()

        # Link the provided ModelLink object to the pipeline
        self._set_modellink(modellink_obj, modellink_loaded)

        # Load emulator data
        self._load_data(self._emul_i)

        # Logging
        logger.info("Finished loading emulator.")

    # This function connects the provided ModelLink class to the pipeline
    def _set_modellink(self, modellink_obj, modellink_loaded):
        """
        Sets the :obj:`~prism.modellink.ModelLink` object that will
        be used for constructing this emulator. If a constructed emulator is
        present, checks if provided `modellink_obj` argument matches the
        :class:`~prism.modellink.ModelLink` subclass used to
        construct it.

        Parameters
        ----------
        modellink_obj : :obj:`~prism.modellink.ModelLink` object
            Instance of the :class:`~prism.modellink.ModelLink` class
            that links the emulated model to this
            :obj:`~prism.Pipeline` object.
            The provided :obj:`~prism.modellink.ModelLink` object
            must match the one used to construct the loaded emulator.
        modellink_loaded : str or None
            If str, the name of the
            :class:`~prism.modellink.ModelLink` subclass that was
            used to construct the loaded emulator.
            If *None*, no emulator is loaded.

        """

        # Logging
        logger = getCLogger('INIT')
        logger.info("Setting ModelLink object.")

        # Try to check the provided modellink_obj
        try:
            # Check if modellink_obj was initialized properly
            if not check_instance(modellink_obj, ModelLink):
                err_msg = ("Provided ModelLink subclass %r was not "
                           "initialized properly!"
                           % (modellink_obj.__class__.__name__))
                raise_error(err_msg, InputError, logger)

        # If this fails, modellink_obj is not an instance of ModelLink
        except TypeError:
            err_msg = ("Input argument 'modellink_obj' must be an instance of "
                       "the ModelLink class!")
            raise_error(err_msg, TypeError, logger)

        # If no existing emulator is loaded, pass
        if modellink_loaded is None:
            logger.info("No constructed emulator is loaded.")
            # Set ModelLink object for Emulator
            self._modellink = modellink_obj

            # Set ModelLink object for Pipeline
            self._pipeline._modellink = self._modellink

        # If an existing emulator is loaded, check if classes are equal
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
            err_msg = ("Provided ModelLink subclass %r does not match the "
                       "ModelLink subclass %r used for emulator construction!"
                       % (modellink_obj._name, modellink_loaded))
            raise_error(err_msg, InputError, logger)

        # Logging
        logger.info("ModelLink object set to %r." % (self._modellink._name))

    # Function that loads in the emulator data
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
        logger = getCLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing emulator data sets.")
        self._n_sam = [[]]
        self._sam_set = [[]]
        self._mod_set = [[]]
        self._active_par_data = [[]]
        self._rsdl_var = [[]]
        self._poly_coef = [[]]
        self._poly_coef_cov = [[]]
        self._poly_powers = [[]]
        self._poly_idx = [[]]
        self._cov_mat_inv = [[]]
        self._exp_dot_term = [[]]
        self._n_data = [[]]
        self._data_val = [[]]
        self._data_err = [[]]
        self._data_spc = [[]]
        self._data_idx = [[]]

        # Initialize emulator system status lists
        self._ccheck = [[]]
        self._active_emul_s = [[]]
        self._n_emul_s = 0

        # Initialize rank specific properties
        if self._is_controller:
            self._n_data_tot = [[]]
            self._n_emul_s_tot = 0
            self._emul_s_to_core = [[] for _ in range(self._size)]
            self._data_idx_to_core = [[]]
            self._active_par = [[]]

        # If no file has been provided
        if(emul_i == 0 or self._emul_load == 0):
            logger.info("Non-existent emulator file provided. No additional "
                        "data needs to be loaded.")
            return

        # Check if requested emulator iteration exists
        elif not(1 <= emul_i <= self._emul_i):
            err_msg = ("Requested emulator iteration %i does not exist!"
                       % (emul_i))
            raise_error(err_msg, RequestError, logger)

        # If both checks succeed, controller determines emul_s assignments
        if self._is_controller:
            # Determine which emulator systems each MPI rank should get
            emul_s_to_core = self._assign_emul_s(emul_i)

            # Save which systems will be assigned to which rank
            self._emul_s_to_core = list(emul_s_to_core)

            # Save total number of emulator systems
            self._n_emul_s_tot = sum([len(seq) for seq in emul_s_to_core])

        # Workers get dummy emul_s_to_core
        else:
            emul_s_to_core = []

        # Assign the emulator systems to the various MPI ranks
        self._emul_s = self._comm.scatter(emul_s_to_core, 0)

        # Temporarily manually swap the CFilter for RFilter
        # Every rank logs what systems were assigned to it
        # TODO: Remove the need to do this manually
        logger.filters = [logger.PRISM_filters['RFilter']]
        logger.info("Received emulator systems %s." % (self._emul_s))
        logger.filters = [logger.PRISM_filters['CFilter']]

        # Determine the number of assigned emulator systems
        self._n_emul_s = len(self._emul_s)

        # Load the corresponding sam_set, mod_set and cov_mat_inv
        logger.info("Loading relevant emulator data up to iteration %i."
                    % (emul_i))

        # Open hdf5-file
        with self._File('r', None) as file:
            # Read in the data
            for i in range(1, emul_i+1):
                group = file['%i' % (i)]

                # Create empty construct check list
                ccheck = []

                # Check if sam_set is available
                try:
                    self._n_sam.append(group.attrs['n_sam'])
                    self._sam_set.append(group['sam_set'][()])
                    self._sam_set[-1].dtype = float
                except KeyError:
                    self._n_sam.append(0)
                    self._sam_set.append([])
                    if self._is_controller:
                        ccheck.append('mod_real_set')

                # Check if active_par is available for the controller
                if self._is_controller:
                    try:
                        par_i = [self._modellink._par_name.index(par.decode(
                            'utf-8')) for par in group.attrs['active_par']]
                        self._active_par.append(np_array(par_i))
                    except KeyError:
                        self._active_par.append([])
                        ccheck.append('active_par')

                # Read in the total number of data points for the controller
                if self._is_controller:
                    self._n_data_tot.append(group.attrs['n_data'])

                # Initialize empty data sets
                mod_set = []
                active_par_data = []
                rsdl_var = []
                poly_coef = []
                poly_coef_cov = []
                poly_powers = []
                poly_idx = []
                cov_mat_inv = []
                exp_dot_term = []
                data_val = []
                data_err = []
                data_spc = []
                data_idx = []

                # Check which emulator systems are active
                self._active_emul_s.append([])
                for j, emul_s in enumerate(self._emul_s):
                    # Create empty construct check list for this system
                    ccheck_s = []

                    # Try to access the emulator system
                    try:
                        data_set = group['emul_%i' % (emul_s)]
                    # If it does not exist, it was passive
                    except KeyError:
                        # Add empty lists for all emulator system data
                        mod_set.append([])
                        active_par_data.append([])
                        rsdl_var.append([])
                        poly_coef.append([])
                        poly_coef_cov.append([])
                        poly_powers.append([])
                        poly_idx.append([])
                        cov_mat_inv.append([])
                        exp_dot_term.append([])
                        data_val.append([])
                        data_err.append([])
                        data_spc.append([])
                        data_idx.append([])
                        ccheck.insert(j, ccheck_s)
                        continue
                    # If it does exist, add emul_s to list of active emul_s
                    else:
                        self._active_emul_s[-1].append(j)

                    # Check if mod_set is available
                    try:
                        mod_set.append(data_set['mod_set'][()])
                    except KeyError:
                        mod_set.append([])

                    # Check if active_par_data is available
                    try:
                        par_i = [self._modellink._par_name.index(
                            par.decode('utf-8')) for par in
                                 data_set.attrs['active_par_data']]
                        active_par_data.append(np_array(par_i))
                    except KeyError:
                        active_par_data.append([])
                        ccheck_s.append('active_par_data')

                    # Check if regression data is available
                    try:
                        rsdl_var.append(data_set.attrs['rsdl_var'])
                        poly_coef.append(data_set['poly_coef'][()])
                        if self._use_regr_cov:
                            poly_coef_cov.append(
                                data_set['poly_coef_cov'][()])
                        else:
                            poly_coef_cov.append([])
                        poly_powers.append(data_set['poly_powers'][()])
                        poly_powers[-1].dtype = int_size
                        poly_idx.append(data_set['poly_idx'][()])
                    except KeyError:
                        rsdl_var.append([])
                        poly_coef.append([])
                        poly_coef_cov.append([])
                        poly_powers.append([])
                        poly_idx.append([])
                        if self._method.lower() in ('regression', 'full'):
                            ccheck_s.append('regression')

                    # Check if cov_mat is available
                    try:
                        cov_mat_inv.append(data_set['cov_mat_inv'][()])
                    except KeyError:
                        cov_mat_inv.append([])
                        ccheck_s.append('cov_mat')

                    # Check if exp_dot_term is available
                    try:
                        exp_dot_term.append(
                            data_set['exp_dot_term'][()])
                    except KeyError:
                        exp_dot_term.append([])
                        ccheck_s.append('exp_dot_term')

                    # Read in data values, errors and spaces
                    data_val.append(data_set.attrs['data_val'])
                    data_err.append(data_set.attrs['data_err'].tolist())
                    data_spc.append(
                        data_set.attrs['data_spc'].decode('utf-8'))

                    # Read in all data_idx parts and combine them
                    data_idx.append(self._read_data_idx(data_set))

                    # Add ccheck_s to ccheck
                    ccheck.insert(j, ccheck_s)

                # Determine the number of data points on this MPI rank
                self._n_data.append(len(self._active_emul_s[-1]))

                # Add all read-in data for this iteration to respective places
                self._mod_set.append(mod_set)
                self._active_par_data.append(active_par_data)
                self._rsdl_var.append(rsdl_var)
                self._poly_coef.append(poly_coef)
                self._poly_coef_cov.append(poly_coef_cov)
                self._poly_powers.append(poly_powers)
                self._poly_idx.append(poly_idx)
                self._cov_mat_inv.append(cov_mat_inv)
                self._exp_dot_term.append(exp_dot_term)
                self._data_val.append(data_val)
                self._data_err.append(data_err)
                self._data_spc.append(data_spc)
                self._data_idx.append(data_idx)

                # Add ccheck for this iteration to global ccheck
                self._ccheck.append(ccheck)

                # If ccheck has no solely empty lists, decrease emul_i by 1
                if(delist(ccheck) != []):
                    self._emul_i -= 1

                # Gather the data_idx from all MPI ranks on the controller
                data_idx_list = self._comm.gather(data_idx, 0)

                # Controller saving the received data_idx_list
                if self._is_controller:
                    self._data_idx_to_core.append(data_idx_list)

        # Log that loading is finished
        logger.info("Finished loading relevant emulator data.")

    # This function saves emulator data to hdf5
    @docstring_substitute(save_data=save_data_doc_e)
    def _save_data(self, emul_i, lemul_s, data_dict):
        """
        Saves a given data dict ``{keyword: data}`` at the given emulator
        iteration `emul_i` and local emulator system `lemul_s` to the HDF5-file
        and as an data attribute to the current :obj:`~Emulator` instance.

        %(save_data)s

        """

        # Do some logging
        logger = getRLogger('SAVE_DATA')

        # If controller keyword contains 'mod_real_set', emul_s must be None
        if((self._is_controller and 'mod_real_set' in data_dict.keys()) or
           lemul_s is None):
            emul_s = None
        # Else, determine what the global emul_s is
        else:
            emul_s = self._emul_s[lemul_s]

        # Open hdf5-file
        with self._File('r+', emul_s) as file:
            # Obtain the dataset this data needs to be saved to
            data_set = file['%i' % (emul_i)]

            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %r data at iteration %i to HDF5."
                            % (keyword, emul_i))

                # Check what data keyword has been provided
                # ACTIVE PARAMETERS
                if(keyword == 'active_par'):
                    # Determine the active parameter names
                    par_names = [self._modellink._par_name[i].encode(
                        'ascii', 'ignore') for i in data]

                    # Save active_par data to file and memory
                    data_set.attrs['active_par'] = par_names
                    self._active_par[emul_i] = data

                    # Remove active_par from ccheck
                    self._ccheck[emul_i].remove('active_par')

                # ACTIVE PARAMETERS DATA
                elif(keyword == 'active_par_data'):
                    # Determine the active parameter names for this system
                    par_names = [self._modellink._par_name[i].encode(
                        'ascii', 'ignore') for i in data]

                    # Save active_par_data data to file and memory
                    data_set.attrs['active_par_data'] = par_names
                    self._active_par_data[emul_i][lemul_s] = data

                    # Remove active_par_data from respective ccheck
                    self._ccheck[emul_i][lemul_s].remove('active_par_data')

                # COV_MAT
                elif(keyword == 'cov_mat'):
                    # Save cov_mat data to file and memory
                    data_set.create_dataset('cov_mat', data=data['cov_mat'])
                    data_set.create_dataset('cov_mat_inv',
                                            data=data['cov_mat_inv'])
                    self._cov_mat_inv[emul_i][lemul_s] = data['cov_mat_inv']

                    # Remove cov_mat from respective ccheck
                    self._ccheck[emul_i][lemul_s].remove('cov_mat')

                # EXP_DOT_TERM
                elif(keyword == 'exp_dot_term'):
                    # Save exp_dot_term data to file and memory
                    data_set.create_dataset('prior_exp_sam_set',
                                            data=data['prior_exp_sam_set'])
                    data_set.create_dataset('exp_dot_term',
                                            data=data['exp_dot_term'])
                    self._exp_dot_term[emul_i][lemul_s] = data['exp_dot_term']

                    # Remove exp_dot_term from respective ccheck
                    self._ccheck[emul_i][lemul_s].remove('exp_dot_term')

                # MOD_REAL_SET (CONTROLLER)
                elif(self._is_controller and keyword == 'mod_real_set'):
                    # Determine dtype-list for compound dataset
                    dtype = [(n, float) for n in self._modellink._par_name]

                    # Convert sam_set to a compound dataset
                    data_c = data['sam_set'].copy()
                    data_c.dtype = dtype

                    # Save sam_set data to file and memory
                    data_set.create_dataset('sam_set', data=data_c)
                    data_set.attrs['n_sam'] = np.shape(data['sam_set'])[0]
                    self._sam_set[emul_i] = data['sam_set']
                    self._n_sam[emul_i] = np.shape(data['sam_set'])[0]

                    # Loop over all received mod_sets
                    for i, lemul_s in enumerate(self._active_emul_s[emul_i]):
                        # Determine the emulator system for this mod_set
                        emul_s = self._emul_s[lemul_s]

                        # Save mod_set data to file and memory
                        data_set_s = data_set['emul_%i' % (emul_s)]
                        data_set_s.create_dataset('mod_set',
                                                  data=data['mod_set'][i])
                        self._mod_set[emul_i][lemul_s] = data['mod_set'][i]

                    # Save whether or not ext_real_set was used
                    data_set.attrs['use_ext_real_set'] =\
                        bool(data['use_ext_real_set'])

                    # Remove mod_real_set from ccheck
                    self._ccheck[emul_i].remove('mod_real_set')

                # MOD_REAL_SET (WORKER)
                elif(self._is_worker and keyword == 'mod_real_set'):
                    # Save mod_set data to file and memory
                    data_set.create_dataset('mod_set', data=data['mod_set'])
                    self._mod_set[emul_i][lemul_s] = data['mod_set']

                # REGRESSION
                elif(keyword == 'regression'):
                    # Determine the new active parameter names for this system
                    par_names = [self._modellink._par_name[i].encode(
                        'ascii', 'ignore') for i in data['active_par']]

                    # Save new active_par_data data to file and memory
                    data_set.attrs['active_par_data'] = par_names
                    self._active_par_data[emul_i][lemul_s] = data['active_par']

                    # Determine dtype-list for compound dataset
                    names = [self._modellink._par_name[par] for par in
                             self._active_par_data[emul_i][lemul_s]]
                    dtype = [(n, int_size) for n in names]

                    # Convert poly_powers to a compound dataset
                    data_c = data['poly_powers'].copy()
                    data_c.dtype = dtype

                    # Save all regression data to file and memory
                    data_set.attrs['rsdl_var'] = data['rsdl_var']
                    data_set.attrs['regr_score'] = data['regr_score']
                    data_set.create_dataset('poly_coef',
                                            data=data['poly_coef'])
                    data_set.create_dataset('poly_powers', data=data_c)
                    data_set.create_dataset('poly_idx', data=data['poly_idx'])
                    self._rsdl_var[emul_i][lemul_s] = data['rsdl_var']
                    self._poly_coef[emul_i][lemul_s] = data['poly_coef']
                    self._poly_powers[emul_i][lemul_s] = data['poly_powers']
                    self._poly_idx[emul_i][lemul_s] = data['poly_idx']
                    if self._use_regr_cov:
                        data_set.create_dataset('poly_coef_cov',
                                                data=data['poly_coef_cov'])
                        self._poly_coef_cov[emul_i][lemul_s] =\
                            data['poly_coef_cov']

                    # Remove regression from respective ccheck
                    self._ccheck[emul_i][lemul_s].remove('regression')

                # INVALID KEYWORD
                else:
                    err_msg = "Invalid keyword argument provided!"
                    raise_error(err_msg, ValueError, logger)

        # More logging
        logger.info("Finished saving data to HDF5.")

    # Read in the emulator attributes
    def _retrieve_parameters(self):
        """
        Reads in the emulator parameters from the provided working directory
        and saves them in the current :obj:`~Emulator` instance.

        """

        # Log that parameters are being read
        logger = getCLogger('INIT')
        logger.info("Retrieving emulator parameters from provided working "
                    "directory.")

        # Open hdf5-file
        with self._File('r', None) as file:
            # Check if provided emulator is the same as requested
            emul_type = file.attrs['emul_type'].decode('utf-8')
            if(emul_type != self._emul_type):
                err_msg = ("Provided emulator system type (%r) does not match "
                           "the requested type (%r)!"
                           % (emul_type, self._emul_type))
                raise_error(err_msg, RequestError, logger)

            # Obtain used PRISM version and check its compatibility
            emul_version = file.attrs['prism_version'].decode('utf-8')
            check_compatibility(emul_version)

            # Read in all the emulator parameters
            self._sigma = file.attrs['sigma']
            self._l_corr = file.attrs['l_corr']
            self._method = file.attrs['method'].decode('utf-8')
            self._use_regr_cov = int(file.attrs['use_regr_cov'])
            self._poly_order = file.attrs['poly_order']
            self._n_cross_val = file.attrs['n_cross_val']
            modellink_name = file.attrs['modellink_name'].decode('utf-8')
            self._use_mock = int(file.attrs['use_mock'])

        # Log that reading is finished
        logger.info("Finished retrieving parameters.")

        # Return the name of the modellink class used to construct the loaded
        # emulator system
        return(modellink_name)

    # This function returns default emulator parameters
    @docstring_append(def_par_doc.format('emulator'))
    def _get_default_parameters(self):
        # Create parameter dict with default parameters
        par_dict = {'sigma': '0.8',
                    'l_corr': '0.3',
                    'method': "'full'",
                    'use_regr_cov': 'False',
                    'poly_order': '3',
                    'n_cross_val': '5',
                    'use_mock': 'False'}

        # Return it
        return(sdict(par_dict))

    # Set the parameters that were read in from the provided parameter file
    @docstring_append(set_par_doc.format("Emulator"))
    def _set_parameters(self):
        # Log that the emulator parameters are being set
        logger = getCLogger('INIT')
        logger.info("Setting emulator parameters.")

        # Obtaining default emulator parameter dict
        par_dict = self._get_default_parameters()

        # Add the read-in prism dict to it
        par_dict.update(self._prism_dict)

        # More logging
        logger.info("Checking compatibility of provided emulator parameters.")

        # GENERAL
        # Gaussian sigma
        self._sigma = check_vals(convert_str_seq(par_dict['sigma'])[0],
                                 'sigma', 'float', 'nzero')

        # Gaussian correlation length
        l_corr = check_vals(convert_str_seq(par_dict['l_corr']), 'l_corr',
                            'float', 'pos', 'normal')
        self._l_corr = l_corr*abs(self._modellink._par_rng[:, 1] -
                                  self._modellink._par_rng[:, 0])

        # Method used to calculate emulator functions
        # Future will include 'gaussian', 'regression', 'auto' and 'full'
        self._method = check_vals(convert_str_seq(par_dict['method'])[0],
                                  'method', 'str')
        if self._method.lower() in ('gaussian', 'regression', 'full'):
            pass
        elif(self._method.lower() == 'auto'):
            raise NotImplementedError
        else:
            err_msg = ("Input argument 'method' is invalid (%r)!"
                       % (self._method.lower()))
            raise_error(err_msg, ValueError, logger)

        # Obtain the bool determining whether or not to use regr_cov
        self._use_regr_cov = check_vals(par_dict['use_regr_cov'],
                                        'use_regr_cov', 'bool')

        # Check if method == 'regression' and set use_regr_cov to True if so
        if(self._method.lower() == 'regression'):
            self._use_regr_cov = 1

        # Obtain the polynomial order for the regression selection process
        self._poly_order =\
            check_vals(convert_str_seq(par_dict['poly_order'])[0],
                       'poly_order', 'int', 'pos')

        # Obtain the number of requested cross-validations
        n_cross_val = check_vals(convert_str_seq(par_dict['n_cross_val'])[0],
                                 'n_cross_val', 'int', 'nneg')

        # Make sure that n_cross_val is not unity
        if(n_cross_val != 1):
            self._n_cross_val = n_cross_val
        else:
            err_msg = ("Input argument 'n_cross_val' cannot be unity!")
            raise_error(err_msg, ValueError, logger)

        # Check whether or not mock data should be used
        # TODO: Allow entire dicts to be given as mock_data (configparser?)
        use_mock = convert_str_seq(par_dict['use_mock'])

        # If use_mock contains a single element, check if it is a bool
        if(len(use_mock) == 1):
            mock_par = None
            use_mock = check_vals(use_mock[0], 'use_mock', 'bool')

        # If not, it must be an array of mock parameter values
        else:
            mock_par = self._modellink._check_sam_set(use_mock, 'use_mock')
            use_mock = True

        # If a currently loaded emulator used mock data and use_mock is False
        # TODO: Becomes obsolete when mock data does not change ModelLink props
        if self._emul_load and self._use_mock and not use_mock:
            # Raise error that ModelLink object needs to be reinitialized
            err_msg = ("Currently loaded emulator uses mock data, while none "
                       "has been requested for new emulator. Reinitialize the "
                       "ModelLink and Pipeline classes to accommodate for this"
                       " change.")
            raise_error(err_msg, RequestError, logger)
        else:
            self._use_mock = use_mock

        # Log that setting has been finished
        logger.info("Finished setting emulator parameters.")

        # Return mock_par
        return(mock_par)

    # This function loads previously generated mock data into ModelLink
    # TODO: Allow user to add/remove mock data? Requires consistency check
    # TODO: Find a way to use mock data without changing ModelLink properties
    def _set_mock_data(self):
        """
        Loads previously used mock data into the
        :class:`~prism.modellink.ModelLink` object, overwriting the
        parameter estimates, data values, data errors, data spaces and data
        identifiers with their mock equivalents.

        Generates
        ---------
        Overwrites the corresponding
        :class:`~prism.modellink.ModelLink` class properties with the
        previously used values (taken from the first emulator iteration).

        """

        # Start logger
        logger = getCLogger('MOCK_DATA')

        # Overwrite ModelLink properties with previously generated values
        # Log that mock_data is being loaded in
        logger.info("Loading previously used mock data into ModelLink object.")

        # Controller only
        if self._is_controller:
            # Open hdf5-file
            with self._File('r', None) as file:
                # Get the number of emulator systems in the first iteration
                group = file['1']
                n_emul_s = group.attrs['n_emul_s']

                # Make empty lists for all model properties
                data_val = []
                data_err = []
                data_spc = []
                data_idx = []

                # Loop over all data points in the first iteration
                for emul_s in range(n_emul_s):
                    # Read in data values, errors and spaces
                    data_set = group['emul_%i' % (emul_s)]
                    data_val.append(data_set.attrs['data_val'])
                    data_err.append(data_set.attrs['data_err'].tolist())
                    data_spc.append(data_set.attrs['data_spc'].decode('utf-8'))

                    # Read in all data_idx parts and combine them
                    data_idx.append(self._read_data_idx(data_set))

                # Overwrite ModelLink properties
                self._modellink._par_est = file.attrs['mock_par'].tolist()
                self._modellink._n_data = group.attrs['n_data']
                self._modellink._data_val = data_val
                self._modellink._data_err = data_err
                self._modellink._data_spc = data_spc
                self._modellink._data_idx = data_idx

        # Broadcast updated modellink properties to workers
        # TODO: Find a cleaner way to write this without bcasting ModelLink obj
        self._modellink._par_est = self._comm.bcast(self._modellink._par_est,
                                                    0)
        self._modellink._n_data = self._comm.bcast(self._modellink._n_data, 0)
        self._modellink._data_val = self._comm.bcast(self._modellink._data_val,
                                                     0)
        self._modellink._data_err = self._comm.bcast(self._modellink._data_err,
                                                     0)
        self._modellink._data_spc = self._comm.bcast(self._modellink._data_spc,
                                                     0)
        self._modellink._data_idx = self._comm.bcast(self._modellink._data_idx,
                                                     0)

        # Log end
        logger.info("Loaded mock data.")
