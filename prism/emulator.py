# -*- coding: utf-8 -*-

"""
Emulator
========
Provides the definition of the class holding the emulator of the *PRISM*
package, the :class:`~Emulator` class.


Available classes
-----------------
:class:`~Emulator`
    Defines the :class:`~Emulator` class of the *PRISM* package.

"""


# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
from collections import Counter
from logging import getLogger
from os import path
from time import time
import sys

# Package imports
from e13tools import InputError
from e13tools.math import diff, nearest_PD
import h5py
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
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
from ._docstrings import (def_par_doc, emul_s_seq_doc, get_emul_i_doc,
                          read_par_doc, save_data_doc_e, std_emul_i_doc)
from ._internal import (PRISM_File, RequestError, check_compatibility,
                        check_val, delist, docstring_append,
                        docstring_substitute, getCLogger, rprint)
from .modellink import ModelLink

# All declaration
__all__ = ['Emulator']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% EMULATOR CLASS DEFINITION
class Emulator(object):
    """
    Defines the :class:`~Emulator` class of the *PRISM* package.

    Description
    -----------
    The :class:`~Emulator` class is the backbone of the *PRISM* package,
    holding all tools necessary to construct, load, save and evaluate the
    emulator of a model. It performs many checks to see if the provided
    :obj:`~ModelLink` object is compatible with the current emulator, advises
    the user on alternatives when certain operations are requested,
    automatically takes care of distributing emulator systems over MPI ranks
    and more.

    Even though the purpose of the :class:`~Emulator` class is to hold only
    information about the emulator and therefore does not require any details
    about the provided :obj:`~ModelLink` object, it will keep track of changes
    made to it. This is to allow the user to modify the properties of the
    :class:`~ModelLink` subclass without causing any desynchronization problems
    by accident.

    The :class:`~Emulator` class requires to be linked to an instance of the
    :class:`~Pipeline` class and will automatically attempt to do so when
    initialized. By default, this class should only be initialized from within
    a :obj:`~Pipeline` object.

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

        # Save the provided Pipeline object
        self._pipeline = pipeline_obj

        # Copy MPI properties to this instance
        self._comm = self._pipeline._comm
        self._size = self._pipeline._size
        self._rank = self._pipeline._rank
        self._is_controller = self._pipeline._is_controller
        self._is_worker = self._pipeline._is_worker

        # Load the emulator and data
        self._load_emulator(modellink_obj)

        # Bind this Emulator instance to the supplied Pipeline object
        self._pipeline._emulator = self

    # %% CLASS PROPERTIES
    # General details
    @property
    def emul_load(self):
        """
        Bool indicating whether or not a previously constructed emulator is
        currently loaded.

        """

        return(bool(self._emul_load))

    @property
    def emul_type(self):
        """
        String indicating what type of emulator is currently loaded.

        """

        return(self._emul_type)

    @property
    def emul_i(self):
        """
        Integer indicating the last/latest available emulator iteration.

        """

        return(self._emul_i)

    @property
    def ccheck(self):
        """
        List of strings indicating which emulator system specific parts are
        still required to complete the construction of the specified emulator
        iteration. The controller rank additionally lists the required parts
        that are emulator iteration specific ('mod_real_set' and 'active_par').

        """

        return(self._ccheck)

    @property
    def n_sam(self):
        """
        Number of model evaluation samples in the specified emulator iteration.

        """

        return(self._n_sam)

    @property
    def n_emul_s(self):
        """
        Number of emulator systems assigned to this MPI rank.

        """

        return(self._n_emul_s)

    @property
    def n_emul_s_tot(self):
        """
        Total number of emulator systems assigned to all MPI ranks combined.
        Only available on the controller rank.

        """

        return(self._n_emul_s_tot)

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
        If `method` == 'regression', this bool is always set to *True*.

        """

        return(bool(self._use_regr_cov))

    @property
    def poly_order(self):
        """
        Polynomial order that is considered for the regression process.
        If `method` == 'gaussian' and :attr:`~Pipeline.do_active_anal` ==
        *False*, this number is not required.

        """

        return(self._poly_order)

    @property
    def active_emul_s(self):
        """
        List containing the indices of the emulator systems on this MPI rank
        that are active in the specified emulator iteration.

        """

        return(self._active_emul_s)

    @property
    def emul_s(self):
        """
        List containing the indices of the emulator systems that are assigned
        to this MPI rank.

        """

        return(self._emul_s)

    @property
    def emul_s_to_core(self):
        """
        List of lists containing the indices of the emulator systems that are
        assigned to every MPI rank. Only available on the controller rank.

        """

        return(self._emul_s_to_core)

    # Active Parameters
    @property
    def active_par(self):
        """
        List containing the model parameter identifiers that are considered
        active in the specified emulator iteration. Only available on the
        controller rank.

        """

        return(self._active_par)

    @property
    def active_par_data(self):
        """
        List containing the model parameter identifiers that are considered
        active in the specified emulator iteration, separated for every data
        point.

        """

        return(self._active_par_data)

    # Regression
    @property
    def rsdl_var(self):
        """
        List with residual variances for every data point in the specified
        emulator iteration.
        Obtained from regression process and replaces the Gaussian sigma.

        """

        return(self._rsdl_var)

    @property
    def poly_coef(self):
        """
        List with non-zero coefficients for the polynomial terms in the
        regression function in the specified emulator iteration, separated per
        data point.

        """

        return(self._poly_coef)

    @property
    def poly_coef_cov(self):
        """
        List with covariances for all polynomial coefficients in the
        regression function in the specified emulator iteration, separated per
        data point.

        """

        return(self._poly_coef_cov)

    @property
    def poly_powers(self):
        """
        List containing the polynomial term powers in the specified emulator
        iteration, separated per data point.

        """

        return(self._poly_powers)

    @property
    def poly_idx(self):
        """
        List containing the indices of the polynomial terms with non-zero
        coefficients in the specified emulator iteration, separated per data
        point.

        """

        return(self._poly_idx)

    # Emulator Data
    @property
    def sam_set(self):
        """
        Array containing all model evaluation samples in the specified emulator
        iteration.

        """

        return(self._sam_set)

    @property
    def mod_set(self):
        """
        Array containing all model outputs in the specified emulator iteration.

        """

        return(self._mod_set)

    @property
    def cov_mat_inv(self):
        """
        Array containing the inverses of the covariance matrices in the
        specified emulator iteration, separated per data point.

        """

        return(self._cov_mat_inv)

    @property
    def exp_dot_term(self):
        """
        Array containing the second expectation adjustment dot-term values of
        all model evaluation samples in the specified emulator iteration.

        """

        return(self._exp_dot_term)

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
    def _get_emul_i(self, emul_i, cur_iter):
        """
        Checks if the provided emulator iteration `emul_i` can be requested or
        replaces it if *None* was provided.

        Parameters
        ----------
        %(emul_i)s
        cur_iter : bool
            Bool determining whether the current or the next emulator iteration
            is requested.

        Returns
        -------
        emul_i : int
            The requested emulator iteration that passed the check.

        """

        # Log that emul_i is being selected
        logger = getCLogger('INIT')
        logger.info("Selecting emulator iteration for user-method.")

        # Determine the emul_i that is constructed on all ranks
        global_emul_i = min(self._comm.allgather(self._emul_i))

        # Check if provided emul_i is correct/allowed
        if cur_iter:
            if(emul_i == 0 or self._emul_load == 0 or global_emul_i == 0):
                raise RequestError("Emulator HDF5-file is not built yet!")
            elif emul_i is None:
                emul_i = global_emul_i
            elif not(1 <= emul_i <= global_emul_i):
                logger.error("Requested emulator iteration %s does not exist!"
                             % (emul_i))
                raise RequestError("Requested emulator iteration %s does not "
                                   "exist!" % (emul_i))
            else:
                emul_i = check_val(emul_i, 'emul_i', 'pos', 'int')
        else:
            if emul_i is None:
                emul_i = global_emul_i+1
            elif not(1 <= emul_i <= global_emul_i+1):
                logger.error("Requested emulator iteration %s cannot be "
                             "requested!" % (emul_i))
                raise RequestError("Requested emulator iteration %s cannot be "
                                   "requested!" % (emul_i))
            else:
                emul_i = check_val(emul_i, 'emul_i', 'pos', 'int')

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
        logger = getCLogger('INIT')
        logger.info("Creating a new emulator system in HDF5-file '%s'."
                    % (self._pipeline._hdf5_file))

        # If no constructed emulator was provided, it will be constructed now
        # Therefore, set emul_load to 1
        self._emul_load = 1

        # Clean-up all emulator system files
        self._cleanup_emul_files(1)

        # Read in parameters from provided parameter file
        self._read_parameters()

        # Controller only
        if self._is_controller:
            # Create hdf5-file
            with PRISM_File('w', None) as file:
                # Save all relevant emulator parameters to hdf5
                file.attrs['sigma'] = self._sigma
                file.attrs['l_corr'] = self._l_corr
                file.attrs['method'] = self._method.encode('ascii', 'ignore')
                file.attrs['use_regr_cov'] = bool(self._use_regr_cov)
                file.attrs['poly_order'] = self._poly_order
                file.attrs['modellink_name'] =\
                    self._modellink._name.encode('ascii', 'ignore')
                file.attrs['prism_version'] =\
                    _prism_version.encode('ascii', 'ignore')
                file.attrs['emul_type'] = self._emul_type.encode('ascii',
                                                                 'ignore')
                file.attrs['use_mock'] = bool(self._use_mock)

        # Check if mock data is requested
        if self._use_mock:
            # If so, temporarily save ModelLInk properties as Emulator props
            # This is to make sure that one version of get_md_var() is required
            self._n_data[0] = self._modellink._n_data
            self._data_val[0] = self._modellink._data_val
            self._data_err[0] = self._modellink._data_err
            self._data_spc[0] = self._modellink._data_spc
            self._data_idx[0] = self._modellink._data_idx

            # Call get_mock_data()
            self._pipeline._get_mock_data()

            # Controller only
            if self._is_controller:
                # Open hdf5
                with PRISM_File('r+', None) as file:
                    # Save mock_data to hdf5
                    file.attrs['mock_par'] = self._modellink._par_est

        # Load relevant data
        self._load_data(0)

        # Prepare first emulator iteration to be constructed
        self._prepare_new_iteration(1)

        # Logging again
        logger.info("Finished creating new emulator system.")

    # This function cleans up all the emulator files
    # TODO: Also delete all projection figures?
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _cleanup_emul_files(self, emul_i):
        """
        Opens all emulator HDF5-files and removes the provided emulator
        iteration `emul_i` and subsequent iterations from the files.

        Parameters
        ----------
        %(emul_i)s

        """

        # Do some logging
        logger = getCLogger('CLEAN-UP')
        logger.info("Cleaning up emulator HDF5-files, starting at emulator"
                    " iteration %s." % (emul_i))

        # Controller only
        if self._is_controller:
            # Check what the maximum number of emulator systems is
            try:
                n_emuls = max(self._n_data_tot[1:])
            except ValueError:
                n_emuls = 0

            # Loop over all emulator system files
            for s in range(0, n_emuls):
                # Open emulator system HDF5-file
                with PRISM_File('r+', s) as file:
                    # Loop over all requested iterations to be removed
                    for i in range(emul_i, self._emul_i+2):
                        # Try to remove it, skip if not possible
                        try:
                            del file['%s' % (i)]
                        except KeyError:
                            pass

            # Open emulator master HDF5-file if it exists
            if n_emuls:
                with PRISM_File('r+', None) as file:
                    # Loop over all requested iterations to be removed
                    for i in range(emul_i, self._emul_i+2):
                        # Try to remove it, skip if not possible
                        try:
                            del file['%s' % (i)]
                        except KeyError:
                            pass

        # MPI Barrier
        self._comm.Barrier()

        # Set emul_i to the last iteration still present in files
        self._emul_i = emul_i-1

        # Do more logging
        logger.info("Finished cleaning up emulator HDF5-files.")

    # This function matches data points with those in a previous iteration
    # TODO: Give every unique data_idx its own emul_s?
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _assign_data_idx(self, emul_i):
        """
        Determines the emulator system each data point in the provided emulator
        iteration `emul_i` should be assigned to, in order to make sure that
        recurring data points have the same emulator system number as in the
        previous emulator iteration. If multiple options are possible, data
        points are assigned such to spread them as much as possible.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        data_to_emul_s : list of int
            The number of the emulator system that each data point should be
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
                    "systems for emulator iteration %s." % (emul_i))

        # Create empty Counter for number of emulator system occurances
        emul_s_counter = Counter()

        # Calculate the total number of active and passive emulator systems
        if self._n_emul_s_tot:
            n_emul_s = max(self._modellink._n_data, self._n_emul_s_tot)
        else:
            n_emul_s = self._modellink._n_data

        active_emul_s_list = [[]]
        data_idx_list = []

        # Open hdf5-file
        with PRISM_File('r', None) as file:
            for i in range(1, emul_i):
                active_emul_s_list.append(
                        [int(key[5:]) for key in file['%s' % (i)].keys() if
                         key[:5] == 'emul_'])

            for emul_s in active_emul_s_list[-1]:
                data_set = file['%s/emul_%s' % (emul_i-1, emul_s)]
                # Read in all data_idx parts and combine them
                idx_keys = [key for key in data_set.attrs.keys()
                            if key[:8] == 'data_idx']
                idx_len = len(idx_keys)
                if(idx_len == 1):
                    if isinstance(data_set.attrs['data_idx'], bytes):
                        data_idx_list.append(
                            data_set.attrs['data_idx'].decode('utf-8'))
                    else:
                        data_idx_list.append(data_set.attrs['data_idx'])
                else:
                    tmp_data_idx = []
                    for key in idx_keys:
                        if isinstance(data_set.attrs[key], bytes):
                            idx_str = data_set.attrs[key].decode('utf-8')
                            tmp_data_idx.append(idx_str)
                        else:
                            tmp_data_idx.append(data_set.attrs[key])
                    data_idx_list.append(tmp_data_idx)

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
                emul_s = data_idx_list.index(data_idx)
            except ValueError:
                pass
            else:
                # If it existed, assign data_idx to corresponding emul_s
                data_to_emul_s[i] = emul_s

                # Also remove emul_s from emul_s_counter
                emul_s_counter.pop(emul_s)

        # Assign all 'new' data points to the non-filled emulator systems
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
        rank in order to balance the number of active emulator systems on every
        rank for every iteration up to the provided emulator iteration
        `emul_i`. If multiple choices can achieve this, the emulator systems
        are automatically spread out such that the total number of active
        emulator systems on a single rank is also balanced as much as possible.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        emul_s : list of int
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
                    "iteration %s for available MPI ranks." % (emul_i))

        # Create empty list of active emulator systems
        active_emul_s_list = [[]]

        # Open hdf5-file
        with PRISM_File('r', None) as file:
            logger.info("Determining active emulator systems in every "
                        "emulator iteration.")

            # Determine the active emulator systems in every iteration
            for i in range(1, emul_i+1):
                active_emul_s_list.append(
                    [int(key[5:]) for key in file['%s' % (i)].keys() if
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
        logger = getCLogger('EMUL_PREP')
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
            raise RequestError("Preparation of emulator iteration %s is "
                               "only available when all previous "
                               "iterations exist!" % (emul_i))
        elif(emul_i-1 == self._emul_i):
            # Set reload flag to 0
            reload = 0
        else:
            logger.info("Emulator iteration %s already exists." % (emul_i))

            # Check if repreparation was actually necessary
            # TODO: Think about how to extend this check
            diff_flag = 1
            for i in range(self._n_data[emul_i]):
                if not(self._data_val[emul_i][i] in
                       self._modellink._data_val):
                    break
                if not(self._data_err[emul_i][i] in
                       self._modellink._data_err):
                    break
                if not(self._data_spc[emul_i][i] in
                       self._modellink._data_spc):
                    break
                if not(self._data_idx[emul_i][i] in
                       self._modellink._data_idx):
                    break
            # If not, set diff_flag to 0
            else:
                diff_flag = 0

            # Gather the diff_flags on the controller
            diff_flag = np.any(self._comm.gather(diff_flag, 0))

            # If all diff_flags were 0, give out a warning
            if self._is_controller and not diff_flag:
                logger.warning("No differences in model comparison data "
                               "detected.\nUnless this repreparation was "
                               "intentional, using the 'analyze' method of"
                               " the Pipeline class is much faster for "
                               "reanalyzing the emulator with new pipeline"
                               " parameters.")
                print("No differences in model comparison data "
                      "detected.\nUnless this repreparation was "
                      "intentional, using the 'analyze' method of "
                      "the Pipeline class is much faster for "
                      "reanalyzing the emulator with new pipeline "
                      "parameters.")

            # Set reload flag to 1
            reload = 1

        # Clean-up all emulator files if emul_i is not 1
        if(emul_i != 1):
            self._cleanup_emul_files(emul_i)

        # Controller preparing the emulator iteration
        if self._is_controller:
            # Open hdf5-file
            with PRISM_File('r+', None) as file:
                # Make group for emulator iteration
                group = file.create_group('%s' % (emul_i))

                # Save the number of data points
                group.attrs['n_data'] = self._modellink._n_data

                # Create an empty data set for statistics as attributes
                group.create_dataset('statistics', data=h5py.Empty(float))

                # Assign data points to emulator systems
                data_to_emul_s, n_emul_s = self._assign_data_idx(emul_i)

                # Save the total number of active and passive emulator systems
                group.attrs['n_emul_s'] = n_emul_s

                # Create groups for all data points
                for i, emul_s in enumerate(data_to_emul_s):
                    with PRISM_File('a', emul_s) as file_i:
                        # Make iteration group for this emulator system
                        data_set = file_i.create_group('%s' % (emul_i))

                        # Save data value, errors and space to this system
                        data_set.attrs['data_val'] =\
                            self._modellink._data_val[i]
                        data_set.attrs['data_err'] =\
                            self._modellink._data_err[i]
                        data_set.attrs['data_spc'] =\
                            self._modellink._data_spc[i].encode('ascii',
                                                                'ignore')

                        # Save data_idx in portions to make it HDF5-compatible
                        if isinstance(self._modellink._data_idx[i], list):
                            for j, idx in enumerate(
                                    self._modellink._data_idx[i]):
                                if isinstance(idx, (str, unicode)):
                                    data_set.attrs['data_idx_%s' % (j)] =\
                                        idx.encode('ascii', 'ignore')
                                else:
                                    data_set.attrs['data_idx_%s' % (j)] = idx
                        else:
                            if isinstance(self._modellink._data_idx[i],
                                          (str, unicode)):
                                data_set.attrs['data_idx'] =\
                                    self._modellink._data_idx[i].encode(
                                        'ascii', 'ignore')
                            else:
                                data_set.attrs['data_idx'] =\
                                    self._modellink._data_idx[i]

                        # Create external link between file_i and master file
                        group['emul_%s' % (emul_s)] = h5py.ExternalLink(
                            path.basename(file_i.filename), '%s' % (emul_i))

        # MPI Barrier
        self._comm.Barrier()

        # All ranks reload the emulator systems to allow for reassignments
        self._emul_i = emul_i
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
        `emul_i`, by performing the given emulator method and calculating the
        prior expectation and variance values of the model evaluation samples.

        Parameters
        ----------
        %(emul_i)s

        Generates
        ---------
        All data sets that are required to evaluate the emulator system.

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
        ccheck_cov_mat = [emul_s for emul_s in emul_s_seq if
                          'cov_mat' in self._ccheck[emul_i][emul_s]]
        if len(ccheck_cov_mat):
            self._get_cov_matrix(emul_i, ccheck_cov_mat)

        # Calculate the prior expectation values of sam_set
        ccheck_pre_calc_term = [emul_s for emul_s in emul_s_seq if
                                'exp_dot_term' in
                                self._ccheck[emul_i][emul_s]]
        if len(ccheck_pre_calc_term):
            self._get_exp_dot_term(emul_i, ccheck_pre_calc_term)

        # If a worker is finished, set current emul_i to constructed emul_i
        if self._is_worker:
            self._emul_i = emul_i

        # MPI Barrier
        self._comm.Barrier()

        # If everything is done, gather the total set of active parameters
        active_par_data = self._comm.gather(self._active_par_data[emul_i], 0)

        # Allow the controller to save them
        if self._is_controller and 'active_par' in self._ccheck[emul_i]:
            active_par = SortedSet()
            for active_par_rank in active_par_data:
                active_par.update(*active_par_rank)
            self._save_data(emul_i, None, {
                'active_par': np.array(list(active_par))})

            # Set current emul_i to constructed emul_i
            self._emul_i = emul_i

            # Save time difference and communicator size
            self._pipeline._save_statistics(emul_i, {
                'emul_construct_time': ['%.2f' % (time()-start_time), 's'],
                'MPI_comm_size_cons': ['%s' % (self._size), '']})

        # MPI Barrier
        self._comm.Barrier()

    # This is function 'E_D(f(x'))'
    # This function gives the adjusted emulator expectation value back
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_adj_exp(self, emul_i, emul_s_seq, par_set, cov_vec):
        """
        Calculates the adjusted emulator expectation value at a given emulator
        iteration `emul_i` for specified parameter set `par_set` and
        corresponding covariance vector `cov_vec`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
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
        prior_exp_par_set = self._get_prior_exp(emul_i, emul_s_seq, par_set)

        # Create empty adj_exp_val
        adj_exp_val = np.zeros(len(emul_s_seq))

        # Calculate the adjusted emulator expectation value at given par_set
        for i, emul_s in enumerate(emul_s_seq):
            adj_exp_val[i] = prior_exp_par_set[i] +\
                np.dot(cov_vec[i].T, self._exp_dot_term[emul_i][emul_s])

        # Return it
        return(adj_exp_val)

    # This is function 'Var_D(f(x'))'
    # This function gives the adjusted emulator variance value back
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_adj_var(self, emul_i, emul_s_seq, par_set, cov_vec):
        """
        Calculates the adjusted emulator variance value at a given emulator
        iteration `emul_i` for specified parameter set `par_set` and
        corresponding covariance vector `cov_vec`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
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
        prior_var_par_set = self._get_prior_var(emul_i, emul_s_seq, par_set)

        # Create empty adj_var_val
        adj_var_val = np.zeros(len(emul_s_seq))

        # Calculate the adjusted emulator variance value at given par_set
        for i, emul_s in enumerate(emul_s_seq):
            adj_var_val[i] = prior_var_par_set[i] -\
                np.dot(cov_vec[i].T,
                       np.dot(self._cov_mat_inv[emul_i][emul_s], cov_vec[i]))

        # Return it
        return(adj_var_val)

    # This function evaluates the emulator at a given emul_i and par_set and
    # returns the adjusted expectation and variance values
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _evaluate(self, emul_i, emul_s_seq, par_set):
        """
        Evaluates the emulator system at iteration `emul_i` for given
        `par_set`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
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
        cov_vec = self._get_cov_vector(emul_i, emul_s_seq, par_set)

        # Calculate the adjusted expectation and variance values
        adj_exp_val = self._get_adj_exp(emul_i, emul_s_seq, par_set, cov_vec)
        adj_var_val = self._get_adj_var(emul_i, emul_s_seq, par_set, cov_vec)

        # Make sure that adj_var_val cannot drop below zero
        for i in range(len(emul_s_seq)):
            if(adj_var_val[i] < 0):
                adj_var_val[i] = 0.0

        # Return adj_exp_val and adj_var_val
        return(adj_exp_val, adj_var_val)

    # This function extracts the set of active parameters
    # TODO: Write code cleaner, if possible
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_active_par(self, emul_i, emul_s_seq):
        """
        Determines the active parameters to be used for every individual data
        point defined in `emul_s_seq` in the provided emulator iteration
        `emul_i`. Uses backwards stepwise elimination to determine the set of
        active parameters.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates
        ---------
        active_par_data : List of 1D :obj:`~numpy.ndarray` objects
            List containing the indices of all the parameters that are active
            in the emulator iteration `emul_i` for every individual data point.

        """

        # Log that active parameters are being determined
        logger = getLogger('ACTIVE_PAR')
        logger.info("Determining active parameters.")

        # Loop over all data points and determine active parameters
        for emul_s in emul_s_seq:
            # Initialize active parameters data set
            active_par_data = SortedSet()

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
                # Obtain frozen+potentially active parameters
                frz_pot_act_par = SortedSet(active_par_data)
                frz_pot_act_par.update(self._pipeline._pot_active_par)
                frz_pot_act_par = list(frz_pot_act_par)
                frz_pot_act_idx = list(range(len(frz_pot_act_par)))

                # Obtain non-frozen potentially active parameters
                non_frz_par = [j for j in self._pipeline._pot_active_par
                               if j not in active_par_data]
                non_frz_idx = [frz_pot_act_par.index(j) for j in non_frz_par]

                # Obtain sam_set of frz_pot_act_par
                frz_pot_act_sam_set = self._sam_set[emul_i][:, frz_pot_act_par]

                # Obtain polynomial terms of frz_pot_act_sam_set
                pf_obj = PF(self._poly_order, include_bias=False)
                frz_pot_act_poly_terms =\
                    pf_obj.fit_transform(frz_pot_act_sam_set)

                # Create SequentialFeatureSelector object
                sfs_obj = SFS(LR(), k_features='parsimonious', forward=False,
                              floating=False, scoring='r2',
                              cv=min(5, self._n_sam[emul_i]))

                # Perform linear regression with linear terms only
                sfs_obj.fit(frz_pot_act_sam_set, self._mod_set[emul_i][emul_s])

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
                    sfs_obj.fit(poly_terms, self._mod_set[emul_i][emul_s])

                    # Extract indices of active polynomial terms
                    act_idx_poly = poly_idx[list(sfs_obj.k_feature_idx_)]

                    # Check if any additional polynomial terms survived
                    # Add i to act_idx if this is the case
                    if np.any([k not in act_idx_lin for k in act_idx_poly]):
                        act_idx.append(j)

                # Update the active parameters for this data set
                active_par_data.update(np.array(frz_pot_act_par)[act_idx])

            # Log the resulting active parameters
            logger.info("Active parameters for emulator system %s: %s"
                        % (self._emul_s[emul_s],
                           [self._modellink._par_name[par]
                            for par in active_par_data]))

            # Convert active_par_data to a NumPy array and save
            self._save_data(emul_i, emul_s, {
                'active_par_data': np.array(list(active_par_data))})

        # Log that active parameter determination is finished
        logger.info("Finished determining active parameters.")

    # This function performs a forward stepwise regression on sam_set
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _do_regression(self, emul_i, emul_s_seq):
        """
        Performs a forward stepwise linear regression on all model evaluation
        samples in emulator iteration `emul_i`. Calculates what the expectation
        values of all polynomial coefficients are. The polynomial order that is
        used in the regression depends on the `poly_order` parameter provided
        during class initialization.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

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
        poly_coef_cov : 2D list (if :attr:`~Emulator.use_regr_cov` == *True*)
            List containing the covariance values of the polynomial
            coefficients for all data points.

        """

        # Create logger
        logger = getLogger('REGRESSION')
        logger.info("Performing regression.")

        # Create SequentialFeatureSelector object
        sfs_obj = SFS(LR(), k_features='best', forward=True, floating=False,
                      scoring='neg_mean_squared_error',
                      cv=min(5, self._n_sam[emul_i]))

        # Create Pipeline object
        # The bias/intercept/constant-term is not included in the SFS object to
        # ensure that it is taken into account in the linear regression, since
        # it is required for getting the residual variance. It also ensures
        # that the SFS does not focus on the constant-term in its calculations.
        pipe = Pipeline_sk([('poly', PF(self._poly_order, include_bias=False)),
                            ('SFS', sfs_obj),
                            ('linear', LR())])

        # Loop over all data points and perform a regression on all of them
        # TODO: Redetermine active parameters after regression process
        for emul_s in emul_s_seq:
            # Extract active_sam_set
            active_sam_set = self._sam_set[emul_i][
                    :, self._active_par_data[emul_i][emul_s]]

            # Perform regression on this data point
            pipe.fit(active_sam_set, self._mod_set[emul_i][emul_s])

            # Obtain the corresponding polynomial indices
            poly_idx_temp = np.array(pipe.named_steps['SFS'].k_feature_idx_)

            # Extract sam_set_poly
            sam_set_poly = pipe.named_steps['poly'].transform(
                active_sam_set)[:, poly_idx_temp]

            # Extract the residual variance
            rsdl_var = mse(self._mod_set[emul_i][emul_s],
                           pipe.named_steps['linear'].predict(sam_set_poly))

            # Log the score of the regression process
            regr_score = pipe.named_steps['linear'].score(
                    sam_set_poly, self._mod_set[emul_i][emul_s])
            logger.info("Regression score for emulator system %s: %s."
                        % (self._emul_s[emul_s], regr_score))

            # Add the intercept term to sam_set_poly
            sam_set_poly = np.concatenate([np.ones([self._n_sam[emul_i], 1]),
                                           sam_set_poly], axis=-1)

            # Calculate the poly_coef covariances
            if self._use_regr_cov:
                poly_coef_cov = rsdl_var*inv(
                    np.dot(sam_set_poly.T, sam_set_poly)).flatten()

            # Obtain polynomial powers and include intercept term
            poly_powers_temp = pipe.named_steps['poly'].powers_[poly_idx_temp]
            poly_powers_intercept =\
                [[0]*len(self._active_par_data[emul_i][emul_s])]
            poly_powers = np.concatenate([poly_powers_intercept,
                                          poly_powers_temp])

            # Add intercept term to polynomial indices
            poly_idx_temp += 1
            poly_idx = np.concatenate([[0], poly_idx_temp])

            # Obtain polynomial coefficients and include intercept term
            poly_coef_temp = pipe.named_steps['linear'].coef_
            poly_coef_intercept = [pipe.named_steps['linear'].intercept_]
            poly_coef = np.concatenate([poly_coef_intercept,
                                        poly_coef_temp])

            # Save everything to hdf5
            if self._use_regr_cov:
                self._save_data(emul_i, emul_s, {
                    'regression': [rsdl_var, regr_score, poly_coef,
                                   poly_powers, poly_idx, poly_coef_cov]})
            else:
                self._save_data(emul_i, emul_s, {
                    'regression': [rsdl_var, regr_score, poly_coef,
                                   poly_powers, poly_idx]})

        # Log that this is finished
        logger.info("Finished performing regression.")

    # This function gives the prior expectation value
    # This is function 'E(f(x'))' or 'u(x')'
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_prior_exp(self, emul_i, emul_s_seq, par_set):
        """
        Calculates the prior expectation value at a given emulator iteration
        `emul_i` for specified parameter set `par_set`. This expectation
        depends on the emulator method used.

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
            Prior expectation values for either sam_set or `par_set` for every
            data point.

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
                    poly_terms = np.product(pow(par_set[
                        self._active_par_data[emul_i][emul_s]],
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
        Pre-calculates the second expectation adjustment dot-term at a given
        emulator iteration `emul_i` for all model evaluation samples and saves
        it for later use.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates
        ---------
        exp_dot_term : 2D :obj:`~numpy.ndarray` object
            2D array containing the pre-calculated values for the second
            adjustment dot-term of the adjusted expectation for all model
            evaluation samples for all data points. Has the same shape as the
            array with model outputs.

        """

        # Create logger
        logger = getLogger('DOT_TERM')
        logger.info("Pre-calculating second expectation adjustment dot-term "
                    "for known samples at emulator iteration %s." % (emul_i))

        # Obtain prior expectation value of sam_set
        prior_exp_sam_set = self._get_prior_exp(emul_i, emul_s_seq, None)

        # Calculate the exp_dot_term values and save it to hdf5
        for i, emul_s in enumerate(emul_s_seq):
            exp_dot_term = np.dot(self._cov_mat_inv[emul_i][emul_s],
                                  (self._mod_set[emul_i][emul_s] -
                                   prior_exp_sam_set[i]))
            self._save_data(emul_i, emul_s, {
                'exp_dot_term': [prior_exp_sam_set[i], exp_dot_term]})

        # Log again
        logger.info("Finished pre-calculating second adjustment dot-term "
                    "values.")

    # This function gives the prior variance value
    # This is function 'Var(f(x'))'
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_prior_var(self, emul_i, emul_s_seq, par_set):
        """
        Calculates the prior variance value at a given emulator iteration
        `emul_i` for specified parameter set `par_set`. This variance depends
        on the emulator method used.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the prior variance for.

        Returns
        -------
        prior_var : 1D :obj:`~numpy.ndarray` object
            Prior variance value for every data point.

        """

        # Return it
        return(self._get_cov(emul_i, emul_s_seq, par_set, par_set))

    # This function calculates the covariance between parameter sets
    # This is function 'Cov(f(x), f(x'))' or 'k(x,x')
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_cov(self, emul_i, emul_s_seq, par_set1, par_set2):
        """
        Calculates the full emulator covariances at emulator iteration `emul_i`
        for given parameter sets `par_set1` and `par_set2`. The contributions
        to these covariances depend on the emulator method used.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
        par_set1, par_set2 : 1D :obj:`~numpy.ndarray` object or None
            If `par_set1` and `par_set2` are both not *None*, calculate
            covariances for `par_set1` with `par_set2`.
            If `par_set1` is not *None* and `par_set2` is *None*, calculate
            covariances for `par_set1` with :attr:`~Emulator.sam_set`
            (covariance vector).
            If `par_set1` and `par_set2` are both *None*, calculate covariances
            for :attr:`~Emulator.sam_set` (covariance matrix).
            When not *None*, `par_set` is the model parameter value set to
            calculate the covariances for.

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
        elif(self.method.lower() == 'gaussian'):
            rsdl_var = [pow(self._sigma, 2) for _ in emul_s_seq]

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
                        np.exp(-1*pow(norm(diff_sam_set[:, :, active_par],
                                           axis=-1), 2) /
                               pow(norm(self._l_corr[active_par]), 2))

                    # Inactive parameter variety
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
                        np.exp(-1*pow(norm(diff_sam_set[:, active_par],
                                           axis=-1), 2) /
                               pow(norm(self._l_corr[active_par]), 2))

                    # Inactive parameter variety
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
                        np.exp(-1*pow(norm(diff_sam_set[active_par],
                                           axis=-1), 2) /
                               pow(norm(self._l_corr[active_par]), 2))

                    # Inactive parameter variety
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
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_regr_cov(self, emul_i, emul_s_seq, par_set1, par_set2):
        """
        Calculates the covariances of the regression function at emulator
        iteration `emul_i` for given parameter sets `par_set1` and `par_set2`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
        par_set1, par_set2 : 1D :obj:`~numpy.ndarray` object or None
            If `par_set1` and `par_set2` are both not *None*, calculate
            regression covariances for `par_set1` with `par_set2`.
            If `par_set1` is not *None* and `par_set2` is *None*, calculate
            regression covariances for `par_set1` with
            :attr:`~Emulator.sam_set` (covariance vector).
            If `par_set1` and `par_set2` are both *None*, calculate regression
            covariances for :attr:`~Emulator.sam_set`(covariance matrix).
            When not *None*, `par_set` is the model parameter value set to
            calculate the regression covariances for.

        Returns
        -------
        regr_cov : 1D, 2D or 3D :obj:`~numpy.ndarray` object
            Depending on the arguments provided, a regression covariance
            value, vector or matrix for all data points.

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
                    np.product(pow(par_set1[
                        self._active_par_data[emul_i][emul_s]],
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
                    np.product(pow(par_set1[
                        self._active_par_data[emul_i][emul_s]],
                        self._poly_powers[emul_i][emul_s]), axis=-1)
                poly_terms2 =\
                    np.product(pow(par_set2[
                        self._active_par_data[emul_i][emul_s]],
                        self._poly_powers[emul_i][emul_s]), axis=-1)

                # Obtain the combined product polynomial terms
                prod_terms = np.kron(poly_terms1, poly_terms2)

                # Calculate the regression covariance
                regr_cov[i] =\
                    np.sum(self._poly_coef_cov[emul_i][emul_s]*prod_terms,
                           axis=-1)

        # Return it
        return(regr_cov)

    # This function calculates the covariance vector of a given parameter set
    # This is function 'Cov(f(x'), D)' or 't(x')'
    # HINT: Calculate cov_vec for all samples at once to save time?
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_cov_vector(self, emul_i, emul_s_seq, par_set):
        """
        Calculates the column vector of covariances between given (`par_set`)
        and known (:attr:`~Emulator.sam_set`) model parameter value sets for a
        given emulator iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
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
        cov_vec = self._get_cov(emul_i, emul_s_seq, par_set, None)

        # Return it
        return(cov_vec)

    # This function calculates the covariance matrix
    # This is function 'Var(D)' or 'A'
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_cov_matrix(self, emul_i, emul_s_seq):
        """
        Calculates the (inverse) matrix of covariances between known model
        evaluation samples for a given emulator iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Generates
        ---------
        cov_mat : 3D :obj:`~numpy.ndarray` object
            Matrix containing the covariances between all known model
            evaluation samples for every data point.
        cov_mat_inv : 3D :obj:`~numpy.ndarray` object
            Inverse of covariance matrix for every data point.

        """

        # Log the creation of the covariance matrix
        logger = getLogger('COV_MAT')
        logger.info("Calculating covariance matrix for emulator iteration %s."
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

        # Loop over all data points
        for i, emul_s in enumerate(emul_s_seq):
            # Make sure that cov_mat is symmetric positive-definite by
            # finding the nearest one
            cov_mat[i] = nearest_PD(cov_mat[i])

            # Calculate the inverse of the covariance matrix
            logger.info("Calculating inverse of covariance matrix %s."
                        % (emul_s))

            # TODO: Maybe I should put an error catch for memory overflow here?
            cov_mat_inv = self._get_inv_matrix(cov_mat[i])

            # Save the covariance matrix and inverse to hdf5
            self._save_data(emul_i, emul_s, {
                'cov_mat': [cov_mat[i], cov_mat_inv]})

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
        logger = getCLogger('EMUL_LOAD')
        logger.info("Loading emulator system.")

        # Check if an existing hdf5-file is provided
        try:
            logger.info("Checking if provided emulator file '%s' is a "
                        "constructed emulator system."
                        % (self._pipeline._hdf5_file))
            with PRISM_File('r', None) as file:
                # Existing emulator was provided
                logger.info("Constructed emulator HDF5-file provided.")
                self._emul_load = 1

                # Obtain the number of emulator iterations constructed
                self._emul_i = len(file.keys())

                # Check if the hdf5-file contains solely groups made by PRISM
                req_keys = [str(i) for i in range(1, self._emul_i+1)]
                if(req_keys != list(file.keys())):
                    logger.error("Provided emulator HDF5-file contains invalid"
                                 " data groups!")
                    raise RequestError("Provided emulator HDF5-file contains "
                                       "invalid data groups!")
        except (OSError, IOError):
            # No existing emulator was provided
            logger.info("Non-existing HDF5-file provided.")
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
        logger = getCLogger('INIT')
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
            # Set ModelLink object for Emulator
            self._modellink = modellink_obj

            # Set ModelLink object for Pipeline
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

        # Logging
        logger.info("ModelLink object set to '%s'." % (self._modellink._name))

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
        logger = getLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing emulator data sets.")
        self._n_sam = [[]]
        self._sam_set = [[]]
        self._mod_set = [[]]
        self._cov_mat_inv = [[]]
        self._exp_dot_term = [[]]
        self._active_par_data = [[]]
        self._rsdl_var = [[]]
        self._poly_coef = [[]]
        self._poly_coef_cov = [[]]
        self._poly_powers = [[]]
        self._poly_idx = [[]]
        self._n_data = [[]]
        self._data_val = [[]]
        self._data_err = [[]]
        self._data_spc = [[]]
        self._data_idx = [[]]

        self._ccheck = [[]]
        self._active_emul_s = [[]]
        self._n_emul_s = 0

        # Initialize rank specific properties
        if self._is_controller:
            self._n_data_tot = [[]]
            self._n_emul_s_tot = 0
            self._emul_s_to_core = [[] for _ in range(self._size)]
            self._active_par = [[]]

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

        # If both checks succeed, assign emulator systems to the MPI ranks
        if self._is_controller:
            # Determine which emulator systems each MPI rank should get
            emul_s_to_core = self._assign_emul_s(emul_i)

            # Controller saving which systems have been assigned to which rank
            self._emul_s_to_core = emul_s_to_core

            # Initialize total number of emulator systems
            self._n_emul_s_tot = 0

            # Assign the emulator systems to the various MPI ranks
            for rank, emul_s_seq in enumerate(emul_s_to_core):
                # Log which systems are assigned to which rank
                logger.info("Assigning emulator systems %s to MPI rank %s."
                            % (emul_s_seq, rank))

                # Update total number of emulator systems
                self._n_emul_s_tot += len(emul_s_seq)

                # Assign the first list of emul_s to the controller
                if not rank:
                    self._emul_s = emul_s_seq

                # Assign the remaining ones to the workers
                else:
                    self._comm.send(emul_s_seq, dest=rank, tag=888+rank)

                # Log that assignments are finished
                logger.info("Finished assigning emulator systems to available "
                            "MPI ranks.")

        # The workers wait for the controller to assign them their systems
        else:
            self._emul_s = self._comm.recv(source=0, tag=888+self._rank)

        # Determine the number of assigned emulator systems
        self._n_emul_s = len(self._emul_s)

        # Load the corresponding sam_set, mod_set and cov_mat_inv
        logger.info("Loading relevant emulator data up to iteration %s."
                    % (emul_i))

        # Open hdf5-file
        with PRISM_File('r', None) as file:
            # Read in the data
            for i in range(1, emul_i+1):
                group = file['%s' % (i)]

                # Create empty construct check list
                ccheck = []

                # Check if sam_set is available
                try:
                    self._n_sam.append(group.attrs['n_sam'])
                    self._sam_set.append(group['sam_set'][()])
                    self._sam_set[-1].dtype = float
                except KeyError:
                    self._n_sam.append([])
                    self._sam_set.append([])
                    if self._is_controller:
                        ccheck.append('mod_real_set')

                # Check if active_par is available for the controller
                if self._is_controller:
                    try:
                        par_i = [self._modellink._par_name.index(par.decode(
                                'utf-8')) for par in group.attrs['active_par']]
                        self._active_par.append(np.array(par_i))
                    except KeyError:
                        self._active_par.append([])
                        ccheck.append('active_par')

                # Read in the total number of data points for the controller
                if self._is_controller:
                    self._n_data_tot.append(group.attrs['n_data'])

                # Initialize empty data sets
                mod_set = []
                cov_mat_inv = []
                exp_dot_term = []
                active_par_data = []
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
                        data_set = group['emul_%s' % (emul_s)]
                    # If it does not exist, it was passive
                    except KeyError:
                        # Add empty lists for all emulator system data
                        mod_set.append([])
                        cov_mat_inv.append([])
                        exp_dot_term.append([])
                        active_par_data.append([])
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

                    # Check if active_par_data is available
                    try:
                        par_i = [self._modellink._par_name.index(
                            par.decode('utf-8')) for par in
                            data_set.attrs['active_par_data']]
                        active_par_data.append(np.array(par_i))
                    except KeyError:
                        active_par_data.append([])
                        ccheck_s.append('active_par_data')

                    # Read in data values, errors and spaces
                    data_val.append(data_set.attrs['data_val'])
                    data_err.append(data_set.attrs['data_err'].tolist())
                    data_spc.append(
                        data_set.attrs['data_spc'].decode('utf-8'))

                    # Read in all data_idx parts and combine them
                    idx_keys = [key for key in data_set.attrs.keys()
                                if key[:8] == 'data_idx']
                    idx_len = len(idx_keys)
                    if(idx_len == 1):
                        if isinstance(data_set.attrs['data_idx'], bytes):
                            data_idx.append(
                                data_set.attrs['data_idx'].decode('utf-8'))
                        else:
                            data_idx.append(data_set.attrs['data_idx'])
                    else:
                        tmp_data_idx = []
                        for key in idx_keys:
                            if isinstance(data_set.attrs[key], bytes):
                                idx_str =\
                                    data_set.attrs[key].decode('utf-8')
                                tmp_data_idx.append(idx_str)
                            else:
                                tmp_data_idx.append(data_set.attrs[key])
                        data_idx.append(tmp_data_idx)

                    # Add ccheck_s to ccheck
                    ccheck.insert(j, ccheck_s)

                # Determine the number of data points on this MPI rank
                self._n_data.append(len(self._active_emul_s[-1]))

                # Add all read-in data to their respective places
                self._mod_set.append(mod_set)
                self._cov_mat_inv.append(cov_mat_inv)
                self._exp_dot_term.append(exp_dot_term)
                self._active_par_data.append(active_par_data)
                self._data_val.append(data_val)
                self._data_err.append(data_err)
                self._data_spc.append(data_spc)
                self._data_idx.append(data_idx)

                # If regression is used, also read in regression data
                if self._method.lower() in ('regression', 'full'):
                    rsdl_var = []
                    poly_coef = []
                    if self._use_regr_cov:
                        poly_coef_cov = []
                    poly_powers = []
                    poly_idx = []
                    for j, emul_s in enumerate(self._emul_s):
                        # Try to access the emulator system
                        try:
                            data_set = group['emul_%s' % (emul_s)]
                        # If it does not exist, it was passive
                        except KeyError:
                            # Add empty lists for all regression data
                            rsdl_var.append([])
                            poly_coef.append([])
                            if self._use_regr_cov:
                                poly_coef_cov.append([])
                            poly_powers.append([])
                            poly_idx.append([])
                            continue

                        # Check if regression variables are available
                        try:
                            rsdl_var.append(data_set.attrs['rsdl_var'])
                            poly_coef.append(data_set['poly_coef'][()])
                            if self._use_regr_cov:
                                poly_coef_cov.append(
                                    data_set['poly_coef_cov'][()])
                            poly_powers.append(data_set['poly_powers'][()])
                            poly_powers[-1].dtype = 'int64'
                            poly_idx.append(data_set['poly_idx'][()])
                        except KeyError:
                            rsdl_var.append([])
                            poly_coef.append([])
                            if self._use_regr_cov:
                                poly_coef_cov.append([])
                            poly_powers.append([])
                            poly_idx.append([])
                            ccheck[j].append('regression')

                    self._rsdl_var.append(rsdl_var)
                    self._poly_coef.append(poly_coef)
                    if self._use_regr_cov:
                        self._poly_coef_cov.append(poly_coef_cov)
                    self._poly_powers.append(poly_powers)
                    self._poly_idx.append(poly_idx)

                # Add ccheck for this iteration to global ccheck
                self._ccheck.append(ccheck)

                # If ccheck has no empty lists, decrease emul_i by 1
                if(delist(ccheck) != []):
                    self._emul_i -= 1

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
        logger = getLogger('SAVE_DATA')

        # If controller keyword contains 'mod_real_set', emul_s must be None
        if((self._is_controller and 'mod_real_set' in data_dict.keys()) or
           lemul_s is None):
            emul_s = None
        # Else, determine what the global emul_s is
        else:
            emul_s = self._emul_s[lemul_s]

        # Open hdf5-file
        with PRISM_File('r+', emul_s) as file:
            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %s data at iteration %s to HDF5."
                            % (keyword, emul_i))

                # Check what data keyword has been provided
                # ACTIVE PARAMETERS
                if(keyword == 'active_par'):
                    par_names = [self._modellink._par_name[i].encode(
                        'ascii', 'ignore') for i in data]
                    data_set = file['%s' % (emul_i)]
                    data_set.attrs['active_par'] = par_names
                    self._active_par[emul_i] = data
                    self._ccheck[emul_i].remove('active_par')

                # ACTIVE PARAMETERS DATA
                elif(keyword == 'active_par_data'):
                    par_names = [self._modellink._par_name[i].encode(
                        'ascii', 'ignore') for i in data]
                    data_set = file['%s' % (emul_i)]
                    data_set.attrs['active_par_data'] = par_names
                    self._active_par_data[emul_i][lemul_s] = data
                    self._ccheck[emul_i][lemul_s].remove('active_par_data')

                # COV_MAT
                elif(keyword == 'cov_mat'):
                    data_set = file['%s' % (emul_i)]
                    data_set.create_dataset('cov_mat', data=data[0])
                    data_set.create_dataset('cov_mat_inv', data=data[1])
                    self._cov_mat_inv[emul_i][lemul_s] = data[1]
                    self._ccheck[emul_i][lemul_s].remove('cov_mat')

                # EXP_DOT_TERM
                elif(keyword == 'exp_dot_term'):
                    data_set = file['%s' % (emul_i)]
                    data_set.create_dataset('prior_exp_sam_set', data=data[0])
                    data_set.create_dataset('exp_dot_term', data=data[1])
                    self._exp_dot_term[emul_i][lemul_s] = data[1]
                    self._ccheck[emul_i][lemul_s].remove('exp_dot_term')

                # MOD_REAL_SET (CONTROLLER)
                elif(self._is_controller and keyword == 'mod_real_set'):
                    dtype = [(n, float) for n in self._modellink._par_name]
                    data_c = data[0].copy()
                    data_c.dtype = dtype
                    file.create_dataset('%s/sam_set' % (emul_i), data=data_c)
                    file['%s' % (emul_i)].attrs['n_sam'] = np.shape(data[0])[0]
                    self._sam_set[emul_i] = data[0]
                    self._n_sam[emul_i] = np.shape(data[0])[0]

                    for i, lemul_s in enumerate(self._active_emul_s[emul_i]):
                        emul_s = self._emul_s[lemul_s]
                        data_set = file['%s/emul_%s' % (emul_i, emul_s)]
                        data_set.create_dataset('mod_set', data=data[1][i])
                        self._mod_set[emul_i][lemul_s] = data[1][i]

                    file['%s' % (emul_i)].attrs['use_ext_real_set'] =\
                        bool(data[2])
                    self._ccheck[emul_i].remove('mod_real_set')

                # MOD_REAL_SET (WORKER)
                elif(self._is_worker and keyword == 'mod_real_set'):
                    self._sam_set[emul_i] = data[0]
                    self._n_sam[emul_i] = np.shape(data[0])[0]

                    data_set = file['%s' % (emul_i)]
                    data_set.create_dataset('mod_set', data=data[1])
                    self._mod_set[emul_i][lemul_s] = data[1]

                # REGRESSION
                elif(keyword == 'regression'):
                    data_set = file['%s' % (emul_i)]
                    names = [self._modellink._par_name[par] for par in
                             self._active_par_data[emul_i][lemul_s]]
                    dtype = [(n, 'int64') for n in names]
                    data_c = data[3].copy()
                    data_c.dtype = dtype
                    data_set.attrs['rsdl_var'] = data[0]
                    data_set.attrs['regr_score'] = data[1]
                    data_set.create_dataset('poly_coef', data=data[2])
                    data_set.create_dataset('poly_powers', data=data_c)
                    data_set.create_dataset('poly_idx', data=data[4])
                    self._rsdl_var[emul_i][lemul_s] = data[0]
                    self._poly_coef[emul_i][lemul_s] = data[2]
                    self._poly_powers[emul_i][lemul_s] = data[3]
                    self._poly_idx[emul_i][lemul_s] = data[4]
                    if self._use_regr_cov:
                        data_set.create_dataset('poly_coef_cov', data=data[5])
                        self._poly_coef_cov[lemul_s] = data[5]
                    self._ccheck[emul_i][lemul_s].remove('regression')

                # INVALID KEYWORD
                else:
                    logger.error("Invalid keyword argument provided!")
                    raise ValueError("Invalid keyword argument provided!")

        # More logging
        logger.info("Finished saving data to HDF5.")

    # Read in the emulator attributes
    def _retrieve_parameters(self):
        """
        Reads in the emulator parameters from the provided HDF5-file and saves
        them in the current :obj:`~Emulator` instance.

        """

        # Log that parameters are being read
        logger = getCLogger('INIT')
        logger.info("Retrieving emulator parameters from provided HDF5-file.")

        # Open hdf5-file
        with PRISM_File('r', None) as file:
            # Read in all the emulator parameters
            self._sigma = file.attrs['sigma']
            self._l_corr = file.attrs['l_corr']
            self._method = file.attrs['method'].decode('utf-8')
            self._use_regr_cov = int(file.attrs['use_regr_cov'])
            self._poly_order = file.attrs['poly_order']
            modellink_name = file.attrs['modellink_name'].decode('utf-8')
            self._use_mock = int(file.attrs['use_mock'])

            # Obtain used PRISM version and emulator type
            emul_version = file.attrs['prism_version'].decode('utf-8')
            emul_type = file.attrs['emul_type'].decode('utf-8')

        # Check if provided emulator is the same as requested
        if(emul_type != self._emul_type):
            logger.error("Provided emulator system type ('%s') does not "
                         "match the requested type ('%s')!"
                         % (emul_type, self._emul_type))
            raise RequestError("Provided emulator system type ('%s') does "
                               "not match the requested type ('%s')!"
                               % (emul_type, self._emul_type))

        # Check if provided emul_version is compatible
        check_compatibility(emul_version)

        # Log that reading is finished
        logger.info("Finished retrieving parameters.")

        # Return the name of the modellink class used to construct the loaded
        # emulator system
        return(modellink_name)

    # This function automatically loads default emulator parameters
    @docstring_append(def_par_doc.format('emulator'))
    def _get_default_parameters(self):
        # Log this
        logger = getCLogger('INIT')
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
    @docstring_append(read_par_doc.format("Emulator"))
    def _read_parameters(self):
        # Log that the PRISM parameter file is being read
        logger = getCLogger('INIT')
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
        self._sigma = check_val(float(par_dict['sigma']), 'sigma', 'nzero')

        # Gaussian correlation length
        self._l_corr = check_val(float(par_dict['l_corr']), 'l_corr', 'pos') *\
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
        self._use_regr_cov = check_val(par_dict['use_regr_cov'],
                                       'use_regr_cov', 'bool')

        # Check if method == 'regression' and set use_regr_cov to True if so
        if self._method.lower() == 'regression':
            self._use_regr_cov = 1

        # Obtain the polynomial order for the regression selection process
        self._poly_order = check_val(int(par_dict['poly_order']), 'poly_order',
                                     'pos')

        # Obtain the bool determining whether or not to use mock data
        self._use_mock = check_val(par_dict['use_mock'], 'use_mock', 'bool')

        # Log that reading has been finished
        logger.info("Finished reading emulator parameters.")

    # This function loads previously generated mock data into ModelLink
    # TODO: Allow user to add/remove mock data? Requires consistency check
    def _set_mock_data(self):
        """
        Loads previously used mock data into the :class:`~ModelLink` object,
        overwriting the parameter estimates, data values, data errors, data
        spaces and data identifiers with their mock equivalents.

        Generates
        ---------
        Overwrites the corresponding :class:`~ModelLink` class properties with
        the previously used values (taken from the first emulator iteration).

        """

        # Start logger
        logger = getCLogger('MOCK_DATA')

        # Overwrite ModelLink properties with previously generated values
        # Log that mock_data is being loaded in
        logger.info("Loading previously used mock data into ModelLink object.")

        # Controller only
        if self._is_controller:
            # Open hdf5-file
            with PRISM_File('r', None) as file:
                # Get the number of emulator systems in the first iteration
                group = file['1']
                n_emul_s = group.attrs['n_emul_s']

                # Make empty lists for all model properties
                data_val = []
                data_err = []
                data_spc = []
                data_idx = []

                # Loop over all data points in the first iteration
                for i in range(n_emul_s):
                    # Read in data values, errors and spaces
                    data_set = group['emul_%s' % (i)]
                    data_val.append(data_set.attrs['data_val'])
                    data_err.append(data_set.attrs['data_err'].tolist())
                    data_spc.append(data_set.attrs['data_spc'].decode('utf-8'))

                    # Read in all data_idx parts and combine them
                    idx_keys = [key for key in data_set.attrs.keys()
                                if key[:8] == 'data_idx']
                    idx_len = len(idx_keys)
                    if(idx_len == 1):
                        if isinstance(data_set.attrs['data_idx'], bytes):
                            data_idx.append(
                                data_set.attrs['data_idx'].decode('utf-8'))
                        else:
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

                # Overwrite ModelLink properties
                self._modellink._par_est = file.attrs['mock_par'].tolist()
                self._modellink._n_data = group.attrs['n_data']
                self._modellink._data_val = data_val
                self._modellink._data_err = data_err
                self._modellink._data_spc = data_spc
                self._modellink._data_idx = data_idx

        # Broadcast updated ModelLink object to workers
        self._modellink = self._comm.bcast(self._modellink, 0)

        # Log end
        logger.info("Loaded mock data.")
