# -*- coding: utf-8 -*-

"""
Pipeline
========
Provides the definition of the main class of the PRISM package, the
:class:`~Pipeline` class.


Available classes
-----------------
:class:`~Pipeline`
    Defines the :class:`~Pipeline` class of the PRISM package.

"""


# %% IMPORTS
# Future imports
from __future__ import (absolute_import, division, print_function,
                        with_statement)

# Built-in imports
from logging import getLogger
import os
from os import path
import sys
from time import strftime, strptime, time

# Package imports
from e13tools import InputError, ShapeError
from e13tools.math import nCr
from e13tools.sampling import lhd
try:
    from mpi4py import MPI
except ImportError:
    import mpi_dummy as MPI
import numpy as np
from numpy.random import normal, random
from sortedcontainers import SortedSet

# PRISM imports
from ._docstrings import (call_emul_i_doc, emul_s_seq_doc, std_emul_i_doc,
                          user_emul_i_doc)
from ._internal import (PRISM_File, RequestError, check_bool, check_float,
                        check_nneg_float, check_pos_int, convert_str_seq,
                        delist, docstring_copy, docstring_substitute,
                        getCLogger, move_logger, rprint, start_logger)
from .emulator import Emulator
from .projection import Projection

# All declaration
__all__ = ['Pipeline']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% PIPELINE CLASS DEFINITION
# OPTIMIZE: Rewrite PRISM into MPI?
# TODO: Allow user to switch between emulation and modelling
# TODO: Implement multivariate implausibilities
# OPTIMIZE: Overlap plausible regions to remove boundary artifacts?
# TODO: Allow ModelLink to provide full data set, Pipeline selects data itself?
# TODO: Think of a way to allow no ModelLink instance to be provided.
# This could be done with a DummyLink, but md_var is then uncallable.
class Pipeline(object):
    """
    Defines the :class:`~Pipeline` class of the PRISM package.

    """

    # TODO: Should prism_file be defaulted to None?
    def __init__(self, modellink, root_dir=None, working_dir=None,
                 prefix='prism_', hdf5_file='prism.hdf5',
                 prism_file='prism.txt', emul_type='default'):
        """
        Initialize an instance of the :class:`~Pipeline` class.

        Parameters
        ----------
        modellink : :obj:`~ModelLink` object
            Instance of the :class:`~ModelLink` class that links the emulated
            model to this :obj:`~Pipeline` object.

        Optional
        --------
        root_dir : str or None. Default: None
            String containing the absolute path of the root directory where all
            working directories are stored. If *None*, root directory will be
            set to the directory this class was initialized at.
        working_dir : str, int or None. Default: None
            String containing the name of the working directory of the emulator
            in `root_dir`. If int, a new working directory will be created in
            `root_dir`. If *None*, working directory is set to the last one
            that was created in `root_dir` that starts with the given `prefix`.
            If no directories are found, one will be created.
        prefix : str. Default: 'prism_'
            String containing a prefix that is used for naming new working
            directories or scan for existing ones.
        hdf5_file : str. Default: 'prism.hdf5'
            String containing the name of the master HDF5-file in `working_dir`
            to be used in this class instance. Different types of master
            HDF5-files can be provided:
                *Non-existing master HDF5-file*: This file will be created and
                used to save the constructed emulator system in.

                *Existing master HDF5-file*: This file will be used to
                regenerate a previously constructed emulator system.
        prism_file : str or None. Default: 'prism.txt'
            String containing the absolute or relative path to the TXT-file
            containing the PRISM parameters that need to be changed from their
            default values. If a relative path is given, its path must be
            relative to `root_dir` or the current directory. If *None*, no
            changes will be made to the default parameters.

        """

        # Obtain MPI communicator, ranks and sizes
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        # Set statuses
        self._is_controller = 1 if not self._rank else 0
        self._is_worker = 1 if self._rank else 0

        # Controller only
        if self._is_controller:
            # Start logging
            logging_file = start_logger()
            logger = getCLogger('PIPELINE')
            logger.info("")

            # Initialize class
            logger = getCLogger('INIT')
            logger.info("Initializing Pipeline class.")

            # Obtain paths
            self._get_paths(root_dir, working_dir, prefix, hdf5_file,
                            prism_file)

            # Move logger to working directory
            move_logger(self._working_dir, logging_file)

        # Remaining workers
        else:
            # Obtain paths
            self._get_paths(root_dir, working_dir, prefix, hdf5_file,
                            prism_file)

            # Start logger for workers as well
            start_logger(path.join(self._working_dir, 'prism_log.log'), 'a')

        # Initialize Emulator class
        if(emul_type == 'default'):
            self._emulator = Emulator(self, modellink)
        else:
            raise RequestError("Input argument 'emul_type' is invalid!")

        # Let everybody read in the pipeline parameters
        self._read_parameters()

        # Let controller load in the data
        if self._is_controller:
            self._load_data()

        # Print out the details of the current state of the pipeline
        self.details()

    # Allows one to call one full loop of the PRISM pipeline
    @docstring_substitute(emul_i=call_emul_i_doc)
    def __call__(self, emul_i=None):
        """
        Calls the :meth:`~construct` method to start the construction of the
        given iteration of the emulator system and creates the projection
        figures right afterward if this construction was successful.

        Optional
        --------
        %(emul_i)s

        """

        # Perform construction
        try:
            self.construct(emul_i)
        except Exception:
            raise
        else:
            try:
                # Perform projection
                self.project()

                # Print details
                self.details()
            except Exception:
                raise


# %% CLASS PROPERTIES
    # TODO: Hide class attributes that do not exist yet
    # MPI properties
    @property
    def comm(self):
        """
        The global MPI world communicator. Currently always MPI.COMM_WORLD.

        """

        return(self._comm)

    @property
    def rank(self):
        """
        The rank of this MPI process in :attr:`~Pipeline.comm`. If no MPI is
        used, this is always 0.

        """

        return(self._rank)

    @property
    def size(self):
        """
        The number of MPI processes in :attr:`~Pipeline.comm`. If no MPI is
        used, this is always 1.

        """

        return(self._size)

    @property
    def is_controller(self):
        """
        Bool indicating whether or not this MPI process is a controller
        process. Is no MPI is used, this is always *True*.

        """

        return(bool(self._is_controller))

    @property
    def is_worker(self):
        """
        Bool indicating whether or not this MPI process is a worker process. If
        no MPI is used, this is always *False*.

        """

        return(bool(self._is_worker))

    # Pipeline Settings/Attributes/Details
    @property
    def root_dir(self):
        """
        Absolute path to the root directory.

        """

        return(self._root_dir)

    @property
    def working_dir(self):
        """
        Absolute path to the working directory.

        """

        return(self._working_dir)

    @property
    def hdf5_file(self):
        """
        Absolute path to the loaded HDF5-file.

        """

        return(self._hdf5_file)

    @property
    def prism_file(self):
        """
        Absolute path to PRISM parameters file.

        """

        return(self._prism_file)

    @property
    def modellink(self):
        """
        The :obj:`~ModelLink` instance provided during Pipeline initialization.

        """

        return(self._modellink)

    @property
    def emulator(self):
        """
        The :obj:`~Emulator` instance created during Pipeline initialization.

        """

        return(self._emulator)

    @property
    def criterion(self):
        """
        String or float indicating which criterion to use in the
        :func:`e13tools.sampling.lhd` function.

        """

        return(self._criterion)

    @property
    def do_active_anal(self):
        """
        Bool indicating whether or not to do an active parameters analysis.

        """

        return(bool(self._do_active_anal))

    @property
    def freeze_active_par(self):
        """
        Bool indicating whether or not previously active parameters always stay
        active.

        """

        return(bool(self._freeze_active_par))

    @property
    def pot_active_par(self):
        """
        List of potentially active parameters. Only parameters from this list
        can become active.

        """

        return([self._modellink._par_name[i] for i in self._pot_active_par])

    @property
    def n_sam_init(self):
        """
        Number of evaluation samples used to construct the first iteration of
        the emulator system.

        """

        return(self._n_sam_init)

    @property
    def n_eval_sam(self):
        """
        List containing the number of evaluation samples used to analyze the
        corresponding emulator iteration of the emulator system. The number of
        plausible evaluation samples is stored in :attr:`~Pipeline.n_impl_sam`.

        """

        return(self._n_eval_sam)

    @property
    def base_eval_sam(self):
        """
        Base number of emulator evaluations used to analyze the emulator
        system. This number is scaled up by the number of model parameters and
        the current emulator iteration to generate the true number of emulator
        evaluations (:attr:`~Pipeline.n_eval_sam`).

        """

        return(self._base_eval_sam)

    @property
    def impl_cut(self):
        """
        List of lists containing all univariate implausibility cut-offs. A zero
        indicates a wildcard.

        """

        return(self._impl_cut)

    @property
    def cut_idx(self):
        """
        List of list indices of the first non-wildcard cut-off in impl_cut.

        """

        return(self._cut_idx)

    @property
    def prc(self):
        """
        Bool indicating whether or not plausible regions have been found in the
        last emulator iteration.

        """

        return(bool(self._prc))

    @property
    def n_impl_sam(self):
        """
        List of number of model evaluation samples that have been added to the
        corresponding emulator iteration.

        """

        return(self._n_impl_sam)

    @property
    def impl_sam(self):
        """
        Array containing all model evaluation samples that will be added to the
        next emulator iteration.

        """

        return(self._impl_sam)


# %% GENERAL CLASS METHODS
    # Function containing the model output for a given set of parameter values
    # TODO: May want to save all model output immediately to prevent data loss
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _call_model(self, emul_i, par_set, data_idx):
        """
        Obtain the output that is generated by the model for a given model
        parameter value set `par_set`. The current emulator iteration `emul_i`
        is also provided in case it is required by the :class:`~ModelLink`
        subclass.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D array_like
            Model parameter value set to calculate the model output for.
        data_idx : list of lists
            The list of data identifiers for which the model is requested to
            return the corresponding data values.

        Returns
        -------
        mod_out : 1D :obj:`~numpy.ndarray` object
            Model output corresponding to given `par_set`.

        """

        # Make sure par_set is at least 1D and a numpy array
        sam = np.array(par_set, ndmin=1)

        # Log that model is being called
        logger = getCLogger('CALL_MODEL')
        logger.info("Calling model at parameters %s." % (sam))

        # Create par_dict
        par_dict = dict(zip(self._modellink._par_name, sam))

        # Obtain model output
        mod_out = self._modellink.call_model(emul_i, par_dict, data_idx)

        # Log that calling model has been finished
        logger.info("Model returned %s." % (mod_out))

        # Return it
        return(np.array(mod_out))

    # Function containing the model output for a given set of parameter samples
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _multi_call_model(self, emul_i, sam_set, data_idx):
        """
        Obtain the output set that is generated by the model for a given set of
        model parameter samples `sam_set`. The current emulator iteration
        `emul_i` is also provided in case it is required by the
        :class:`~ModelLink` subclass.

        This is a multi-version of the :meth:`~Pipeline._call_model` method.

        Parameters
        ----------
        %(emul_i)s
        sam_set : 2D array_like
            Model parameter sample set to calculate the model output for.
        data_idx : list of lists
            The list of data identifiers for which the model is requested to
            return the corresponding data values.

        Returns
        -------
        mod_set : 2D :obj:`~numpy.ndarray` object
            Model output set corresponding to given `sam_set`.

        """

        # Make sure that sam_set is at least 2D and a numpy array
        sam_set = np.array(sam_set, ndmin=2)

        # Log that model is being multi-called
        logger = getCLogger('CALL_MODEL')
        logger.info("Multi-calling model for sample set of size %s."
                    % (np.shape(sam_set)[0]))

        # Create sam_dict
        sam_dict = dict(zip(self._modellink._par_name, sam_set.T))

        # Obtain set of model outputs
        mod_set = self._modellink.call_model(emul_i, sam_dict, data_idx)

        # Log that multi-calling model has been finished
        logger.info("Finished model multi-call.")

        # Return it
        return(np.array(mod_set))

    # This function automatically loads default pipeline parameters
    def _get_default_parameters(self):
        """
        Generates a dict containing default values for all pipeline parameters.

        Returns
        -------
        par_dict : dict
            Dict containing all default pipeline parameter values.

        """

        # Log this
        logger = getCLogger('INIT')
        logger.info("Generating default pipeline parameter dict.")

        # Create parameter dict with default parameters
        par_dict = {'n_sam_init': '500',
                    'base_eval_sam': '800',
                    'impl_cut': '[0, 4.0, 3.8, 3.5]',
                    'criterion': "'multi'",
                    'do_active_anal': 'True',
                    'freeze_active_par': 'True',
                    'pot_active_par': 'None'}

        # Log end
        logger.info("Finished generating default pipeline parameter dict.")

        # Return it
        return(par_dict)

    # Read in the parameters from the provided parameter file
    def _read_parameters(self):
        """
        Reads in the pipeline parameters from the provided PRISM parameter file
        saves them in the current :obj:`~Pipeline` instance.

        """

        # Log that the PRISM parameter file is being read
        logger = getCLogger('INIT')
        logger.info("Reading pipeline parameters.")

        # Obtaining default pipeline parameter dict
        par_dict = self._get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._prism_file is not None:
            pipe_par = np.genfromtxt(self._prism_file, dtype=(str),
                                     delimiter=':', autostrip=True)

            # Make sure that pipe_par is 2D
            pipe_par = np.array(pipe_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(pipe_par)

        # More logging
        logger.info("Checking compatibility of provided pipeline parameters.")

        # GENERAL
        # Number of starting samples
        self._n_sam_init = check_pos_int(int(par_dict['n_sam_init']),
                                         'n_sam_init')

        # Base number of emulator evaluation samples
        self._base_eval_sam = check_pos_int(int(par_dict['base_eval_sam']),
                                            'base_eval_sam')

        # Criterion parameter used for Latin Hypercube Sampling
        if(par_dict['criterion'].lower() == 'none'):
            self._criterion = None
        elif par_dict['criterion'].lower() in ('false', 'true'):
            logger.error("Input argument 'criterion' does not accept values "
                         "of type 'bool'!")
            raise TypeError("Input argument 'criterion' does not accept "
                            "values of type 'bool'!")
        else:
            try:
                self._criterion = float(par_dict['criterion'])
            except ValueError:
                self._criterion = str(par_dict['criterion']).replace("'", '')

        # Obtain the bool determining whether to do an active parameters
        # analysis
        self._do_active_anal = check_bool(par_dict['do_active_anal'],
                                          'do_active_anal')

        # Obtain the bool determining whether active parameters stay active
        self._freeze_active_par = check_bool(par_dict['freeze_active_par'],
                                             'freeze_active_par')

        # Check which parameters can potentially be active
        if(par_dict['pot_active_par'].lower() == 'none'):
            self._pot_active_par = np.array(range(self._modellink._n_par))
        elif par_dict['pot_active_par'].lower() in ('false', 'true'):
            logger.error("Input argument 'pot_active_par' does not accept "
                         "values of type 'bool'!")
            raise TypeError("Input argument 'pot_active_par' does not accept "
                            "values of type 'bool'!")
        else:
            # Remove all unwanted characters from the string and split it up
            pot_active_par = convert_str_seq(par_dict['pot_active_par'])

            # Check elements if they are ints or strings, and if they are valid
            for i, string in enumerate(pot_active_par):
                try:
                    try:
                        par_idx = int(string)
                    except ValueError:
                        pot_active_par[i] =\
                            self._modellink._par_name.index(string)
                    else:
                        self._modellink._par_name[par_idx]
                        pot_active_par[i] = par_idx % self._modellink._n_par
                except Exception as error:
                    logger.error("Input argument 'pot_active_par' is invalid! "
                                 "(%s)" % (error))
                    raise InputError("Input argument 'pot_active_par' is "
                                     "invalid! (%s)" % (error))

            # If everything went without exceptions, check if list is not empty
            if(len(pot_active_par) != 0):
                self._pot_active_par =\
                    np.array(list(SortedSet(pot_active_par)))
            else:
                logger.error("Input argument 'pot_active_par' is empty!")
                raise ValueError("Input argument 'pot_active_par' is empty!")

        # Log that reading has been finished
        logger.info("Finished reading pipeline parameters.")

    # This function controls how n_eval_samples is calculated
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_n_eval_sam(self, emul_i):
        """
        This function calculates the total amount of emulator evaluation
        samples at a given emulator iteration `emul_i` from the
        `base_eval_sam` provided during class initialization.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        n_eval_sam : int
            Number of emulator evaluation samples.

        """

        # Calculate n_eval_sam
        return(emul_i*self._base_eval_sam*self._modellink._n_par)

    # Obtains the paths for the root directory, working directory, pipeline
    # hdf5-file and prism parameters file
    def _get_paths(self, root_dir, working_dir, prefix, hdf5_file, prism_file):
        """
        Obtains the path for the root directory, working directory, HDF5-file
        and parameters file for PRISM.

        Parameters
        ----------
        root_dir : str or None
            String containing the absolute path to the root directory where all
            working directories are stored. If *None*, root directory will be
            set to the directory where this class was initialized at.
        working_dir : str or None
            String containing the name of the working directory of the emulator
            in `root_dir`. If int, a new working directory will be created in
            `root_dir`. If *None*, working directory is set to the last one
            that was created in `root_dir` that starts with the given `prefix`.
            If no directories are found, one will be created.
        prefix : str
            String containing a prefix that is used for naming new working
            directories or scan for existing ones.
        hdf5_file : str
            String containing the name of the master HDF5-file in `working_dir`
            to be used in this class instance.
        prism_file : str or None
            String containing the absolute or relative path to the TXT-file
            containing the PRISM parameters that need to be changed from their
            default values. If a relative path is given, its path must be
            relative to `root_dir` or the current directory. If *None*, no
            changes will be made to the default parameters.

        Generates
        ---------
        The absolute paths to the root directory, working directory, pipeline
        HDF5-file and PRISM parameters file.

        """

        # Controller obtaining the paths
        if self._is_controller:
            # Set logging system
            logger = getCLogger('INIT')
            logger.info("Obtaining related directory and file paths.")

            # Obtain root directory path
            # If one did not specify a root directory, set it to default
            if root_dir is None:
                self._root_dir = path.abspath('.')
                logger.info("No root directory specified, set to '%s'."
                            % (self._root_dir))

            # If one specified a root directory, use it
            elif isinstance(root_dir, (str, unicode)):
                self._root_dir = path.abspath(root_dir)
                logger.info("Root directory set to '%s'." % (self._root_dir))

                # Check if this directory already exists
                try:
                    os.mkdir(self._root_dir)
                except OSError:
                    pass
                else:
                    logger.info("Root directory did not exist, created it.")
            else:
                logger.error("Input argument 'root_dir' is invalid!")
                raise InputError("Input argument 'root_dir' is invalid!")

            # Check if a valid working directory prefix string is given
            if isinstance(prefix, (str, unicode)):
                prefix = prefix
                prefix_len = len(prefix)
            else:
                logger.error("Input argument 'prefix' is not of type 'str'!")
                raise TypeError("Input argument 'prefix' is not of type "
                                "'str'!")

            # Obtain working directory path
            # If one did not specify a working directory, obtain it
            if working_dir is None:
                logger.info("No working directory specified, trying to load "
                            "last one created.")
                dirnames = next(os.walk(self._root_dir))[1]
                emul_dirs = list(dirnames)

                # Check which directories in the root_dir satisfy the default
                # naming scheme of the emulator directories
                for dirname in dirnames:
                    if(dirname[0:prefix_len] != prefix):
                        emul_dirs.remove(dirname)
                    else:
                        try:
                            strptime(dirname[prefix_len:prefix_len+10],
                                     '%Y-%m-%d')
                        except ValueError:
                            emul_dirs.remove(dirname)

                # If no working directory exists, make a new one
                if(len(emul_dirs) == 0):
                    working_dir = ''.join([prefix, strftime('%Y-%m-%d')])
                    self._working_dir = path.join(self._root_dir, working_dir)
                    os.mkdir(self._working_dir)
                    logger.info("No working directories found, created '%s'."
                                % (working_dir))

                # If working directories exist, load last one created
                else:
                    emul_dirs.sort(reverse=True)
                    working_dir = emul_dirs[0]
                    self._working_dir = path.join(self._root_dir, working_dir)
                    logger.info("Working directories found, set to '%s'."
                                % (working_dir))

            # If one requested a new working directory
            elif isinstance(working_dir, int):
                working_dir = ''.join([prefix, strftime('%Y-%m-%d')])
                dirnames = next(os.walk(self._root_dir))[1]
                emul_dirs = list(dirnames)

                for dirname in dirnames:
                    if(dirname[0:prefix_len+10] != working_dir):
                        emul_dirs.remove(dirname)

                # Check if working directories already exist with the same
                # prefix and append a number to the name if this is the case
                emul_dirs.sort(reverse=True)
                if(len(emul_dirs) == 0):
                    pass
                elif(len(emul_dirs) == 1):
                    working_dir = ''.join([working_dir, '_1'])
                else:
                    working_dir =\
                        ''.join([working_dir, '_%s'
                                 % (int(emul_dirs[0][prefix_len+11:])+1)])

                self._working_dir = path.join(self._root_dir, working_dir)
                os.mkdir(self._working_dir)
                logger.info("New working directory requested, created '%s'."
                            % (working_dir))

            # If one specified a working directory, use it
            elif isinstance(working_dir, (str, unicode)):
                self._working_dir =\
                    path.join(self._root_dir, working_dir)
                logger.info("Working directory set to '%s'." % (working_dir))

                # Check if this directory already exists
                try:
                    os.mkdir(self._working_dir)
                except OSError:
                    pass
                else:
                    logger.info("Working directory did not exist, created it.")
            else:
                logger.error("Input argument 'working_dir' is invalid!")
                raise InputError("Input argument 'working_dir' is invalid!")

            # Obtain hdf5-file path
            if isinstance(hdf5_file, (str, unicode)):
                # Check if provided filename has correct extension
                ext_idx = hdf5_file.find('.')
                if hdf5_file[ext_idx:] not in ('.hdf5', '.h5'):
                    logger.error("Input argument 'hdf5_file' has invalid HDF5 "
                                 "extension!")
                    raise ValueError("Input argument 'hdf5_file' has invalid "
                                     "HDF5 extension!")

                # Save hdf5-file path and name
                self._hdf5_file = path.join(self._working_dir, hdf5_file)
                logger.info("Master HDF5-file set to '%s'." % (hdf5_file))

                # Save hdf5-file path as a PRISM_File class attribute
                PRISM_File._hdf5_file = self._hdf5_file
            else:
                logger.error("Input argument 'hdf5_file' is not of type "
                             "'str'!")
                raise TypeError("Input argument 'hdf5_file' is not of type "
                                "'str'!")

            # Obtain PRISM parameter file path
            # If no PRISM parameter file was provided
            if prism_file is None:
                self._prism_file = None

            # If a PRISM parameter file was provided
            elif isinstance(prism_file, (str, unicode)):
                if path.exists(prism_file):
                    self._prism_file = path.abspath(prism_file)
                elif path.exists(path.join(self._root_dir, prism_file)):
                    self._prism_file = path.join(self._root_dir, prism_file)
                else:
                    logger.error("Input argument 'prism_file' is a "
                                 "non-existing path (%s)!" % (prism_file))
                    raise OSError("Input argument 'prism_file' is a "
                                  "non-existing path (%s)!" % (prism_file))
                logger.info("PRISM parameters file set to '%s'."
                            % (self._prism_file))
            else:
                logger.error("Input argument 'prism_file' is invalid!")
                raise InputError("Input argument 'prism_file' is invalid!")

        # Workers get dummy paths
        else:
            self._root_dir = None
            self._working_dir = None
            self._hdf5_file = None
            self._prism_file = None

        # Broadcast paths to workers
        self._root_dir = self._comm.bcast(self._root_dir, 0)
        self._working_dir = self._comm.bcast(self._working_dir, 0)
        self._hdf5_file = self._comm.bcast(self._hdf5_file, 0)
        self._prism_file = self._comm.bcast(self._prism_file, 0)

    # This function generates mock data and loads it into ModelLink
    def _get_mock_data(self):
        """
        Generates mock data and loads it into the :obj:`~ModelLink` object that
        was provided during class initialization.
        This function overwrites the :class:`~ModelLink` properties holding the
        parameter estimates, data values and data errors.

        Generates
        ---------
        Overwrites the corresponding :class:`~ModelLink` class properties with
        the generated values.

        """

        # Controller only
        if self._is_controller:
            # Start logger
            logger = getLogger('MOCK_DATA')

            # Log new mock_data being created
            logger.info("Generating mock data for new emulator system.")

            # Set non-default parameter estimate
            self._modellink._par_est =\
                (self._modellink._par_rng[:, 0] +
                 random(self._modellink._n_par) *
                 (self._modellink._par_rng[:, 1] -
                  self._modellink._par_rng[:, 0])).tolist()

        # MPI Barrier
        self._comm.Barrier()

        # Set non-default model data values
        if self._is_controller or self._modellink._MPI_call:
            if self._modellink._multi_call:
                # Multi-call model
                mod_out = self._multi_call_model(0, self._modellink._par_est,
                                                 self._modellink._data_idx)

                # Only controller receives output and can thus use indexing
                if self._is_controller:
                    self._emulator._data_val[0] = mod_out[0].tolist()

            else:
                # Controller only call model
                self._emulator._data_val[0] =\
                    self._call_model(0, self._modellink._par_est,
                                     self._modellink._data_idx).tolist()

        # Controller only
        if self._is_controller:
            # Use model discrepancy variance as model data errors
            md_var = self._get_md_var(0, np.arange(self._modellink._n_data))
            err = np.sqrt(md_var).tolist()
            self._modellink._data_err = err

            # Add model data errors as noise to model data values
            noise = normal(size=self._modellink._n_data)
            for i in range(self._modellink._n_data):
                # If value space is linear
                if(self._modellink._data_spc[i] == 'lin'):
                    if(noise[i] < 0):
                        noise[i] *= err[i][0]
                    else:
                        noise[i] *= err[i][1]
                # If value space is log10
                elif(self._modellink._data_spc[i] == 'log10'):
                    if(noise[i] < 0):
                        noise[i] =\
                            np.log10((pow(10, -1*err[i][0])-1)*-1*noise[i]+1)
                    else:
                        noise[i] = np.log10((pow(10, err[i][0])-1)*noise[i]+1)
                # If value space is ln
                elif(self._modellink._data_spc[i] == 'ln'):
                    if(noise[i] < 0):
                        noise[i] =\
                            np.log((pow(np.e, -1*err[i][0])-1)*-1*noise[i]+1)
                    else:
                        noise[i] = np.log((pow(np.e, err[i][0])-1)*noise[i]+1)
                # If value space is anything else
                else:
                    raise NotImplementedError
            self._modellink._data_val =\
                (self._emulator._data_val[0]+noise).tolist()

            # Logger
            logger.info("Generated mock data.")

        # Broadcast modellink object
        # TODO: Should entire modellink be broadcasted or just the changes?
        self._modellink = self._comm.bcast(self._modellink, 0)

    # This function loads pipeline data
    def _load_data(self):
        """
        Loads in all the important pipeline data into memory for the controller
        rank.
        If it is detected that the last emulator iteration has not been
        analyzed yet, the implausibility analysis parameters are read in from
        the PRISM parameters file and temporarily stored in memory.

        Generates
        ---------
        All relevant pipeline data is loaded into memory.

        """

        # Set the logger
        logger = getLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing pipeline data sets.")
        self._n_impl_sam = [[]]
        self._impl_cut = [[]]
        self._cut_idx = [[]]
        self._n_eval_sam = [[]]

        # If an emulator system currently exists, load in all data
        if self._emulator._emul_i:
            # Open hdf5-file
            with PRISM_File('r', None) as file:
                # Read in the data up to the last emulator iteration
                for i in range(1, self._emulator._emul_i+1):
                    # Get this emulator
                    emul = file['%s' % (i)]

                    # Check if analysis has been carried out (only if i=emul_i)
                    try:
                        self._impl_cut.append(emul.attrs['impl_cut'])

                    # If not, no plausible regions were found
                    except KeyError:
                        self._get_impl_par(True)

                    # If so, load in all data
                    else:
                        self._cut_idx.append(emul.attrs['cut_idx'])
                    finally:
                        self._n_impl_sam.append(emul.attrs['n_impl_sam'])
                        self._n_eval_sam.append(emul.attrs['n_eval_sam'])

                # Read in the samples that survived the implausibility check
                self._prc = int(emul.attrs['prc'])
                self._impl_sam = emul['impl_sam'][()]
                self._impl_sam.dtype = float

    # This function saves pipeline data to hdf5
    def _save_data(self, data_dict):
        """
        Saves a given data dict ``{keyword: data}`` at the last emulator
        iteration to the HDF5-file and as an data attribute to the current
        :obj:`~Pipeline` instance.

        Parameters
        ----------
        data_dict : dict
            Dict containing the data that needs to be saved to the HDF5-file.

        Dict Variables
        --------------
        keyword : {'impl_cut', 'impl_sam', 'n_eval_sam'}
            String specifying the type of data that needs to be saved.
        data : int, float, list
            The actual data that needs to be saved at data keyword `keyword`.

        Generates
        ---------
        The specified data is saved to the HDF5-file.

        """

        # Do some logging
        logger = getLogger('SAVE_DATA')

        # Obtain last emul_i
        emul_i = self._emulator._emul_i

        # Open hdf5-file
        with PRISM_File('r+', None) as file:
            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %s data at iteration %s to HDF5."
                            % (keyword, emul_i))

                # Check what data keyword has been provided
                # IMPL_CUT
                if(keyword == 'impl_cut'):
                    # Check if impl_cut data has been saved before
                    try:
                        self._impl_cut[emul_i] = data[0]
                        self._cut_idx[emul_i] = data[1]
                    except IndexError:
                        self._impl_cut.append(data[0])
                        self._cut_idx.append(data[1])
                    finally:
                        file['%s' % (emul_i)].attrs['impl_cut'] = data[0]
                        file['%s' % (emul_i)].attrs['cut_idx'] = data[1]

                # IMPL_SAM
                elif(keyword == 'impl_sam'):
                    # Check if any plausible regions have been found at all
                    n_impl_sam = np.shape(data)[0]
                    prc = 1 if(n_impl_sam != 0) else 0

                    # Check if impl_sam data has been saved before
                    try:
                        self._n_impl_sam[emul_i] = n_impl_sam
                    except IndexError:
                        file.create_dataset('%s/impl_sam' % (emul_i),
                                            data=data)
                        self._n_impl_sam.append(n_impl_sam)
                    else:
                        dtype = [(n, float) for n in self._modellink._par_name]
                        data_c = list(data)
                        data_c.dtype = dtype
                        del file['%s/impl_sam' % (emul_i)]
                        file.create_dataset('%s/impl_sam' % (emul_i),
                                            data=data_c)
                    finally:
                        self._prc = prc
                        self._impl_sam = data
                        file['%s' % (emul_i)].attrs['prc'] = bool(prc)
                        file['%s' % (emul_i)].attrs['n_impl_sam'] = n_impl_sam

                # N_EVAL_SAM
                elif(keyword == 'n_eval_sam'):
                    # Check if n_eval_sam has been saved before
                    try:
                        self._n_eval_sam[emul_i] = data
                    except IndexError:
                        self._n_eval_sam.append(data)
                    finally:
                        file['%s' % (emul_i)].attrs['n_eval_sam'] = data

                # INVALID KEYWORD
                else:
                    logger.error("Invalid keyword argument provided!")
                    raise ValueError("Invalid keyword argument provided!")

    # This function saves a statistic to hdf5
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _save_statistics(self, emul_i, stat_dict):
        """
        Saves a given statistics dict ``{keyword: [value, unit]}`` at
        emulator iteration `emul_i` to the HDF5-file. The provided values are
        always saved as strings.

        Parameters
        ----------
        %(emul_i)s

        Dict Variables
        --------------
        keyword : str
            String containing the name/keyword of the statistic that is being
            saved.
        value : int, float or str
            The value of the statistic.
        unit : str
            The unit of the statistic.

        """

        # Do logging
        logger = getLogger('STATISTICS')
        logger.info("Saving statistics to HDF5.")

        # Open hdf5-file
        with PRISM_File('r+', None) as file:
            # Save statistics
            for keyword, (value, unit) in stat_dict.items():
                file['%s/statistics' % (emul_i)].attrs[keyword] =\
                    [str(value).encode('ascii', 'ignore'),
                     unit.encode('ascii', 'ignore')]

    # This is function 'k'
    # Reminder that this function should only be called once per sample set
    # TODO: Allow variable n_sam for each data point? More versatile and chaos
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _evaluate_model(self, emul_i, sam_set, ext_sam_set, ext_mod_set):
        """
        Evaluates the model at all specified model evaluation samples at a
        given emulator iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s
        sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the model evaluation samples.
        ext_sam_set : 1D or 2D :obj:`~numpy.ndarray` object
            Array containing the externally provided model evaluation samples.
        ext_mod_set : 1D or 2D :obj:`~numpy.ndarray` object
            Array containing the model outputs of all specified externally
            provided model evaluation samples.

        Generates
        ---------
        sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the model evaluation samples for emulator
            iteration `emul_i`.
        mod_set : 2D :obj:`~numpy.ndarray` object
            Array containing the model outputs of all specified model
            evaluation samples for emulator iteration `emul_i`.

        """

        # Log that evaluation of model samples is started
        logger = getCLogger('MODEL')
        logger.info("Evaluating model samples.")

        # Do model evaluations
        start_time = time()

        # Obtain number of samples
        n_sam = np.shape(sam_set)[0]

        # If there are any samples in sam_set, evaluate them in the model
        if n_sam:
            # Gather the data_idx from all MPI ranks on the controller
            data_idx_list = self._comm.gather(
                delist(self._emulator._data_idx[emul_i]), 0)

            # Flatten the received data_idx_list on the controller
            if self._is_controller:
                data_idx_flat = []
                data_idx_len = []
                for rank, data_idx_rank in enumerate(data_idx_list):
                    data_idx_len.append(len(data_idx_rank))
                    for data_idx in data_idx_rank:
                        data_idx_flat.append(data_idx)

            # Use dummy data_idx_flat on workers
            else:
                data_idx_flat = None

            # For sake of consistency, broadcast data_idx_flat to workers
            data_idx_flat = self._comm.bcast(data_idx_flat, 0)

            # Check who needs to call the model
            if self._is_controller or self._modellink._MPI_call:
                # Request all evaluation samples at once
                if self._modellink._multi_call:
                    mod_set = self._multi_call_model(emul_i, sam_set,
                                                     data_idx_flat).T

                # Request evaluation samples one-by-one
                else:
                    # Initialize mod_set
                    mod_set = np.zeros([self._modellink._n_data, n_sam])

                    # Loop over all requested evaluation samples
                    for i in range(n_sam):
                        mod_set[:, i] = self._call_model(emul_i, sam_set[i],
                                                         data_idx_flat)

        # Controller processing the received data values
        if self._is_controller:
            # Get end time
            end_time = time()-start_time

            # Check if ext_real_set and/or sam_set were provided
            if(ext_sam_set.shape[0] and sam_set.shape[0]):
                sam_set = np.concatenate([sam_set, ext_sam_set], axis=0)
                mod_set = np.concatenate([mod_set, ext_mod_set], axis=1)
                use_ext_real_set = 1
            elif(ext_sam_set.shape[0]):
                sam_set = ext_sam_set
                mod_set = ext_mod_set
                use_ext_real_set = 1
            else:
                use_ext_real_set = 0

            # Sent the specific mod_set parts to the corresponding workers
            s_idx = 0
            for rank, n_data in enumerate(data_idx_len):
                # Controller data must be saved last
                if not rank:
                    pass

                # Send the remaining parts to the workers
                else:
                    self._comm.send([sam_set, mod_set[s_idx:s_idx+n_data]],
                                    dest=rank, tag=777+rank)
                s_idx += n_data
            else:
                # If workers received their data, save controller data
                self._emulator._save_data(emul_i, None, {
                    'mod_real_set': [sam_set, mod_set[0:data_idx_len[0]],
                                     use_ext_real_set]})

        # Workers waiting for controller to send them their data values
        else:
            sam_set, mod_set = self._comm.recv(source=0, tag=777+self._rank)

            # Save all the data to the specific hdf5-files
            for i, lemul_s in enumerate(self._emulator._active_emul_s[emul_i]):
                self._emulator._save_data(emul_i, lemul_s, {
                    'mod_real_set': [sam_set, mod_set[i]]})

        # Controller finishing up
        if self._is_controller:
            # Log that this is finished
            self._save_statistics(emul_i, {
                'tot_model_eval_time': ['%.2f' % (end_time), 's'],
                'avg_model_eval_time': ['%.3g' % (end_time/n_sam), 's'],
                'MPI_comm_size_model': ['%s' % (self._size), '']})
            print("Finished evaluating model samples in %.2f seconds, "
                  "averaging %.3g seconds per model evaluation."
                  % (end_time, end_time/n_sam))
            logger.info("Finished evaluating model samples in %.2f seconds, "
                        "averaging %.3g seconds per model evaluation."
                        % (end_time, end_time/n_sam))

        # MPI Barrier
        self._comm.Barrier()

    # This function generates a large Latin Hypercube sample set to evaluate
    # the emulator at
    # TODO: Maybe make sure that n_sam_init samples are used for next iteration
    # This can be done by evaluating a 1000 samples in the emulator, check how
    # many survive and then use an LHD with the number of samples required to
    # let n_sam_init samples survive.
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_eval_sam_set(self, emul_i):
        """
        Generates an emulator evaluation sample set to be used for updating an
        emulator iteration. Currently uses the
        :func:`~e13tools.sampling.lhd` function.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        eval_sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the evaluation samples.

        """

        # Log about this
        logger = getLogger('EVAL_SAMS')

        # Obtain number of samples
        n_eval_sam = self._get_n_eval_sam(emul_i)

        # Create array containing all new samples to evaluate with emulator
        logger.info("Creating emulator evaluation sample set with size %s."
                    % (n_eval_sam))
        eval_sam_set = lhd(n_eval_sam, self._modellink._n_par,
                           self._modellink._par_rng, 'center',
                           self._criterion, 100,
                           constraints=self._emulator._sam_set[emul_i])
        logger.info("Finished creating sample set.")

        # Return it
        return(eval_sam_set)

    # This function performs an implausibility cut-off check on a given sample
    # TODO: Implement dynamic impl_cut
    @staticmethod
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _do_impl_check(obj, emul_i, uni_impl_val):
        """
        Performs an implausibility cut-off check on the provided implausibility
        values `uni_impl_val` at emulator iteration `emul_i`, using the
        impl_cut values given in `obj`.

        Parameters
        ----------
        obj : :obj:`~Pipeline` object or :obj:`~Projection` object
            Instance of the :class:`~Pipeline` class or :class:`~Projection`
            class.
        %(emul_i)s
        uni_impl_val : 1D array_like
            Array containing all univariate implausibility values corresponding
            to a certain parameter set for all data points.

        Returns
        -------
        result : bool
            *True* if check was successful, *False* if it was not.
        impl_cut_val : float
            Implausibility value at the first real implausibility cut-off.

        """

        # Sort impl_val to compare with the impl_cut list
        # TODO: Maybe use np.partition here?
        sorted_impl_val = np.flip(np.sort(uni_impl_val, axis=-1), axis=-1)

        # Save the implausibility value at the first real cut-off
        impl_cut_val = sorted_impl_val[obj._cut_idx[emul_i]]

        # Scan over all data points in this sample
        for impl_val, cut_val in zip(sorted_impl_val, obj._impl_cut[emul_i]):
            # If impl_cut is not 0 and impl_val is not below impl_cut, break
            if(cut_val != 0 and impl_val > cut_val):
                return(0, impl_cut_val)
        else:
            # If for-loop ended in a normal way, the check was successful
            return(1, impl_cut_val)

    # This is function 'I²(x)'
    # This function calculates the univariate implausibility values
    # TODO: Introduce check if emulator variance is much lower than other two
    # TODO: Alternatively, remove covariance calculations when this happens
    # TODO: Parameter uncertainty should be implemented at some point
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_uni_impl(self, emul_i, emul_s_seq, adj_exp_val, adj_var_val):
        """
        Calculates the univariate implausibility values at a given emulator
        iteration `emul_i` for specified expectation and variance values
        `adj_exp_val` and `adj_var_val`.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s
        adj_exp_val, adj_var_val : 1D array_like
            The adjusted expectation and variance values to calculate the
            univeriate implausibility for.

        Returns
        -------
        uni_impl_val : 1D :obj:`~numpy.ndarray` object
            Univariate implausibility value for every data point.

        """

        # Obtain model discrepancy variance
        md_var = self._get_md_var(emul_i, emul_s_seq)

        # Initialize empty univariate implausibility
        uni_impl_val_sq = np.zeros(len(emul_s_seq))

        # Calculate the univariate implausibility values
        for i, emul_s in enumerate(emul_s_seq):
            # Use the lower errors by default
            err_idx = 0

            # If adj_exp_val > data_val, use the upper error instead
            if(adj_exp_val[i] > self._emulator._data_val[emul_i][emul_s]):
                err_idx = 1

            # Calculate the univariate implausibility value
            uni_impl_val_sq[i] =\
                pow(adj_exp_val[i]-self._emulator._data_val[emul_i][emul_s],
                    2)/(adj_var_val[i]+md_var[i][err_idx] +
                        pow(self._emulator._data_err[emul_i][emul_s][err_idx],
                            2))

        # Take square root
        uni_impl_val = np.sqrt(uni_impl_val_sq)

        # Return it
        return(uni_impl_val)

    # This function calculates the model discrepancy variance
    # Basically takes all uncertainties of Sec. 3.1 of Vernon into account that
    # are not already in the emulator ([3] and [5])
    @docstring_substitute(emul_i=std_emul_i_doc, emul_s_seq=emul_s_seq_doc)
    def _get_md_var(self, emul_i, emul_s_seq):
        """
        Retrieves the model discrepancy variance, which includes all variances
        that are created by the model provided by the :obj:`~ModelLink`
        instance. This method tries to call the :meth:`~ModelLink.get_md_var`
        method, and assumes a default model discrepancy variance of ``1/6th``
        the data value if it cannot be called. If the data value space is not
        linear, then this default value is calculated such to reflect that.

        Parameters
        ----------
        %(emul_i)s
        %(emul_s_seq)s

        Returns
        -------
        var_md : 2D :obj:`~numpy.ndarray` object
            Variance of the model discrepancy.

        """

        # Obtain md variances
        # Try to use the user-defined md variances
        try:
            md_var = self._modellink.get_md_var(
                emul_i, delist(self._emulator._data_idx[emul_i]))

        # If it was not user-defined, use a default value
        except NotImplementedError:
            # Use factor 2 difference on 2 sigma as acceptable
            # Imagine that 2 sigma range is given if lower and upper are factor
            # 2 apart. This gives that sigma must be 1/6th of the data value

            # Create empty md_var list
            md_var = []

            # Loop over all data points and check their values spaces
            for data_val, data_spc in\
                zip(delist(self._emulator._data_val[emul_i]),
                    delist(self._emulator._data_spc[emul_i])):
                # If value space is linear, take 1/6ths of the data value
                if(data_spc == 'lin'):
                    md_var.append([pow(data_val/6, 2)]*2)
                # If value space is log10, take log10(5/6) and log10(7/6)
                elif(data_spc == 'log10'):
                    md_var.append([0.006269669725654501,
                                   0.0044818726418455815])
                # If value space is ln, take ln(5/6) and ln(7/6)
                elif(data_spc == 'ln'):
                    md_var.append([0.03324115007177121, 0.023762432091205918])
                # If value space is anything else
                else:
                    raise NotImplementedError

            # Make sure that md_var is a NumPy array
            md_var = np.array(md_var, ndmin=2)

        # If it was user-defined, check if the values are compatible
        else:
            # Make sure that md_var is a NumPy array
            md_var = np.array(md_var)

            # Check if md_var contains n_data values
            if not(md_var.shape[0] == self._emulator._n_data[emul_i]):
                raise ShapeError("Received array of model discrepancy "
                                 "variances 'md_var' has incorrect number of "
                                 "data points (%s != %s)!"
                                 % (md_var.shape[0],
                                    self._emulator._n_data[emul_i]))

            # Check if single or dual values were given
            if(md_var.ndim == 1):
                md_var = np.array([md_var]*2).T
            elif(md_var.shape[1] == 2):
                pass
            else:
                raise ShapeError("Received array of model discrepancy "
                                 "variances 'md_var' has incorrect number of "
                                 "values (%s != 2)!" % (md_var.shape[1]))

            # Check if all values are non-negative floats
            for i, value in enumerate(md_var):
                check_nneg_float(value[0], 'lower_md_var[%s]' % (i))
                check_nneg_float(value[1], 'upper_md_var[%s]' % (i))

        # Return it
        return(md_var)

    # This function completes the list of implausibility cut-offs
    @staticmethod
    def _get_impl_cut(obj, impl_cut, temp):
        """
        Generates the full list of impl_cut-offs from the incomplete, shortened
        `impl_cut` list and saves them in the given `obj`.

        Parameters
        ----------
        obj : :obj:`~Pipeline` object or :obj:`~Projection` object
            Instance of the :class:`~Pipeline` class or :class:`~Projection`
            class.
        impl_cut : 1D list
            Incomplete, shortened impl_cut-offs list provided during class
            initialization.
        temp : bool
            Whether the implausibility parameters should only be stored in
            memory (*True*) or should also be saved to HDF5 (*False*).

        Generates
        ---------
        impl_cut : 1D :obj:`~numpy.ndarray` object
            Full list containing the impl_cut-offs for all data points provided
            to the emulator.
        cut_idx : int
            Index of the first impl_cut-off in the impl_cut list that is not
            a wildcard.

        """

        # Log that impl_cut-off list is being acquired
        logger = getCLogger('INIT')
        logger.info("Generating full implausibility cut-off list.")

        # Complete the impl_cut list
        impl_cut[0] = check_nneg_float(impl_cut[0], 'impl_cut[0]')
        for i in range(1, len(impl_cut)):
            impl_cut[i] = check_nneg_float(impl_cut[i], 'impl_cut[%s]' % (i))
            if(impl_cut[i] == 0):
                impl_cut[i] = impl_cut[i-1]
            elif(impl_cut[i-1] != 0 and impl_cut[i] > impl_cut[i-1]):
                raise ValueError("Cut-off %s is higher than cut-off %s "
                                 "(%s > %s)" % (i, i-1, impl_cut[i],
                                                impl_cut[i-1]))

        # Get the index identifying where the first real impl_cut is
        for i, impl in enumerate(impl_cut[:obj._emulator._n_data_tot[-1]]):
            if(impl != 0):
                cut_idx = i
                break
        else:
            raise ValueError("No non-wildcard implausibility cut-off is "
                             "provided!")

        # Save both impl_cut and cut_idx
        if temp:
            # If they need to be stored temporarily
            obj._impl_cut.append(np.array(impl_cut))
            obj._cut_idx.append(cut_idx)
        else:
            obj._save_data({
                'impl_cut': [np.array(impl_cut), cut_idx]})

        # Log end of process
        logger.info("Finished generating implausibility cut-off list.")

    # This function reads in the impl_cut list from the PRISM parameters file
    # TODO: Make impl_cut dynamic
    def _get_impl_par(self, temp):
        """
        Reads in the impl_cut list and other parameters for implausibility
        evaluations from the PRISM parameters file and saves them in the last
        emulator iteration.

        Parameters
        ----------
        temp : bool
            Whether the implausibility parameters should only be stored in
            memory (*True*) or should also be saved to HDF5 (*False*).

        Generates
        ---------
        impl_cut : 1D :obj:`~numpy.ndarray` object
            Full list containing the impl_cut-offs for all data points provided
            to the emulator.
        cut_idx : int
            Index of the first impl_cut-off in the impl_cut list that is not
            0.

        """

        # Controller only
        if self._is_controller:
            # Do some logging
            logger = getCLogger('INIT')
            logger.info("Obtaining implausibility analysis parameters.")

            # Obtaining default pipeline parameter dict
            par_dict = self._get_default_parameters()

            # Read in data from provided PRISM parameters file
            if self._prism_file is not None:
                pipe_par = np.genfromtxt(self._prism_file, dtype=(str),
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
            self._get_impl_cut(
                self, list(float(impl_cut) for impl_cut in impl_cut_str), temp)

            # Finish logging
            logger.info("Finished obtaining implausibility analysis "
                        "parameters.")

    # This function processes an externally provided real_set
    # TODO: Perform maximin/mincorr analysis on provided samples?
    def _get_ext_real_set(self, ext_real_set):
        """
        Processes an externally provided model realization set `ext_real_set`,
        containing the used sample set and the corresponding data value set.

        Parameters
        ----------
        ext_real_set : list, dict or None
            List of arrays containing an externally calculated set of model
            evaluation samples and its data values, a dict with keys
            ``[sam_set, mod_set]`` containing these arrays or *None* if no
            external set needs to be used.

        Returns
        -------
        ext_sam_set : 1D or 2D :obj:`~numpy.ndarray` object
            Array containing the externally provided model evaluation samples.
        ext_mod_set : 1D or 2D :obj:`~numpy.ndarray` object
            Array containing the model outputs of all specified externally
            provided model evaluation samples.

        """

        # If no ext_real_set is provided, return empty arrays without logging
        if ext_real_set is None:
            return(np.array([]), np.array([]))

        # Do some logging
        logger = getCLogger('INIT')
        logger.info("Processing externally provided model realization set.")

        # If a list is given
        if isinstance(ext_real_set, list):
            # Check if ext_real_set contains 2 elements
            if(len(ext_real_set) != 2):
                logger.error("Input argument 'ext_real_set' is not of length "
                             "2!")
                raise ShapeError("Input argument 'ext_real_set' is not of "
                                 "length 2!")

            # Try to extract ext_sam_set and ext_mod_set
            try:
                ext_sam_set = ext_real_set[0]
                ext_mod_set = ext_real_set[1]
            except Exception as error:
                logger.error("Input argument 'ext_real_set' is invalid! (%s)"
                             % (error))
                raise InputError("Input argument 'ext_real_set' is invalid! "
                                 "(%s)" % (error))

        # If a dict is given
        elif isinstance(ext_real_set, dict):
            # Check if ext_real_set contains correct keys
            if 'sam_set' not in ext_real_set.keys():
                logger.error("Input argument 'ext_real_set' does not contain "
                             "key 'sam_set'!")
                raise KeyError("Input argument 'ext_real_set' does not contain"
                               " key 'sam_set'!")
            if 'mod_set' not in ext_real_set.keys():
                logger.error("Input argument 'ext_real_set' does not contain "
                             "key 'mod_set'!")
                raise KeyError("Input argument 'ext_real_set' does not contain"
                               " key 'mod_set'!")

            # Try to extract ext_sam_set and ext_mod_set
            try:
                ext_sam_set = ext_real_set['sam_set']
                ext_mod_set = ext_real_set['mod_set']
            except Exception as error:
                logger.error("Input argument 'ext_real_set' is invalid! (%s)"
                             % (error))
                raise InputError("Input argument 'ext_real_set' is invalid! "
                                 "(%s)" % (error))

        # If anything else is given
        else:
            logger.error("Input argument 'ext_real_set' is invalid!")
            raise InputError("Input argument 'ext_real_set' is invalid!")

        # Check if ext_sam_set and ext_mod_set can be converted to NumPy arrays
        try:
            ext_sam_set = np.array(ext_sam_set, ndmin=2)
            ext_mod_set = np.array(ext_mod_set, ndmin=2)
        except Exception as error:
            logger.error("Input argument 'ext_real_set' is invalid! (%s)"
                         % (error))
            raise InputError("Input argument 'ext_real_set' is invalid! (%s)"
                             % (error))

        # Check if ext_sam_set and ext_mod_set have correct shapes
        if not(ext_sam_set.shape[1] == self._modellink._n_par):
            logger.error("External sample set has incorrect number of "
                         "parameters (%s != %s)!"
                         % (ext_sam_set.shape[1], self._modellink._n_par))
            raise ShapeError("External sample set has incorrect number of "
                             "parameters (%s != %s)!"
                             % (ext_sam_set.shape[1], self._modellink._n_par))
        if not(ext_mod_set.shape[1] == self._modellink._n_data):
            logger.error("External model output set has incorrect number of "
                         "data values (%s != %s)!"
                         % (ext_mod_set.shape[1], self._modellink._n_data))
            raise ShapeError("External model output set has incorrect number "
                             "of data values (%s != %s)!"
                             % (ext_mod_set.shape[1], self._modellink._n_data))
        if not(ext_sam_set.shape[0] == ext_mod_set.shape[0]):
            logger.error("External sample and model output sets do not contain"
                         " the same number of samples (%s != %s)!"
                         % (ext_sam_set.shape[0], ext_mod_set.shape[0]))

        # Check if ext_sam_set and ext_mod_set solely contain floats
        for i, (par_set, mod_out) in enumerate(zip(ext_sam_set, ext_mod_set)):
            for j, (par, out) in enumerate(zip(par_set, mod_out)):
                check_float(par, 'ext_sam_set[%s, %s]' % (i, j))
                check_float(out, 'ext_mod_set[%s, %s]' % (i, j))

        # Check if all samples are within parameter space
        lower_bnd = self._modellink._par_rng[:, 0]
        upper_bnd = self._modellink._par_rng[:, 1]
        for i, par_set in enumerate(ext_sam_set):
            if not(((lower_bnd <= par_set)*(par_set <= upper_bnd)).all()):
                logger.error("External sample set contains a sample outside of"
                             " parameter space at index %s!" % (i))
                raise ValueError("External sample set contains a sample "
                                 "outside of parameter space at index %s!"
                                 % (i))

        # Log that processing has been finished
        logger.info("Finished processing externally provided model realization"
                    " set of size %s." % (ext_sam_set.shape[0]))

        # If all checks are passed, return ext_sam_set and ext_mod_set
        return(ext_sam_set, ext_mod_set.T)

    # This function analyzes the emulator for given sam_set using code snippets
    # TODO: Determine whether using arrays or lists is faster
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _analyze_sam_set(self, obj, emul_i, sam_set, pre_code, eval_code,
                         anal_code, post_code, exit_code):
        """
        Analyzes a provided set of emulator evaluation samples `sam_set` at a
        given emulator iteration `emul_i`, using the impl_cut values given in
        `obj`. The provided code snippets `pre_code`; `eval_code`; `anal_code`;
        `post_code` and `exit_code` are executed using Python's :func:`~exec`
        function at specific points during the analysis.

        Parameters
        ----------
        obj : :obj:`~Pipeline` object or :obj:`~Projection` object
            Instance of the :class:`~Pipeline` class or :class:`~Projection`
            class.
        %(emul_i)s
        sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing model parameter value sets to be evaluated in all
            emulator systems in emulator iteration `emul_i`.
        pre_code : str or code object
            Code snippet to be executed before the analysis of `sam_set`
            starts.
        eval_code : str or code object
            Code snippet to be executed after the evaluation of each sample in
            `sam_set`.
        anal_code : str or code object
            Code snippet to be executed after the analysis of each sample in
            `sam_set`.
        post_code : str or code object
            Code snippet to be executed after the analysis of `sam_set` ends.
        exit_code : str or code object
            Code snippet to be executed before returning the results of the
            analysis of `sam_set`.

        Returns
        -------
        results : object
            The object that is assigned to :attr:`~Pipeline.results`, which is
            defaulted to *None* if no code snippet changes it. Preferably, the
            execution of `post_code` and/or `exit_code` modifies
            :attr:`~Pipeline.results`.

        Notes
        -----
        If any of the code snippets is provided as a string, it will be
        compiled before being executed every time the :func:`~exec` function is
        called. Providing a code object instead will simply allow the code
        snippet to be executed without any compilations.

        """

        # Determine number of samples
        n_sam = sam_set.shape[0]

        # Start logging
        logger = getCLogger('ANALYZE_SS')
        logger.info("Starting analysis of evaluation sample set of size %s."
                    % (n_sam))

        # Make a filled bool list containing which samples are plausible
        impl_check = np.ones(n_sam, dtype=int)

        # Make a list of plausible sample indices
        sam_idx_full = np.array(list(range(n_sam)))
        sam_idx = sam_idx_full[impl_check == 1]

        # Set the results property to None
        self.results = None

        # Execute the pre_code snippet
        exec(pre_code)

        # If the default emulator type is used
        if(self._emulator._emul_type == 'default'):
            # Analyze sam_set in every emulator iteration
            for i in range(1, emul_i+1):
                # Determine the number of samples
                n_sam = len(sam_idx)

                # Log that this iteration is being evaluated
                logger.info("Analyzing evaluation sample set of size %s in "
                            "emulator iteration %s." % (n_sam, i))

                # Determine which samples should be used in this iteration
                eval_sam_set = sam_set[sam_idx]

                # Determine the active emulator systems on every rank
                active_emul_s = self._emulator._active_emul_s[i]

                # Make empty uni_impl_vals list
#                uni_impl_vals = np.zeros([n_sam, self._emulator._n_data[i]])
                uni_impl_vals = []

                # Loop over all samples in sam_set
                for j, par_set in enumerate(eval_sam_set):
                    # Evaluate par_set
                    adj_val = self._emulator._evaluate(i, active_emul_s,
                                                       par_set)

                    # Calculate univariate implausibility value
#                    uni_impl_val[j] = self._get_uni_impl(i, active_emul_s,
#                                                         *adj_val)
                    uni_impl_val = self._get_uni_impl(i, active_emul_s,
                                                      *adj_val)
                    uni_impl_vals.append(uni_impl_val)

                    # Execute the eval_code snippet
                    exec(eval_code)

                # Gather the results on the controller after evaluating
#                uni_impl_vals_list = self._comm.gather(uni_impl_vals, 0)
                uni_impl_vals_list = self._comm.gather(np.array(uni_impl_vals), 0)

                # Controller performs implausibility analysis
                if self._is_controller:
                    # Convert uni_impl_vals_list to an array
                    uni_impl_vals_array =\
                        np.concatenate(*[uni_impl_vals_list], axis=1)

                    # Perform implausibility cutoff check on all elements
                    for j, uni_impl_val in enumerate(uni_impl_vals_array):
                        impl_check_val, impl_cut_val =\
                            self._do_impl_check(obj, i, uni_impl_val)

                        # Add impl_check_val to the impl_check list
                        impl_check[sam_idx[j]] = impl_check_val

                        # Execute the anal_code snippet
                        exec(anal_code)

                    # Modify sam_idx with those that are still plausible
                    sam_idx = sam_idx_full[impl_check == 1]

                # MPI Barrier
                self._comm.Barrier()

                # Broadcast the updated sam_idx to the workers if not last i
                if(i != emul_i):
                    sam_idx = self._comm.bcast(sam_idx, 0)

                # Check that sam_idx is still holding plausible samples
                if not len(sam_idx):
                    break

        # If any other emulator type is used
        else:
            raise NotImplementedError

        # Execute the post_code snippet
        exec(post_code)

        # Log that analysis is finished
        logger.info("Finished analyzing evaluation sample set.")

        # Controller sending the results back
        if self._is_controller:
            # Execute the exit_code snippet
            exec(exit_code)

            # Retrieve the results and delete the attribute
            results = self.results
            del self.results

            # Return the results
            return(results)


# %% VISIBLE CLASS METHODS
    # This function analyzes the emulator and determines the plausible regions
    # TODO: Implement check if impl_idx is big enough to be used in next emul_i
    def analyze(self):
        """
        Analyzes the emulator system at the last emulator iteration for a large
        number of emulator evaluation samples. All samples that survive the
        implausibility checks, are used in the construction of the next
        emulator iteration.

        Generates
        ---------
        impl_sam : 2D :obj:`~numpy.ndarray` object
            Array containing all emulator evaluation samples that survived the
            implausibility checks.
        prc : bool
            Bool indicating whether or not plausible regions have been found
            during this analysis.

        """

        # Get emul_i
        emul_i = self._emulator._get_emul_i(None, True)

        # Begin logging
        logger = getCLogger('ANALYZE')

        # Begin analyzing
        logger.info("Analyzing emulator system at iteration %s." % (emul_i))

        # Controller only
        if self._is_controller:
            # Save current time
            start_time1 = time()

            # Get the impl_cut list
            self._get_impl_par(False)

            # Create an emulator evaluation sample set
            eval_sam_set = self._get_eval_sam_set(emul_i)
            n_eval_sam = eval_sam_set.shape[0]

        # Remaining workers get dummy eval_sam_set
        else:
            eval_sam_set = []

        # Broadcast eval_sam_set to workers
        eval_sam_set = self._comm.bcast(eval_sam_set, 0)

        # Define the various code snippets
        pre_code = compile("", '<string>', 'exec')
        eval_code = compile("", '<string>', 'exec')
        anal_code = compile("", '<string>', 'exec')
        post_code = compile("", '<string>', 'exec')
        exit_code = compile("self.results = impl_check", '<string>', 'exec')

        # Combine code snippets into a tuple
        exec_code = (pre_code, eval_code, anal_code, post_code, exit_code)

        # Save current time again
        start_time2 = time()

        # Analyze eval_sam_set
        impl_check = self._analyze_sam_set(self, emul_i, eval_sam_set,
                                           *exec_code)

        # Controller finishing up
        if self._is_controller:
            # Obtain some timers
            end_time = time()
            time_diff_total = end_time-start_time1
            time_diff_eval = end_time-start_time2

            # Save the results
            self._save_data({
                'impl_sam': eval_sam_set[impl_check == 1],
                'n_eval_sam': n_eval_sam})

            # Save statistics about anal time, evaluation speed, par_space
            avg_eval_rate = n_eval_sam/time_diff_eval
            par_space_rem = (sum(impl_check)/n_eval_sam)*100
            self._save_statistics(emul_i, {
                'tot_analyze_time': ['%.2f' % (time_diff_total), 's'],
                'avg_emul_eval_rate': ['%.2f' % (avg_eval_rate), '1/s'],
                'par_space_remaining': ['%.3g' % (par_space_rem), '%'],
                'MPI_comm_size_anal': ['%s' % (self._size), '']})

            # Log that analysis has been finished and save their statistics
            print("Finished analysis of emulator system in %.2f seconds, "
                  "averaging %.2f emulator evaluations per second."
                  % (time_diff_total, n_eval_sam/time_diff_eval))
            print("There is %.3g%% of parameter space remaining."
                  % ((sum(impl_check)/n_eval_sam)*100))
            logger.info("Finished analysis of emulator system in %.2f seconds,"
                        "averaging %.2f emulator evaluations per second."
                        % (time_diff_total, n_eval_sam/time_diff_eval))
            logger.info("There is %.3g%% of parameter space remaining."
                        % ((sum(impl_check)/n_eval_sam)*100))

        # Display details about current state of pipeline
        self.details()

    # This function constructs a specified iteration of the emulator system
    # TODO: Make time and RAM cost plots
    # TODO: Fix the timers for interrupted constructs
    @docstring_substitute(emul_i=call_emul_i_doc)
    def construct(self, emul_i=None, analyze=True, ext_init_set=None,
                  force=False):
        """
        Constructs the emulator at the specified emulator iteration `emul_i`,
        and performs an implausibility analysis on the emulator systems right
        afterward if requested (:meth:`~analyze`).

        Optional
        --------
        %(emul_i)s
        analyze : bool. Default: True
            Bool indicating whether or not to perform an analysis after the
            specified emulator iteration has been successfully constructed,
            which is required for constructing the next iteration.
        ext_init_set : list, dict or None. Default: None
            List of arrays containing an externally calculated set of initial
            model evaluation samples and its data values, a dict with keys
            ``[sam_set, mod_set]`` containing these arrays or *None* if no
            external realization set needs to be used.
            This parameter has no use for `emul_i` != 1.
        force : bool. Default: False
            Controls what to do if the specified emulator iteration `emul_i`
            already (partly) exists.
            If *False*, finish construction of the specified iteration or skip
            it if already finished.
            If *True*, reconstruct the specified iteration entirely.

        Generates
        ---------
        A new HDF5-group with the emulator iteration value as its name, in the
        loaded emulator file, containing emulator data required for this
        emulator iteration.

        Notes
        -----
        Using an emulator iteration that has been (partly) constructed before,
        will finish that construction or skip construction if already finished
        when `force` = *False*; or it will delete that and all following
        iterations, and reconstruct the specified iteration when `force` =
        *True*. Using `emul_i` = 1 and `force` = *True* is equivalent to
        reconstructing the entire emulator system.

        If no implausibility analysis is requested, then the implausibility
        parameters are read in from the PRISM parameters file and temporarily
        stored in memory in order to enable the usage of the :meth:`~evaluate`
        method.

        """

        # Log that a new emulator iteration is being constructed
        logger = getCLogger('CONSTRUCT')

        # Set emul_i correctly
        emul_i = self._emulator._get_emul_i(emul_i, False)

        # Only the controller should run this
        if self._is_controller:
            # Save current time
            start_time = time()

            # Check if force-parameter received a bool
            force = check_bool(force, 'force')

            # Check if iteration was interrupted or not, or if force is True
            logger.info("Checking state of emulator iteration %s." % (emul_i))
            try:
                # If force is True, reconstruct full iteration
                if force:
                    logger.info("Emulator iteration %s has been requested to "
                                "be (re)constructed." % (emul_i))
                    c_from_start = 1

                # If interrupted at start, reconstruct full iteration
                elif('mod_real_set' in self._emulator._ccheck[emul_i]):
                    logger.info("Emulator iteration %s does not contain "
                                "evaluated model realization data. Will be "
                                "constructed from start." % (emul_i))
                    c_from_start = 1

                # If already finished, skip everything
                elif(emul_i <= self._emulator._emul_i):
                    logger.info("Emulator iteration %s has already been fully "
                                "constructed. Skipping construction process."
                                % (emul_i))
                    print("Emulator iteration %s has already been fully "
                          "constructed. Skipping construction process."
                          % (emul_i))
                    c_from_start = None

                # If interrupted midway, do not reconstruct full iteration
                else:
                    logger.info("Construction of emulator iteration %s was "
                                "interrupted. Continuing from point of "
                                "interruption." % (emul_i))
                    c_from_start = 0

            # If never constructed before, construct full iteration
            except IndexError:
                logger.info("Emulator iteration %s has not been constructed."
                            % (emul_i))
                c_from_start = 1

        # Remaining workers
        else:
            c_from_start = None

        # Check if analyze-parameter received a bool
        analyze = check_bool(analyze, 'analyze')

        # Broadcast construct_emul_i to workers
        c_from_start = self._comm.bcast(c_from_start, 0)

        # If iteration is already finished, show the details
        if c_from_start is None:
            self.details(emul_i)
            return

        # If iteration needs to be constructed completely, create/prepare it
        # Log that construction of emulator iteration is being started
        logger.info("Starting construction of emulator iteration %s."
                    % (emul_i))

        # If iteration needs to be constructed from the beginning
        if c_from_start:
            # Initialize data sets
            add_sam_set = []
            ext_sam_set = []
            ext_mod_set = []

            # If this is the first iteration
            if(emul_i == 1):
                # Controller only
                if self._is_controller:
                    # Process ext_init_set
                    ext_sam_set, ext_mod_set =\
                        self._get_ext_real_set(ext_init_set)

                    # Obtain number of externally provided model realizations
                    n_ext_sam = np.shape(ext_sam_set)[0]

                    # Create a new emulator
                    self._emulator._create_new_emulator()

                    # Reload the data
                    self._load_data()

                    # Create initial set of model evaluation samples
                    n_sam_init = max(0, self._n_sam_init-n_ext_sam)
                    if n_sam_init:
                        logger.info("Creating initial model evaluation sample "
                                    "set of size %s." % (n_sam_init))
                        add_sam_set = lhd(n_sam_init, self._modellink._n_par,
                                          self._modellink._par_rng, 'center',
                                          self._criterion,
                                          constraints=ext_sam_set)
                        logger.info("Finished creating initial sample set.")
                    else:
                        add_sam_set = np.array([])

                # Remaining workers
                else:
                    # Create a new emulator
                    self._emulator._create_new_emulator()

            # If this is any other iteration
            else:
                # Controller only
                if self._is_controller:
                    # Get dummy ext_real_set
                    ext_sam_set, ext_mod_set = self._get_ext_real_set(None)

                    # Check if previous iteration was analyzed, do so if not
                    if not self._n_eval_sam[emul_i-1]:
                        # Let workers know that emulator needs analyzing
                        for rank in range(1, self._size):
                            self._comm.send(1, dest=rank, tag=999+rank)

                        # Analyze previous iteration
                        logger.info("Previous emulator iteration has not been "
                                    "analyzed. Performing analysis first.")
                        self.analyze()
                    else:
                        # If not, let workers know
                        for rank in range(1, self._size):
                            self._comm.send(0, dest=rank, tag=999+rank)

                    # Check if a new emulator iteration can be constructed
                    if(not self._prc and self._emulator._emul_i != emul_i):
                        logger.error("No plausible regions were found in the "
                                     "analysis of the previous emulator "
                                     "iteration. Construction is not "
                                     "possible!")
                        raise RequestError("No plausible regions were found in"
                                           " the analysis of the previous "
                                           "emulator iteration. Construction "
                                           "is not possible!")

                    # Make the emulator prepare for a new iteration
                    reload = self._emulator._prepare_new_iteration(emul_i)

                    # Make sure the correct pipeline data is loaded in
                    if reload:
                        self._load_data()

                    # Obtain additional sam_set
                    add_sam_set = self._impl_sam

                # Remaining workers
                else:
                    # Listen for calls from controller for analysis
                    do_analyze = self._comm.recv(source=0, tag=999+self._rank)

                    # If previous iteration needs analyzing, call analyze()
                    if do_analyze:
                        self.analyze()

                    # Make the emulator prepare for a new iteration
                    self._emulator._prepare_new_iteration(emul_i)

            # MPI Barrier to free up workers
            self._comm.Barrier()

            # Broadcast add_sam_set to workers
            add_sam_set = self._comm.bcast(add_sam_set, 0)

            # Obtain corresponding set of model evaluations
            self._evaluate_model(emul_i, add_sam_set, ext_sam_set, ext_mod_set)

        # Construct emulator
        self._emulator._construct_iteration(emul_i)

        # Only controller
        if self._is_controller:
            # Save that emulator system has not been analyzed yet
            self._save_data({
                'impl_sam': [],
                'n_eval_sam': 0})

            # Log that construction has been completed
            time_diff_total = time()-start_time
            self._save_statistics(emul_i, {
                'tot_construct_time': ['%.2f' % (time_diff_total), 's']})
            print("Finished construction of emulator system in %.2f seconds."
                  % (time_diff_total))
            logger.info("Finished construction of emulator system in %.2f "
                        "seconds." % (time_diff_total))

        # Analyze the emulator system if requested
        if analyze:
            self.analyze()
        else:
            self._get_impl_par(True)
            self.details(emul_i)

    # This function allows one to obtain the pipeline details/properties
    # TODO: Allow the viewing of the entire polynomial function in SymPy
    # TODO: Rewrite to add MPI compatibility
    @docstring_substitute(emul_i=user_emul_i_doc)
    def details(self, emul_i=None):
        """
        Prints the details/properties of the currently loaded pipeline instance
        at given emulator iteration `emul_i`. See ``Notes`` for detailed
        descriptions of all printed properties.

        Optional
        --------
        %(emul_i)s

        Notes
        -----
        HDF5-file name
            The relative path to the loaded HDF5-file starting at `root_dir`,
            which consists of `working_dir` and `hdf5_file`.
        Emulator type
            The type of this emulator system, corresponding to the provided
            `emul_type` during :class:`~Pipeline` initialization.
        ModelLink subclass
            Name of the :class:`~ModelLink` subclass used to construct this
            emulator system.
        Emulation method
            Indicates the combination of regression and Gaussian emulation
            methods that have been used for this emulator system.
        Mock data used?
            Whether or not mock data has been used to construct this emulator
            system. If so, the printed estimates for all model parameters are
            the parameter values used to create the mock data.

        Emulator iteration
            The iteration of the emulator system this details overview is
            about. By default, this is the last constructed iteration.
        Construction completed?
            Whether or not the construction of this emulator iteration is
            completed. If not, the missing components are listed and the
            remaining information of this iteration is not printed.
        Plausible regions?
            Whether or not plausible regions have been found during the
            analysis of this emulator iteration. If no analysis has been done
            yet, "N/A" will be printed.
        Projections available?
            Whether or not projections have been created for this emulator
            iteration. If projections are available and analysis has been done,
            but with different implausibility cut-offs, a "desync" note is
            added. Also prints number of available projections versus maximum
            number of projections in brackets.

        # of model evaluation samples
            The total number of model evaluation samples used to construct all
            emulator iterations up to this iteration, with the number for every
            individual iteration in brackets.
        # of plausible/analyzed samples
            The number of emulator evaluation samples that passed the
            implausibility check out of the total number of analyzed samples in
            this emulator iteration.
            This is the number of model evaluation samples that was/will be
            used for the construction of the next emulator iteration.
            If no analysis has been done, the numbers show up as "-".
        %% of parameter space remaining
            The percentage of the total number of analyzed samples that passed
            the implausibility check in this emulator iteration.
            If no analysis has been done, the number shows up as "-".
        # of active/total parameters
            The number of model parameters that was considered active during
            the construction of this emulator iteration, compared to the total
            number of model parameters defined in the used :class:`~ModelLink`
            subclass.
        # of emulated data points
            The number of data points that have been emulated in this
            emulator iteration.

        Parameter space
            Lists the name, lower and upper value boundaries and estimate (if
            provided) of all model parameters defined in the used
            :class:`~ModelLink` subclass. An asterisk is printed in front of
            the parameter name if this model parameter was considered active
            during the construction of this emulator iteration. A question mark
            is used instead if the construction of this emulator iteration is
            not finished.

        """

        # Define details logger
        logger = getCLogger("DETAILS")
        logger.info("Collecting details about current pipeline instance.")

        # Check if last emulator iteration is finished constructing
        if(len(self._emulator._ccheck[-1]) == 0 or
           delist(self._emulator._ccheck[-1]) == []):
            ccheck_flag = 1
        else:
            ccheck_flag = 0

        # Gather the ccheck_flags on all ranks to see if all are finished
        ccheck_flag = np.all(self._comm.allgather(ccheck_flag))

        # Check what kind of hdf5-file was provided
        try:
            if ccheck_flag:
                emul_i = self._emulator._get_emul_i(emul_i, True)
            else:
                emul_i = self._emulator._get_emul_i(emul_i, False)
        except RequestError:
            # MPI Barrier
            self._comm.Barrier()
            return
        else:
            # Gather ccheck information from all MPI ranks on the controller
            ccheck_list = self._comm.gather(self._emulator._ccheck[emul_i], 0)

        # Controller only
        if self._is_controller:
            # Flatten the received ccheck_list
            ccheck_flat = [[] for _ in range(self._emulator._n_emul_s_tot)]
            for ccheck_iter in ccheck_list[0][self._emulator._n_emul_s:]:
                ccheck_flat.append(ccheck_iter)
            for rank, ccheck_rank in enumerate(ccheck_list):
                for emul_s, ccheck in zip(self._emulator._emul_s_to_core[rank],
                                          ccheck_rank):
                    ccheck_flat[emul_s] = ccheck

            # Get max lengths of various strings for parameter section
            name_len =\
                max([len(par_name) for par_name in self._modellink._par_name])
            lower_len =\
                max([len(str(i)) for i in self._modellink._par_rng[:, 0]])
            upper_len =\
                max([len(str(i)) for i in self._modellink._par_rng[:, 1]])
            est_len =\
                max([len('%.5f' % (i)) for i in self._modellink._par_est
                     if i is not None])

            # Open hdf5-file
            with PRISM_File('r', None) as file:
                # Check if projection data is available
                try:
                    file['%s/proj_hcube' % (emul_i)]
                except KeyError:
                    proj = 0
                    n_proj = 0

                # If projection data is available
                else:
                    n_proj = len(file['%s/proj_hcube' % (emul_i)].keys())
                    proj_impl_cut =\
                        file['%s/proj_hcube' % (emul_i)].attrs['impl_cut']
                    proj_cut_idx =\
                        file['%s/proj_hcube' % (emul_i)].attrs['cut_idx']

                    # Check if projections were made with the same impl_cut
                    try:
                        # If it was, projections are synced
                        if((proj_impl_cut == self._impl_cut[emul_i]).all()
                           and proj_cut_idx == self._cut_idx[emul_i]):
                            proj = 1

                        # If not, projections are desynced
                        else:
                            proj = 2

                    # If analysis was never done, projections are synced
                    except IndexError:
                        proj = 1

            # Log file being closed
            logger.info("Finished collecting details about current pipeline "
                        "instance.")

            # Obtain number of model parameters
            n_par = self._modellink._n_par

            # Obtain number of data points
            n_data = self._emulator._n_data_tot[emul_i]

            # Obtain number of emulator systems
            n_emul_s = self._emulator._n_emul_s_tot

            # Determine the relative path to the master HDF5-file
            hdf5_rel_path = path.relpath(self._hdf5_file, self._root_dir)

            # Set width of detail names
            width = 31

            # PRINT DETAILS
            # HEADER
            print("\nPIPELINE DETAILS")
            print("="*width)

            # GENERAL
            print("\nGENERAL")
            print("-"*width)

            # General details about loaded emulator system
            print("{0: <{1}}\t'{2}'".format("HDF5-file name", width,
                                            hdf5_rel_path))
            print("{0: <{1}}\t'{2}'".format("Emulator type", width,
                                            self._emulator._emul_type))
            print("{0: <{1}}\t{2}".format("ModelLink subclass", width,
                                          self._modellink._name))
            if(self._emulator._method.lower() == 'regression'):
                print("{0: <{1}}\t{2}".format("Emulation method", width,
                                              "Regression"))
            elif(self._emulator._method.lower() == 'gaussian'):
                print("{0: <{1}}\t{2}".format("Emulation method", width,
                                              "Gaussian"))
            elif(self._emulator._method.lower() == 'full'):
                print("{0: <{1}}\t{2}".format("Emulation method", width,
                                              "Regression + Gaussian"))
            print("{0: <{1}}\t{2}".format("Mock data used?", width,
                                          "Yes" if self._emulator._use_mock
                                          else "No"))

            # ITERATION DETAILS
            print("\nITERATION")
            print("-"*width)

            # Emulator iteration corresponding to this details overview
            print("{0: <{1}}\t{2}".format("Emulator iteration", width, emul_i))

            # Availability flags
            # If this iteration is fully constructed, print flags and numbers
            if(delist(ccheck_flat) == []):
                # Determine the number of (active) parameters
                n_active_par = len(self._emulator._active_par[emul_i])

                # Calculate the maximum number of projections
                n_proj_max = nCr(n_active_par, 1 if(n_par == 2) else 2)

                print("{0: <{1}}\t{2}".format("Construction completed?", width,
                                              "Yes"))
                if not self._n_eval_sam[emul_i]:
                    print("{0: <{1}}\t{2}".format("Plausible regions?", width,
                                                  "N/A"))
                else:
                    print("{0: <{1}}\t{2}".format(
                        "Plausible regions?", width,
                        "Yes" if self._prc else "No"))
                if not proj:
                    print("{0: <{1}}\t{2}".format("Projections available?",
                                                  width, "No"))
                else:
                    print("{0: <{1}}\t{2} ({3}/{4})".format(
                        "Projections available?", width,
                        "Yes%s" % ("" if proj == 1 else ", desynced"),
                        n_proj, n_proj_max))
                print("-"*width)

                # Number details
                if(self._emulator._emul_type == 'default'):
                    print("{0: <{1}}\t{2} ({3})".format(
                        "# of model evaluation samples", width,
                        sum(self._emulator._n_sam[1:emul_i+1]),
                        self._emulator._n_sam[1:emul_i+1]))
                else:
                    raise NotImplementedError
                if not self._n_eval_sam[emul_i]:
                    print("{0: <{1}}\t{2}/{3}".format(
                        "# of plausible/analyzed samples", width, "-", "-"))
                    print("{0: <{1}}\t{2}".format(
                        "% of parameter space remaining", width, "-"))
                else:
                    print("{0: <{1}}\t{2}/{3}".format(
                        "# of plausible/analyzed samples", width,
                        self._n_impl_sam[emul_i], self._n_eval_sam[emul_i]))
                    print("{0: <{1}}\t{2:.3g}%".format(
                        "% of parameter space remaining", width,
                        (self._n_impl_sam[emul_i] /
                         self._n_eval_sam[emul_i])*100))
                print("{0: <{1}}\t{2}/{3}".format(
                    "# of active/total parameters", width,
                    n_active_par, n_par))
                print("{0: <{1}}\t{2}".format("# of emulated data points",
                                              width, n_data))

            # If not, then print which components are still missing
            else:
                print("{0: <{1}}\t{2}".format("Construction completed?", width,
                                              "No"))
                print("  - {0: <{1}}\t{2}".format(
                    "'mod_real_set'?", width-4,
                    "No" if 'mod_real_set' in ccheck_flat else "Yes"))

                # Check if all active parameters have been determined
                ccheck_i = [i for i in range(n_emul_s) if
                            'active_par_data' in ccheck_flat[i]]
                print("  - {0: <{1}}\t{2}".format(
                    "'active_par'?", width-4, "No (%s)" % (ccheck_i) if
                    len(ccheck_i) else "Yes"))

                # Check if all regression processes have been done
                if self._emulator._method.lower() in ('regression', 'full'):
                    ccheck_i = [i for i in range(n_emul_s) if
                                'regression' in ccheck_flat[i]]
                    print("  - {0: <{1}}\t{2}".format(
                        "'regression'?", width-4, "No (%s)" % (ccheck_i) if
                        len(ccheck_i) else "Yes"))

                # Check if all prior_exp_sam_sets have been determined
                ccheck_i = [i for i in range(n_emul_s) if
                            'prior_exp_sam_set' in ccheck_flat[i]]
                print("  - {0: <{1}}\t{2}".format(
                    "'prior_exp_sam_set'?", width-4, "No (%s)" % (ccheck_i) if
                    len(ccheck_i) else "Yes"))

                # Check if all covariance matrices have been determined
                ccheck_i = [i for i in range(n_emul_s) if
                            'cov_mat' in ccheck_flat[i]]
                print("  - {0: <{1}}\t{2}".format(
                    "'cov_mat'?", width-4, "No (%s)" % (ccheck_i) if
                    len(ccheck_i) else "Yes"))
            print("-"*width)

            # PARAMETER SPACE
            print("\nPARAMETER SPACE")
            print("-"*width)

            # Define string format if par_est was provided
            str_format1 = "{8}{0: <{1}}: [{2: >{3}}, {4: >{5}}] ({6: >{7}.5f})"

            # Define string format if par_est was not provided
            str_format2 = "{8}{0: <{1}}: [{2: >{3}}, {4: >{5}}] ({6:->{7}})"

            # Print details about every model parameter in parameter space
            for i in range(n_par):
                # Determine what string to use for the active flag
                if(delist(ccheck_flat) != []):
                    active_str = "?"
                elif i in self._emulator._active_par[emul_i]:
                    active_str = "*"
                else:
                    active_str = " "

                # Check if par_est is given and use correct string formatting
                if self._modellink._par_est[i] is not None:
                    print(str_format1.format(
                        self._modellink._par_name[i], name_len,
                        self._modellink._par_rng[i, 0], lower_len,
                        self._modellink._par_rng[i, 1], upper_len,
                        self._modellink._par_est[i], est_len, active_str))
                else:
                    print(str_format2.format(
                        self._modellink._par_name[i], name_len,
                        self._modellink._par_rng[i, 0], lower_len,
                        self._modellink._par_rng[i, 1], upper_len,
                        "", est_len, active_str))

            # FOOTER
            print("="*width)

        # MPI Barrier
        self._comm.Barrier()

    # This function allows the user to evaluate a given sam_set in the emulator
    # TODO: Plot emul_i_stop for large LHDs, giving a nice mental statistic
    @docstring_substitute(emul_i=user_emul_i_doc)
    def evaluate(self, sam_set, emul_i=None):
        """
        Evaluates the given model parameter sample set `sam_set` at given
        emulator iteration `emul_i`.
        The output of this function depends on the number of dimensions in
        `sam_set`. The output is always provided on the controller rank.

        Parameters
        ----------
        sam_set : 1D or 2D array_like
            Array containing model parameter value sets to be evaluated in the
            emulator system.

        Optional
        --------
        %(emul_i)s

        Returns (if ndim(sam_set) > 1)
        ------------------------------
        impl_check : list of bool
            List of bool indicating whether or not the given samples passed the
            implausibility check at the given emulator iteration `emul_i`.
        emul_i_stop : list of int
            List containing the last emulator iteration identifiers at which
            the given samples are still within the emulator system.
        adj_exp_val : list of 1D :obj:`~numpy.ndarray` objects
            List of arrays containing the adjusted expectation values for all
            given samples.
        adj_var_val : list of 1D :obj:`~numpy.ndarray` objects
            List of arrays containing the adjusted variance values for all
            given samples.
        uni_impl_val : list of 1D :obj:`~numpy.ndarray` objects
            List of arrays containing the univariate implausibility values for
            all given samples.

        Prints (if ndim(sam_set) == 1)
        ------------------------------
        impl_check : bool
            Bool indicating whether or not the given sample passed the
            implausibility check at the given emulator iteration `emul_i`.
        emul_i_stop : int
            Last emulator iteration identifier at which the given sample is
            still within the emulator system.
        adj_exp_val : 1D :obj:`~numpy.ndarray` object
            The adjusted expectation values for the given sample.
        adj_var_val : 1D :obj:`~numpy.ndarray` object
            The adjusted variance values for the given sample.
        sigma_val : 1D :obj:`~numpy.ndarray` object
            The corresponding sigma value for the given sample.
        uni_impl_val : 1D :obj:`~numpy.ndarray` object
            The univariate implausibility values for the given sample.

        Notes
        -----
        If given emulator iteration `emul_i` has been analyzed before, the
        implausibility parameters of the last analysis are used. If not, then
        the values are used that were read in when the emulator system was
        loaded.

        """

        # Get emulator iteration
        emul_i = self._emulator._get_emul_i(emul_i, True)

        # Do some logging
        logger = getCLogger('EVALUATE')
        logger.info("Evaluating emulator system for provided set of model "
                    "parameter samples.")

        # Make sure that sam_set is a NumPy array
        sam_set = np.array(sam_set)

        # Controller checking the contents of sam_set
        if self._is_controller:
            # Check the number of dimensions in sam_set
            if(sam_set.ndim == 1):
                sam_set = np.array(sam_set, ndmin=2)
                print_output = 1
            elif(sam_set.ndim == 2):
                print_output = 0
            else:
                logger.error("Input argument 'sam_set' is not one-dimensional "
                             "or two-dimensional!")
                raise ShapeError("Input argument 'sam_set' is not "
                                 "one-dimensional or two-dimensional!")

            # Check if sam_set has n_par parameter values
            if not(sam_set.shape[1] == self._modellink._n_par):
                logger.error("Input argument 'sam_set' has incorrect number of"
                             " parameters (%s != %s)!"
                             % (sam_set.shape[1], self._modellink._n_par))
                raise ShapeError("Input argument 'sam_set' has incorrect "
                                 "number of parameters (%s != %s)!"
                                 % (sam_set.shape[1], self._modellink._n_par))

            # Check if sam_set consists only out of floats (or ints)
            else:
                for i, par_set in enumerate(sam_set):
                    for j, par_val in enumerate(par_set):
                        check_float(par_val, 'sam_set[%s, %s]' % (i, j))

        # The workers make sure that sam_set is also two-dimensional
        else:
            sam_set = np.array(sam_set, ndmin=2)

        # MPI Barrier
        self._comm.Barrier()

        # Define the various code snippets
        pre_code = compile("adj_exp_val = [[] for _ in range(n_sam)]\n"
                           "adj_var_val = [[] for _ in range(n_sam)]\n"
                           "uni_impl_val_list = [[] for _ in range(n_sam)]\n"
                           "emul_i_stop = [[] for _ in range(n_sam)]",
                           '<string>', 'exec')
        eval_code = compile("adj_exp_val[sam_idx[j]] = adj_val[0]\n"
                            "adj_var_val[sam_idx[j]] = adj_val[1]\n"
                            "uni_impl_val_list[sam_idx[j]] = uni_impl_val",
                            '<string>', 'exec')
        anal_code = compile("emul_i_stop[sam_idx[j]] = i", '<string>', 'exec')
        post_code = compile("adj_exp_val_list = self._comm.gather(np.array("
                            "adj_exp_val), 0)\n"
                            "adj_var_val_list = self._comm.gather(np.array("
                            "adj_var_val), 0)\n"
                            "uni_impl_val_list = self._comm.gather(np.array("
                            "uni_impl_val_list), 0)",
                            '<string>', 'exec')
        exit_code = compile("adj_exp_val = np.concatenate("
                            "*[adj_exp_val_list], axis=1)\n"
                            "adj_var_val = np.concatenate("
                            "*[adj_var_val_list], axis=1)\n"
                            "uni_impl_val = np.concatenate("
                            "*[uni_impl_val_list], axis=1)\n"
                            "self.results = (adj_exp_val, adj_var_val, "
                            "uni_impl_val, emul_i_stop, impl_check)",
                            '<string>', 'exec')

        # Combine code snippets into a tuple
        exec_code = (pre_code, eval_code, anal_code, post_code, exit_code)

        # Analyze sam_set
        results = self._analyze_sam_set(self, emul_i, sam_set, *exec_code)

        # Do more logging
        logger.info("Finished evaluating emulator system.")

        # Controller finishing up
        if self._is_controller:
            # Extract data arrays from results
            adj_exp_val, adj_var_val, uni_impl_val, emul_i_stop, impl_check =\
                results

            # If ndim(sam_set) == 1, print the results
            if print_output:
                # Print results
                if impl_check[0]:
                    print("Plausible? Yes")
                    print("-"*14)
                else:
                    print("Plausible? No")
                    print("-"*13)
                print("emul_i_stop = %s" % (emul_i_stop[0]))
                print("adj_exp_val = %s" % (adj_exp_val[0]))
                print("adj_var_val = %s" % (adj_var_val[0]))
                print("sigma_val = %s" % (np.sqrt(adj_var_val[0])))
                print("uni_impl_val = %s" % (uni_impl_val[0]))

            # Else, return the lists
            else:
                # MPI Barrier for controller
                self._comm.Barrier()

                # Return results
                return(impl_check.tolist(), emul_i_stop, adj_exp_val,
                       adj_var_val, uni_impl_val)

        # MPI Barrier
        self._comm.Barrier()

    # This function creates the projection figures of a given emul_i
    @docstring_copy(Projection.__call__)
    def project(self, emul_i=None, proj_par=None, figure=True, show=False,
                force=False):

        # Initialize the Projection class and make the figures
        Projection(self, emul_i, proj_par, figure, show, force)

        # MPI Barrier
        self._comm.Barrier()
