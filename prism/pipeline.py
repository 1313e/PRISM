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
from __future__ import absolute_import, division, print_function

# Built-in imports
import os
from os import path
from time import strftime, strptime

# Package imports
from e13tools import InputError
from e13tools.sampling import lhd
import h5py
import logging
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import numpy as np
from numpy.random import normal, random
from sklearn.linear_model import LinearRegression as LR

# PRISM imports
from ._internal import RequestError, check_nneg_float, check_pos_int,\
                       check_str, docstring_copy, move_logger, start_logger
from .emulator import Emulator
from .projection import Projection

# All declaration
__all__ = ['Pipeline']


# %% PIPELINE CLASS DEFINITION
# OPTIMIZE: Introduce multiple versions of functions/methods that are loaded in
# This way, the Pipeline only has to do versatility checks once and not every
# single time, thus removing many if-statements by just loading the correct
# function in during class initialization.
# TODO: Rewrite PRISM into MPI
# TODO: Allow user to switch between emulation and modelling
# TODO: Implement multivariate implausibilities
# TODO: Implement checks for user-provided ints, floats, positive, negative...
class Pipeline(object):
    """
    Defines the :class:`~Pipeline` class of the PRISM package.

    """

    def __init__(self, modellink, root_dir=None, working_dir=None,
                 prefix='prism_', hdf5_file='prism.hdf5',
                 prism_file='prism.txt'):
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
            that was created in `root_dir`. If no directories are found, one
            will be created.
        prefix : str. Default: 'prism_'
            String containing a prefix that is used for naming new working
            directories or scan for existing ones.
        hdf5_file : str. Default: 'prism.hdf5'
            String containing the name of the HDF5-file in `working_dir` to be
            used in this class instance. Different types of HDF5-files can be
            provided:
                *Non-existing HDF5-file*: This file will be created and used to
                save the constructed emulator in.

                *Existing HDF5-file*: This file will be used to regenerate a
                previously constructed emulator.
        prism_file : str or None. Default: 'prism.txt'
            String containing the absolute or relative path to the TXT-file
            containing the PRISM parameters that need to be changed from their
            default values. If a relative path is given, this file must be
            located inside `root_dir`. If *None*, no changes will be made to
            the default parameters.

        """

        # Start logging
        start_logger()
        logger = logging.getLogger('PIPELINE')
        logger.info("")

        # Initialize class
        logger = logging.getLogger('INIT')
        logger.info("Initializing Pipeline class.")

        # Obtain paths
        self._get_paths(root_dir, working_dir, prefix, hdf5_file, prism_file)

        # Move logger to working directory
        move_logger(self._working_dir)

        # Initialize Emulator class
        self._emulator = Emulator(self)

        # Link provided ModelLink subclass to Emulator class
        self._emulator._set_modellink(modellink)

        # Link provided ModelLink subclass to Pipeline class
        self._modellink = self._emulator._modellink
        self._modellink_name = self._emulator._modellink_name

        # Read/load in pipeline parameters
        self._read_parameters()
        self._load_data()

        # Print out the details of the current state of the pipeline
        self.details()

    # Allows one to call one full loop of the PRISM pipeline
    def __call__(self, emul_i=None):
        """
        Calls the :meth:`~construct` method to start the construction of the
        next iteration of the emulator system and creates the projection
        figures right afterward if this construction was successful.

        Optional
        --------
        emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the last iteration of the loaded emulator system will
            be used.

        """

        # Check the provided emul_i
        if emul_i is None:
            emul_i = self._emulator._emul_i+1
        else:
            emul_i = check_pos_int(emul_i, 'emul_i')

        # Perform construction
        try:
            self.construct(emul_i)
        except Exception:
            raise
        else:
            # Perform projection
            if self._prc[emul_i]:
                self.create_projection(emul_i)

            # Print details
            self.details(emul_i)


# %% CLASS PROPERTIES
    # TODO: Hide class attributes that do not exist yet
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
    def prefix(self):
        """
        String used as a prefix when naming new working directories.

        """

        return(self._prefix)

    @property
    def hdf5_file(self):
        """
        Absolute path to the loaded HDF5-file.

        """

        return(self._hdf5_file)

    @property
    def hdf5_file_name(self):
        """
        Name of loaded HDF5-file.

        """

        return(self._hdf5_file_name)

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
    def modellink_name(self):
        """
        Name of the :obj:`~ModelLink` instance provided during Pipeline
        initialization.

        """

        return(self._modellink_name)

    @property
    def emulator(self):
        """
        The :obj:`~Emulator` instance created during Pipeline initialization.

        """

        return(self._emulator)

    @property
    def criterion(self):
        """
        String indicating which criterion to use in the
        :func:`e13tools.sampling.lhd` function.

        """

        # TODO: Maybe an integer should also be allowed as criterion?
        return(self._criterion)

    @property
    def do_active_par(self):
        """
        Bool indicating whether or not to use active parameters.

        """

        return(bool(self._do_active_par))

    @property
    def n_sam_init(self):
        """
        Number of evaluation samples used to construct the first iteration of
        the emulator system.

        """

        return(self._n_sam_init)

    @property
    def n_eval_samples(self):
        """
        Array with the base number of emulator evaluations used to scan the
        plausible parameter region and update the emulator system. This number
        is scaled up by the number of model parameters and the current emulator
        iteration to generate the true number of emulator evaluations.

        """

        return(self._n_eval_samples)

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
        List of bools indicating whether or not plausible regions have been
        found in the corresponding emulator iteration.

        """

        return([bool(self._prc[i]) for i in range(self._emul_i+1)])

    @property
    def impl_sam(self):
        """
        List of arrays containing all model evaluation samples that will be
        added to the next emulator iteration.

        """

        return(self._impl_sam)

    @property
    def use_mock(self):
        """
        Bool indicating whether or not mock data has been used for the creation
        of this emulator system instead of actual data.

        """

        return(bool(self._use_mock))


# %% GENERAL CLASS METHODS
    # Function containing the model output for a given set of parameter values
    # Might want to save all model output immediately to prevent data loss
    def _call_model(self, emul_i, par_set):
        """
        Obtain the data set that is generated by the model for a given model
        parameter value set `par_set`. The current emulator iteration `emul_i`
        is also provided in case it is required by the :class:`~ModelLink`
        subclass.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.
        par_set : 1D array_like
            Model parameter value set to calculate the model output for.

        Returns
        -------
        mod_set : 1D :obj:`~numpy.ndarray` object
            Model output corresponding to given `par_set`.

        """

        # Log that model is being called
        logger = logging.getLogger('MODEL')
        logger.info("Calling model at parameters %s." % (par_set))

        # Make sure par_setp is at least 1D and a numpy array
        sam = np.array(par_set, ndmin=1)

        # Create par_dict
        par_dict = dict(zip(self._modellink._par_names, sam))

        # Obtain model data output
        mod_set = self._modellink.call_model(emul_i, par_dict,
                                             self._modellink._data_idx)

        # Log that calling model has been finished
        logger.info("Model returned %s." % (mod_set))

        # Return it
        return(np.array(mod_set))

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

    # Read in the pipeline attributes
    def _retrieve_parameters(self):
        """
        Reads in the pipeline parameters from the provided HDF5-file and saves
        them in the current :obj:`~Pipeline` instance.

        """

        # Log that parameters are being read
        logger = logging.getLogger('INIT')
        logger.info("Retrieving pipeline parameters from provided HDF5-file.")

        # Open hdf5-file
        file = self._open_hdf5('r')

        # Read in all the pipeline attributes
        self._n_sam_init = file.attrs['n_sam_init']
        self._do_active_par = file.attrs['do_active_par']
        self._criterion = file.attrs['criterion'].decode('utf-8')

        # Close hdf5-file
        self._close_hdf5(file)

        # Log that reading is finished
        logger.info("Finished retrieving parameters.")

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
        logger = logging.getLogger('INIT')
        logger.info("Generating default pipeline parameter dict.")

        # Create parameter dict with default parameters
        par_dict = {'n_sam_init': '500',
                    'n_eval_samples': '800',
                    'criterion': "'multi'",
                    'do_active_par': 'True',
                    'impl_cut': '0, 4.0, 3.8, 3.5',
                    'use_mock': 'False'}

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
        logger = logging.getLogger('INIT')
        logger.info("Reading pipeline parameters.")

        # Obtaining default pipeline parameter dict
        par_dict = self._get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._prism_file is not None:
            pipe_par = np.genfromtxt(self._prism_file, dtype=(str),
                                     delimiter=': ', autostrip=True)

            # Make sure that pipe_par is 2D
            pipe_par = np.array(pipe_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(pipe_par)

        # GENERAL
        # Number of starting samples
        self._n_sam_init = check_pos_int(int(par_dict['n_sam_init']),
                                         'n_sam_init')

        # Criterion parameter used for Latin Hypercube Sampling
        self._criterion = check_str(
            str(par_dict['criterion']).replace("'", ''), 'criterion')

        # Obtain the bool determining whether or not to have active parameters
        if(par_dict['do_active_par'].lower() in ('false', '0')):
            self._do_active_par = 0
        elif(par_dict['do_active_par'].lower() in ('true', '1')):
            self._do_active_par = 1
        else:
            logger.error("Pipeline parameter 'do_active_par' is not of type "
                         "'bool'!")
            raise TypeError("Pipeline parameter 'do_active_par' is not of type"
                            " 'bool'!")

        # Obtain the bool determining whether or not to use mock data
        if(par_dict['use_mock'].lower() in ('false', '0')):
            self._use_mock = 0
        elif(par_dict['use_mock'].lower() in ('true', '1')):
            self._use_mock = 1
            self._get_mock_data(self._emulator._emul_i)
        else:
            logger.error("Pipeline parameter 'use_mock' is not of type "
                         "'bool'!")
            raise TypeError("Pipeline parameter 'use_mock' is not of type "
                            "'bool'!")

        # Log that reading has been finished
        logger.info("Finished reading pipeline parameters.")

    # This function either generates mock_data or loads it into ModelLink
    def _get_mock_data(self, emul_i):
        """
        Generates mock_data if `emul_i` = 0 or loads it into the
        :obj:`~ModelLink` instance that was provided during class
        initialization if otherwise.
        This function overwrites the :class:`~ModelLink` properties holding the
        parameter estimates, data values and data errors.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.

        Generates
        ---------
        Mock values for the parameter estimates, data values and data errors if
        `emul_i` = 0.
        Overwrites the corresponding :class:`~ModelLink` class properties with
        either the generated values or previously used values.

        """

        # Start logger
        logger = logging.getLogger('INIT')

        # If emul_i is 0, generate new mock values
        if not emul_i:
            # Log new mock_data being created
            logger.info("Generating mock_data for new emulator system.")

            # Set non-default parameter estimate
            self._modellink._par_estimate =\
                self._modellink._par_rng[:, 0] +\
                random(self._modellink._par_dim) *\
                (self._modellink._par_rng[:, 1] -
                 self._modellink._par_rng[:, 0])

            # Set non-default model data values
            self._modellink._data_val =\
                self._call_model(0, self._modellink._par_estimate).tolist()

            # Use model discrepancy variance as model data errors
            try:
                md_var = self._modellink.get_md_var(
                    emul_i, self._emulator._data_idx[emul_i])
            except NotImplementedError:
                md_var = pow(np.array(self._modellink._data_val)/6, 2)
            finally:
                # Check if all values are non-negative floats
                for value in md_var:
                    check_nneg_float(value, 'md_var')
                self._modellink._data_err = np.sqrt(md_var).tolist()

            # Add model data errors as noise to model data values
            self._modellink._data_val =\
                (self._modellink._data_val +
                 normal(scale=self._modellink._data_err)).tolist()

        # If emul_i has any other value, overwrite ModelLink properties with
        # previously generated values
        else:
            # Log that mock_data is being loaded in
            logger.info("Loading previously used mock_data into ModelLink.")

            # Open hdf5-file
            file = self._open_hdf5('r')

            # Overwrite ModelLink properties
            self._modellink._par_estimate = file.attrs['mock_par']
            self._modellink._data_val = self._emulator._data_val[emul_i]
            self._modellink._data_err = self._emulator._data_err[emul_i]

            # Close hdf5-file
            self._close_hdf5(file)

        # Log end
        logger.info("Loaded mock_data.")

    # This function controls how n_eval_samples is calculated
    def _get_n_eval_samples(self, emul_i):
        """
        This function calculates the total amount of emulator evaluation
        samples at a given emulator iteration `emul_i` from the
        `n_eval_samples` provided during class initialization.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.

        Returns
        -------
        n_eval_samples : int
            Number of emulator evaluation samples.

        """

        # Calculate n_eval_samples
        return(emul_i*self._n_eval_samples[emul_i]*self._modellink._par_dim)

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
            in `root_dir`. If *None*, working directory is set to the last one
            that was created in `root_dir`. If no directories are found, one
            will be created.
        prefix : str
            String containing a prefix that is used for naming new working
            directories or scan for existing ones.
        hdf5_file : str
            String containing the name of the HDF5-file in `working_dir` to be
            used in this class instance.
        prism_file : str or None
            String containing the absolute or relative path to the TXT-file
            containing the PRISM parameters that need to be changed from their
            default values. If a relative path is given, this file must be
            located inside `root_dir`. If *None*, no changes will be made to
            the default parameters.

        Generates
        ---------
        The absolute paths to the root directory, working directory, pipeline
        HDF5-file and PRISM parameters file.

        """

        # Set logging system
        logger = logging.getLogger('INIT')
        logger.info("Obtaining related directory and file paths.")

        # Obtain root directory path
        # If one did not specify a root directory, set it to default
        if root_dir is None:
            logger.info("No root directory specified, setting it to default.")
            self._root_dir = path.abspath('.')
            logger.info("Root directory set to '%s'." % (self._root_dir))

        # If one specified a root directory, use it
        elif isinstance(root_dir, str):
            logger.info("Root directory specified.")
            self._root_dir = path.abspath(root_dir)
            logger.info("Root directory set to '%s'." % (self._root_dir))

            # Check if this directory already exists
            try:
                logger.info("Checking if root directory already exists.")
                os.mkdir(self._root_dir)
            except OSError:
                logger.info("Root directory already exists.")
                pass
            else:
                logger.info("Root directory did not exist, created it.")
                pass
        else:
            raise InputError("Input argument 'root_dir' is invalid!")

        # Check if a valid working directory prefix string is given
        if isinstance(prefix, str):
            self._prefix = prefix
            prefix_len = len(prefix)
        else:
            raise TypeError("Input argument 'prefix' is not of type 'str'!")

        # Obtain working directory path
        # If one did not specify a working directory, obtain it
        if working_dir is None:
            logger.info("No working directory specified, trying to load last "
                        "one created.")
            dirnames = next(os.walk(self._root_dir))[1]
            emul_dirs = list(dirnames)

            # Check which directories in the root_dir satisfy the default
            # naming scheme of the emulator directories
            for dirname in dirnames:
                if(dirname[0:prefix_len] != prefix):
                    emul_dirs.remove(dirname)
                else:
                    try:
                        strptime(dirname[prefix_len:prefix_len+10], '%Y-%m-%d')
                    except ValueError:
                        emul_dirs.remove(dirname)

            # If no working directory exists, make a new one
            if(len(emul_dirs) == 0):
                logger.info("No working directories found, creating it.")
                working_dir = ''.join([prefix, strftime('%Y-%m-%d')])
                self._working_dir = path.join(self._root_dir, working_dir)
                os.mkdir(self._working_dir)
                logger.info("Working directory set to '%s'." % (working_dir))
            else:
                logger.info("Working directories found, loading last one.")
                emul_dirs.sort(reverse=True)
                working_dir = emul_dirs[0]
                self._working_dir = path.join(self._root_dir, working_dir)
                logger.info("Working directory set to '%s'." % (working_dir))

        # If one requested a new working directory
        elif isinstance(working_dir, int):
            logger.info("New working directory requested, creating it.")
            working_dir = ''.join([self._prefix, strftime('%Y-%m-%d')])
            dirnames = next(os.walk(self._root_dir))[1]
            emul_dirs = list(dirnames)

            for dirname in dirnames:
                if(dirname[0:prefix_len+10] != working_dir):
                    emul_dirs.remove(dirname)

            # Check if other working directories already exist with the same
            # prefix and append a number to the name if this is the case
            emul_dirs.sort(reverse=True)
            if(len(emul_dirs) == 0):
                pass
            elif(len(emul_dirs[0]) == prefix_len+10):
                working_dir = ''.join([working_dir, '_1'])
            else:
                working_dir =\
                    ''.join([working_dir, '_%s'
                             % (int(emul_dirs[0][prefix_len+11:])+1)])

            self._working_dir = path.join(self._root_dir, working_dir)
            os.mkdir(self._working_dir)
            logger.info("Working directory set to '%s'." % (working_dir))

        # If one specified a working directory, use it
        elif isinstance(working_dir, str):
            logger.info("Working directory specified.")
            self._working_dir =\
                path.join(self._root_dir, working_dir)
            logger.info("Working directory set to '%s'." % (working_dir))

            # Check if this directory already exists
            try:
                logger.info("Checking if working directory already exists.")
                os.mkdir(self._working_dir)
#            except FileExistsError:
            except OSError:
                logger.info("Working directory already exists.")
                pass
            else:
                logger.info("Working directory did not exist, created it.")
                pass
        else:
            raise InputError("Input argument 'working_dir' is invalid!")

        # Obtain hdf5-file path
        if isinstance(hdf5_file, str):
            self._hdf5_file = path.join(self._working_dir, hdf5_file)
            logger.info("HDF5-file set to '%s'." % (hdf5_file))
            self._hdf5_file_name = path.join(working_dir, hdf5_file)
        else:
            raise TypeError("Input argument 'hdf5_file' is not of type 'str'!")

        # Obtain PRISM parameter file path
        # If no PRISM parameter file was provided
        if prism_file is None:
            self._prism_file = None

        # If a PRISM parameter file was provided
        elif isinstance(prism_file, str):
            if path.exists(prism_file):
                self._prism_file = prism_file
            else:
                self._prism_file = path.join(self._root_dir, prism_file)
            logger.info("PRISM parameters file set to '%s'." % (prism_file))
        else:
            raise InputError("Input argument 'prism_file' is invalid!")

    # This function loads pipeline data
    def _load_data(self):
        """
        Loads in all the important pipeline data into memory.

        Generates
        ---------
        All relevant pipeline data is loaded into memory.

        """

        # Set the logger
        logger = logging.getLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing pipeline data sets.")
        self._prc = [[]]
        self._impl_sam = [[]]
        self._impl_cut = [[]]
        self._cut_idx = [[]]
        self._n_eval_samples = [[]]

        # If an emulator system currently exists, load in all data
        if self._emulator._emul_i:
            # Open hdf5-file
            file = self._open_hdf5('r')

            # Read in the data up to the last emulator iteration
            for i in range(1, self._emulator._emul_i+1):
                # Check if analysis has been carried out (only if i=emul_i)
                try:
                    self._prc.append(file['%s' % (i)].attrs['prc'])

                # If not, no plausible regions were found
                except KeyError:
                    self._prc.append(0)
                    self._impl_sam.append([])
                    self._n_eval_samples.append(1)

                # If so, load in all data
                else:
                    self._impl_sam.append(file['%s/impl_sam' % (i)][()])
                    self._impl_cut.append(file['%s' % (i)].attrs['impl_cut'])
                    self._cut_idx.append(file['%s' % (i)].attrs['cut_idx'])
                    self._n_eval_samples.append(
                        file['%s' % (i)].attrs['n_eval_samples'])

            # Close hdf5-file
            self._close_hdf5(file)

    # This function saves pipeline data to hdf5
    def _save_data(self, keyword, data):
        """
        Saves the provided `data` for the specified data-type `keyword` at the
        last emulator iteration to the HDF5-file and as an data
        attribute to the current :obj:`~Pipeline` instance.

        Parameters
        ----------
        keyword : {'impl_cut', 'impl_sam', 'n_eval_samples'}
            String specifying the type of data that needs to be saved.
        data : list
            The actual data that needs to be saved at data keyword `keyword`.

        Generates
        ---------
        The specified data is saved to the HDF5-file.

        """

        # Do some logging
        logger = logging.getLogger('SAVE_DATA')
        logger.info("Saving %s data at iteration %s to HDF5."
                    % (keyword, self._emulator._emul_i))

        # Open hdf5-file
        file = self._open_hdf5('r+')

        # Check what data keyword has been provided
        # IMPL_CUT
        if(keyword == 'impl_cut'):
            # Check if impl_cut data has been saved before (analysis was done)
            try:
                self._impl_cut[self._emulator._emul_i] = data[0]
                self._cut_idx[self._emulator._emul_i] = data[1]
            except IndexError:
                self._impl_cut.append(data[0])
                self._cut_idx.append(data[1])
            finally:
                file['%s' % (self._emulator._emul_i)].attrs['impl_cut'] =\
                    data[0]
                file['%s' % (self._emulator._emul_i)].attrs['cut_idx'] =\
                    data[1]

        # IMPL_SAM
        elif(keyword == 'impl_sam'):
            # Check if any plausible regions have been found at all
            if(np.shape(data)[0] == 0):
                prc = 0
            else:
                prc = 1

            # Check if impl_sam data has been saved before (analysis was done)
            try:
                self._prc[self._emulator._emul_i] = prc
            except IndexError:
                self._prc.append(prc)
                file.create_dataset('%s/impl_sam'
                                    % (self._emulator._emul_i), data=data)
                self._impl_sam.append(data)
            else:
                del file['%s/impl_sam' % (self._emulator._emul_i)]
                file.create_dataset('%s/impl_sam' % (self._emulator._emul_i),
                                    data=data)
                self._impl_sam[self._emulator._emul_i] = data
            finally:
                file['%s' % (self._emulator._emul_i)].attrs['prc'] = prc

        # N_EVAL_SAMPLES
        elif(keyword == 'n_eval_samples'):
            # Check if n_eval_samples has been saved before (analysis was done)
            try:
                self._n_eval_samples[self._emulator._emul_i] = data
            except IndexError:
                self._n_eval_samples.append(data)
            finally:
                file['%s'
                     % (self._emulator._emul_i)].attrs['n_eval_samples'] = data

        # INVALID KEYWORD
        else:
            logger.error("Invalid keyword argument provided!")
            raise ValueError("Invalid keyword argument provided!")

        # Close hdf5
        self._close_hdf5(file)

    # This is function 'k'
    # Reminder that this function should only be called once per sample set
    def _evaluate_model(self, emul_i, sam_set):
        """
        Evaluates the model at all specified model evaluation samples at a
        given emulator iteration `emul_i`.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.
        sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the model evaluation samples.

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
        logger = logging.getLogger('MODEL')
        logger.info("Evaluating model samples.")

        # Obtain sample and parameter dimension
        mod_dim = np.shape(sam_set)[0]

        # Generate mod_set
        mod_set = np.zeros([self._emulator._n_data[emul_i], mod_dim])

        # Do model evaluations
        for i in range(mod_dim):
            mod_set[:, i] = self._call_model(emul_i, sam_set[i])

        # Save data to hdf5
        self._emulator._save_data(emul_i, 'sam_set', sam_set)
        self._emulator._save_data(emul_i, 'mod_set', mod_set)

        # Log that this is finished
        logger.info("Finished evaluating model samples.")

    # This function extracts the set of active parameters
    # TODO: Allow user to manually specify the active parameters
    # TODO: Perform exhaustive backward stepwise regression on order > 1
    def _get_active_par(self, emul_i):
        """
        Determines the active parameters to be used for every individual data
        point in the provided emulator iteration `emul_i`. Uses backwards
        stepwise elimination to determine the set of active parameters.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.

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

        # Check if active parameters have been requested
        if(self._do_active_par == 0):
            # If not, then save all parameters as active parameters
            active_par = list(range(self._modellink._par_dim))
            active_par_data = []
            for i in range(self._emulator._n_data[emul_i]):
                active_par_data.append(active_par)

        else:
            # Perform an exhaustive backward stepwise regression
            active_par = set()
            active_par_data = []
            for i in range(self._emulator._n_data[emul_i]):
                # Create ExhaustiveFeatureSelector object
                efs_obj = EFS(LR(), 1, int(self._modellink._par_dim), False,
                              'r2')

                # Fit the data set
                efs_obj.fit(self._emulator._sam_set[emul_i],
                            self._emulator._mod_set[emul_i][i])

                # Extract the active parameters for this data set
                active_par_data.append(np.sort(efs_obj.best_idx_))

                # And extract the unique active parameters for this iteration
                active_par.update(active_par_data[i])

        # Save the active parameters
        self._emulator._save_data(emul_i, 'active_par',
                                  [np.array(list(active_par)),
                                   active_par_data])

        # Log that active parameter determination is finished
        logger.info("Finished determining active parameters.")

    # This function generates a large Latin Hypercube sample set to evaluate
    # the emulator at
    # TODO: Maybe make sure that n_sam_init samples are used for next iteration
    # This can be done by evaluating a 1000 samples in the emulator, check how
    # many survive and then use an LHD with the number of samples required to
    # let n_sam_init samples survive.
    def _get_eval_sam_set(self, emul_i):
        """
        Generates an emulator evaluation sample set to be used for updating an
        emulator iteration. Currently uses the
        :func:`~e13tools.sampling.lhd` function.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.

        Returns
        -------
        eval_sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the evaluation samples.

        """

        # Log about this
        logger = logging.getLogger('EVAL_SAMS')

        # Obtain number of samples
        n_samples = self._get_n_eval_samples(emul_i)

        # Create array containing all new samples to evaluate with emulator
        logger.info("Creating emulator evaluation sample set with size %s."
                    % (n_samples))
        eval_sam_set = lhd(n_samples, self._modellink._par_dim,
                           self._modellink._par_rng, 'fixed',
                           self._criterion, 100,
                           constraints=self._emulator._sam_set[emul_i])
        logger.info("Finished creating sample set.")

        # Return it
        return(eval_sam_set)

    # This function performs an implausibility cut-off check on a given sample
    # TODO: Implement dynamic impl_cut
    @staticmethod
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
        emul_i : int
            Number indicating the requested emulator iteration.
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

        # Log that impl_check is being carried out
        logger = logging.getLogger('IMPL_CHECK')
        logger.info("Performing implausibility cut-off check on %s."
                    % (uni_impl_val))

        # Sort impl_val to compare with the impl_cut list
        # TODO: Maybe use np.partition here?
        sorted_impl_val = np.flip(np.sort(uni_impl_val, axis=-1), axis=-1)

        # Save the implausibility value at the first real cut-off
        impl_cut_val = sorted_impl_val[obj._cut_idx[emul_i]]

        # Scan over all data points in this sample
        for i, val in enumerate(obj._impl_cut[emul_i]):
            # If impl_cut is not 0 and impl_val is not below impl_cut, break
            if(val != 0 and
               sorted_impl_val[i] > val):
                logger.info("Check result is negative.")
                return(False, impl_cut_val)
        else:
            # If for-loop ended in a normal way, the check was successful
            logger.info("Check result is positive.")
            return(True, impl_cut_val)

    # This is function 'I²(x)'
    # This function calculates the univariate implausibility values
    # TODO: Introduce check if emulator variance is much lower than other two
    # TODO: Parameter uncertainty should be implemented at some point
    def _get_uni_impl(self, emul_i, adj_exp_val, adj_var_val):
        """
        Calculates the univariate implausibility values at a given emulator
        iteration `emul_i` for specified expectation and variance values
        `adj_exp_val` and `adj_var_val`.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.
        adj_exp_val, adj_var_val : 1D array_like
            The adjusted expectation and variance values to calculate the
            univeriate implausibility for.

        Returns
        -------
        uni_impl_val : 1D :obj:`~numpy.ndarray` object
            Univariate implausibility value for every data point.

        """

        # Log that univariate implausibility value is calculated
        logger = logging.getLogger('UNI_IMPL')
        logger.info("Calculating univariate implausibility value.")

        # Obtain model discrepancy variance
        md_var = self._get_md_var(emul_i)

        # Initialize empty univariate implausibility
        uni_impl_val_sq = np.zeros(self._emulator._n_data[emul_i])

        # Calcualte the univariate implausibility values
        for i in range(self._emulator._n_data[emul_i]):
            uni_impl_val_sq[i] =\
                pow(adj_exp_val[i]-self._emulator._data_val[emul_i][i], 2) /\
                (adj_var_val[i]+md_var[i] +
                 pow(self._emulator._data_err[emul_i][i], 2))

        # Take square root
        uni_impl_val = np.sqrt(uni_impl_val_sq)

        # Log the result
        logger.info("Univariate implausibility value is %s." % (uni_impl_val))

        # Return it
        return(uni_impl_val)

    # This function calculates the model discrepancy variance
    # Basically takes all uncertainties of Sec. 3.1 of Vernon into account that
    # are not already in the emulator ([3] and [5])
    def _get_md_var(self, emul_i):
        """
        Retrieves the model discrepancy variance, which includes all variances
        that are created by the model provided by the :obj:`~ModelLink`
        instance. This method tries to call the :meth:`~ModelLink.get_md_var`
        method, and assumes a default model discrepancy variance of 1/6th the
        data value if it cannot be called.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.

        Returns
        -------
        var_md : 1D :obj:`~numpy.ndarray` object
            Variance of the model discrepancy.

        """

        # Obtain md variances
        # Try to use the user-defined md variances
        try:
            md_var =\
                self._modellink.get_md_var(emul_i,
                                           self._emulator._data_idx[emul_i])

        # If it was not user-defined, use a default value
        except NotImplementedError:
            # Use factor 2 difference on 2 sigma as acceptable
            # Imagine that 2 sigma range is given if lower and upper are factor
            # 2 apart. This gives that sigma must be 1/6th of the data value
            md_var = pow(np.array(self._emulator._data_val[emul_i])/6, 2)

        # Check if all values are non-negative floats
        for value in md_var:
            check_nneg_float(value, 'md_var')

        # Return it
        return(md_var)

    # This function completes the list of implausibility cut-offs
    @staticmethod
    def _get_impl_cut(obj, impl_cut):
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

        Generates
        ---------
        impl_cut : 1D :obj:`~numpy.ndarray` object
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
        for i in range(1, len(impl_cut)):
            impl_cut[i] = check_nneg_float(impl_cut[i], 'impl_cut')
            if(impl_cut[i] == 0):
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
        obj._save_data('impl_cut', [np.array(impl_cut), cut_idx])

        # Log end of process
        logger.info("Finished generating implausibility cut-off list.")

    # This function reads in the impl_cut list from the PRISM parameters file
    # TODO: Make impl_cut dynamic
    def _get_impl_par(self, emul_i):
        """
        Reads in the impl_cut list and other parameters for implausibility
        evaluations from the PRISM parameters file and saves them in the given
        emulator iteration `emul_i`.

        Parameters
        ----------
        emul_i : int
            Number indicating the requested emulator iteration.

        Generates
        ---------
        impl_cut : 1D :obj:`~numpy.ndarray` object
            Full list containing the impl_cut-offs for all data points provided
            to the emulator.
        cut_idx : int
            Index of the first impl_cut-off in the impl_cut list that is not
            0.
        n_eval_samples : int
            Number of emulator evaluation samples used for implausibility
            evaluations.

        """

        # Do some logging
        logger = logging.getLogger('INIT')
        logger.info("Obtaining implausibility analysis parameters.")

        # Obtaining default pipeline parameter dict
        par_dict = self._get_default_parameters()

        # Read in data from provided PRISM parameters file
        if self._prism_file is not None:
            pipe_par = np.genfromtxt(self._prism_file, dtype=(str),
                                     delimiter=': ', autostrip=True)

            # Make sure that pipe_par is 2D
            pipe_par = np.array(pipe_par, ndmin=2)

            # Combine default parameters with read-in parameters
            par_dict.update(pipe_par)

        # Implausibility cut-off
        impl_cut_str = str(par_dict['impl_cut']).replace(',', '').split()
        self._get_impl_cut(
            self, list(float(impl_cut) for impl_cut in impl_cut_str))

        # Number of samples used for implausibility evaluations
        n_eval_samples = int(par_dict['n_eval_samples'])
        self._save_data('n_eval_samples',
                        check_pos_int(n_eval_samples, 'n_eval_samples'))

        # Finish logging
        logger.info("Finished obtaining implausibility analysis parameters.")


# %% VISIBLE CLASS METHODS
    # This function analyzes the emulator and determines the plausible regions
    # TODO: Implement check if impl_idx is big enough to be used in next emul_i
    # HINT: Allow analyze to be used on earlier iterations?
    def analyze(self, emul_i=None):
        """
        Analyzes the emulator system at the specified emulator iteration
        `emul_i` for a large number of emulator evaluation samples. All samples
        that survive the implausibility checks, are used in the construction of
        the next emulator iteration.

        Optional
        --------
        emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the last iteration of the loaded emulator system will
            be used.

        Generates
        ---------
        impl_sam : 2D :obj:`~numpy.ndarray` object
            Array containing all emulator evaluation samples that survived the
            implausibility checks.
        prc : bool
            Bool indicating whether or not plausible regions have been found
            during this analysis.

        """

        # Begin logging
        logger = logging.getLogger('ANALYZE')
        logger.info("Analyzing emulator system at iteration %s." % (emul_i))

        # Check emul_i
        if emul_i is None:
            emul_i = self._emulator._emul_i
        elif not(emul_i == self._emulator._emul_i):
            logger.error("Reanalysis of the emulator system is only possible "
                         "on the last emulator iteration created (%s)!"
                         % (self._emulator._emul_i))
            raise RequestError("Reanalysis of the emulator system is only "
                               "possible on the last emulator iteration "
                               "created (%s)!"
                               % (self._emulator._emul_i))
        else:
            emul_i = check_pos_int(emul_i, 'emul_i')

        # Get the impl_cut list and n_eval_samples
        self._get_impl_par(emul_i)

        # Create an emulator evaluation sample set
        eval_sam_set = self._get_eval_sam_set(emul_i)

        # Obtain number of samples
        n_samples = eval_sam_set.shape[0]

        # Create empty list holding indices of samples that pass the impl_check
        impl_idx = []

        # Calculate expectation, variance, implausibility for these samples
        for i in range(n_samples):
            for j in range(1, emul_i+1):
                # Obtain implausibility
                adj_val = self._emulator._evaluate(j, eval_sam_set[i])
                uni_impl_val = self._get_uni_impl(j, *adj_val)

                # Do implausibility cut-off check
                if self._do_impl_check(self, j, uni_impl_val)[0] is False:
                    break
            else:
                impl_idx.append(i)

        # Save the results
        self._save_data('impl_sam', eval_sam_set[impl_idx])

        # Log that analysis has been finished
        logger.info("Finished analysis of emulator system.")

        # Display details about current state of pipeline
        self.details(emul_i)

    # This function constructs a specified iteration of the emulator system
    def construct(self, emul_i=None, analyze=True):
        """
        Constructs the emulator at the specified emulator iteration `emul_i`,
        and performs an implausibility analysis on the emulator system right
        afterward if requested (:meth:`~analyze`).

        Optional
        --------
        emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the last iteration of the loaded emulator system will
            be used.

        Generates
        ---------
        A new group with the emulator iteration value as its name, in the
        loaded emulator file, containing emulator data.

        Notes
        -----
        Using an emulator iteration that has been constructed before, will
        remove that and all following iterations, and reconstruct the specified
        iteration. Using `emul_i` = 1 is equivalent to reconstructing the whole
        emulator system.

        """

        # Log that a new emulator iteration is being constructed
        logger = logging.getLogger('CONSTRUCT')

        # Set emul_i correctly
        if emul_i is None:
            emul_i = self._emulator._emul_i+1
        else:
            emul_i = check_pos_int(emul_i, 'emul_i')

        # Log that construction of emulator iteration is being started
        logger.info("Starting construction of emulator iteration %s."
                    % (emul_i))

        # Check emul_i and act accordingly
        if(emul_i == 1):
            # Create mock data if requested
            if self._use_mock:
                self._get_mock_data(0)

            # Create a new emulator system
            self._emulator._create_new_emulator(self._use_mock)

            # Reload the data
            self._load_data()

            # Create initial set of model evaluation samples
            add_sam_set = lhd(self._n_sam_init, self._modellink._par_dim,
                              self._modellink._par_rng, 'fixed',
                              self._criterion)

        else:
            # Check if a new emulator iteration can be constructed
            if not self._prc[emul_i-1]:
                logger.error("No plausible regions were found in the analysis "
                             "of the previous emulator iteration. Construction"
                             " is not possible!")
                raise RequestError("No plausible regions were found in the "
                                   "analysis of the previous emulator "
                                   "iteration. Construction is not possible!")

            # Make the emulator prepare for a new iteration
            reload = self._emulator._prepare_new_iteration(emul_i)

            # Make sure the correct pipeline data is loaded in
            if reload:
                self._load_data()

            # Obtain additional sam_set
            add_sam_set = self._impl_sam[emul_i-1]

        # Obtain corresponding set of model evaluations
        self._evaluate_model(emul_i, add_sam_set)

        # Determine active parameters
        self._get_active_par(emul_i)

        # Construct emulator
        self._emulator._construct_iteration(emul_i)
        self._emulator._emul_i = emul_i

        # Analyze the emulator system if requested
        if analyze:
            self.analyze(emul_i)
        else:
            self._save_data('impl_sam', [])
            self.details(emul_i)

    # This function creates the projection figures of a given emul_i
    @docstring_copy(Projection.__call__)
    def create_projection(self, emul_i=None, proj_par=None, figure=True,
                          show=False, force=False):

        # Initialize the Projection class and make the figures
        Projection(self)(emul_i, proj_par, figure, show, force)

    # This function allows one to obtain the pipeline details/properties
    def details(self, emul_i=None):
        """
        Prints the details/properties of the currently loaded pipeline instance
        at given emulator iteration `emul_i`.

        Optional
        --------
        emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the last iteration of the loaded emulator system will
            be used.

        """

        # Define details logger
        logger = logging.getLogger("DETAILS")
        logger.info("Collecting details about current pipeline instance.")

        # Check what kind of hdf5-file was provided
        try:
            emul_i = self._emulator._get_emul_i(emul_i)
        except RequestError:
            return
        else:
            n_impl_sam = len(self._impl_sam[emul_i])
            n_eval_samples = self._get_n_eval_samples(emul_i)
            name_len =\
                max([len(par_name) for par_name in self._modellink._par_names])
            lower_len =\
                max([len(str(i)) for i in self._modellink._par_rng[:, 0]])
            upper_len =\
                max([len(str(i)) for i in self._modellink._par_rng[:, 1]])

            # Open hdf5-file
            file = self._open_hdf5('r')

            # Check if mock_data was used by trying to access mock_par
            try:
                file.attrs['mock_par']
            except KeyError:
                use_mock = 0
            else:
                use_mock = 1

            # Check if projection data is available
            try:
                file['%s/proj_hcube' % (emul_i)]
            except KeyError:
                proj = 0

            # If projection data is available
            else:
                proj_impl_cut =\
                    file['%s/proj_hcube' % (emul_i)].attrs['impl_cut']
                proj_cut_idx =\
                    file['%s/proj_hcube' % (emul_i)].attrs['cut_idx']

                # Check if projection was made with the same impl_cut
                try:
                    # If it was, projection is synced
                    if((proj_impl_cut == self._impl_cut[emul_i]).all() and
                       proj_cut_idx == self._cut_idx[emul_i]):
                        proj = 1

                    # If not, projection is desynced
                    else:
                        proj = 2

                # If analysis was never done, projection is considered synced
                except IndexError:
                    proj = 1

        # Close hdf5-file
        self._close_hdf5(file)

        # Log file being closed
        logger.info("Finished collecting details about current pipeline "
                    "instance.")

        # Print details about hdf5-file provided
        width = 30
        print("\n")
        print("PIPELINE DETAILS")
        print("="*width)
        print("\nGeneral")
        print("-"*width)
        print("{0: <{1}}\t'{2}'".format("HDF5-file name", width,
                                        self._hdf5_file_name))
        print("{0: <{1}}\t{2}".format("ModelLink subclass", width,
                                      self._modellink_name))
        if(self._emulator._method.lower() == 'regression'):
            print("{0: <{1}}\t{2}".format("Emulation type", width,
                                          "Regression"))
        elif(self._emulator._method.lower() == 'gaussian'):
            print("{0: <{1}}\t{2}".format("Emulation type", width,
                                          "Gaussian"))
        elif(self._emulator._method.lower() == 'full'):
            print("{0: <{1}}\t{2}".format("Emulation type", width,
                                          "Regression + Gaussian"))
        print("{0: <{1}}\t{2}".format("Emulator iteration", width, emul_i))
        if not bool(self._prc[emul_i]):
            print("{0: <{1}}\t{2}".format("Plausible regions?", width,
                                          "No"))
        else:
            print("{0: <{1}}\t{2}".format("Plausible regions?", width,
                                          "Yes"))
        if(self._emulator._emul_load == 0 or proj == 0):
            print("{0: <{1}}\t{2}".format("Projection available?", width,
                                          "No"))
        elif(proj == 1):
            print("{0: <{1}}\t{2}".format("Projection available?", width,
                                          "Yes"))
        else:
            print("{0: <{1}}\t{2}".format("Projection available?", width,
                                          "Yes (desynced)"))
        if(use_mock):
            print("{0: <{1}}\t{2}".format("Mock data used?", width,
                                          "Yes"))
        else:
            print("{0: <{1}}\t{2}".format("Mock data used?", width,
                                          "No"))
        print("-"*width)
        print("{0: <{1}}\t{2}".format("# of model evaluation samples", width,
                                      self._emulator._n_sam[1:emul_i+1]))
        if self._emulator._emul_load:
            print("{0: <{1}}\t{2}".format("# of plausible samples",
                                          width, n_impl_sam))
            print("{0: <{1}}\t{2:.3g}%".format(
                    "% of parameter space remaining", width,
                    (n_impl_sam/n_eval_samples)*100))
        print("{0: <{1}}\t{2}".format("# of model parameters", width,
                                      self._modellink._par_dim))
        print("{0: <{1}}\t{2}".format("# of active model parameters", width,
                                      len(self._emulator._active_par[emul_i])))
        print("{0: <{1}}\t{2}".format("# of data points", width,
                                      self._modellink._n_data))
        print("-"*width)
        print("\nParameter space")
        print("-"*width)
        str_format = "{0: >{1}}: [{2: >{3}}, {4: >{5}}] ({6:.5g})"
        for i in range(self._modellink._par_dim):
            print(str_format.format(self._modellink._par_names[i], name_len,
                                    self._modellink._par_rng[i, 0], lower_len,
                                    self._modellink._par_rng[i, 1], upper_len,
                                    self._modellink._par_estimate[i]))
        print("="*width)
