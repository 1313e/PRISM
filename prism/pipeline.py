# -*- coding: utf-8 -*-

"""
Pipeline
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
from _internal import RequestError, move_logger, start_logger
from emulator import Emulator
from modellink import ModelLink

# All declaration
__all__ = ['Pipeline']


# %% PIPELINE CLASS DEFINITION
# TODO: Write pydocs
class Pipeline(object):
    """
    Defines the :class:`~Pipeline` class of the PRISM package.

    """

    def __init__(self, modellink, root_dir=None, working_dir=None,
                 prefix='emul_', hdf5_file='prism.hdf5',
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
        prefix : str. Default: 'emul_'
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
        self._emulator = Emulator(self._prism_file, self._open_hdf5,
                                  self._close_hdf5)

        # Link provided ModelLink subclass to Emulator class
        self._emulator.modellink = modellink

        # Link provided ModelLink subclass to Pipeline class
        self._modellink = self._emulator._modellink
        self._modellink_name = self._emulator._modellink_name

        # Read in pipeline parameters
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
            If a constructed emulator file was provided during class
            initialization, the iteration of the emulator corresponding to
            `emul_i` will be used automatically.
            If *None*, the last iteration of the currently loaded emulator will
            be used.

        """

        # Check the provided emul_i
        if emul_i is None:
            emul_i = self._emulator._emul_i+1

        # Perform construction
        try:
            self.construct(emul_i)
        except Exception:
            raise
        else:
            if self._emulator._prc[emul_i]:
                self.create_projection(emul_i)
                self.details(emul_i)


# %% CLASS PROPERTIES
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
        Bool determining whether or not to use active parameters.

        """

        return(bool(self._do_active_par))

    @property
    def n_sam_init(self):
        """
        Initial number of model evaluation samples in currently loaded emulator
        instance.

        """

        return(self._n_sam_init)

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
    def n_eval_samples(self):
        """
        Base number of emulator evaluations used to scan the plausible
        parameter region and update the emulator. This number is scaled up by
        the number of model parameters and the current emulator iteration to
        generate the true number of emulator evaluations.

        """

        return(self._n_eval_samples)

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

    @property
    def prc(self):
        """
        Bool indicating whether or not plausible regions have been found in the
        currently loaded emulator iteration.

        """

        return([bool(self._prc[i]) for i in range(self._emul_i+1)])

    @property
    def impl_sam(self):
        """
        Array containing all model evaluation samples that will be added to the
        next emulator iteration.

        """

        return(self._impl_sam)


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
            Number indicating the current emulator iteration.
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

        # Open hdf5-file
        file = self._open_hdf5('r')

        # Read in all the pipeline attributes
        self._n_sam_init = file.attrs['n_sam_init']
        self._n_proj_samples = file.attrs['n_proj_samples']
        self._n_hidden_samples = file.attrs['n_hidden_samples']
        self._n_eval_samples = file.attrs['n_eval_samples']
        self._do_active_par = file.attrs['do_active_par']
        self._criterion = file.attrs['criterion'].decode('utf-8')
        # TODO: Think about how to make these dynamic
        self._impl_cut = file.attrs['impl_cut']
        self._cut_idx = file.attrs['cut_idx']

        # Close hdf5-file
        self._close_hdf5(file)

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
                    'n_proj_samples': '15',
                    'n_hidden_samples': '75',
                    'n_eval_samples': '600',
                    'criterion': "'multi'",
                    'do_active_par': 'True',
                    'impl_cut': '0, 4.0, 3.8, 3.5'}

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
        self._n_sam_init = int(par_dict['n_sam_init'])

        # Number of samples used for implausibility evaluations
        self._n_proj_samples = int(par_dict['n_proj_samples'])
        self._n_hidden_samples = int(par_dict['n_hidden_samples'])
        self._n_eval_samples = int(par_dict['n_eval_samples'])

        # Set non-default parameter estimate
        self._modellink._par_estimate = self._modellink._par_rng[:, 0] +\
            random(self._modellink._par_dim) *\
            (self._modellink._par_rng[:, 1]-self._modellink._par_rng[:, 0])

        # Set non-default model data
        self._modellink._data_val =\
            self._call_model(0, self._modellink._par_estimate).tolist()
        self._modellink._data_err =\
            (0.1*np.abs(self._modellink._data_val)).tolist()
        self._modellink._data_val =\
            (self._modellink._data_val +
             normal(scale=self._modellink._data_err)).tolist()

        # Criterion parameter used for Latin Hypercube Sampling
        self._criterion = str(par_dict['criterion']).replace("'", '')

        # Obtain the bool determining whether or not to have active parameters
        if(par_dict['do_active_par'].lower() in ('false', '0')):
            self._do_active_par = 0
        elif(par_dict['do_active_par'].lower() in ('true', '1')):
            self._do_active_par = 1
        else:
            raise TypeError("Pipeline parameter 'do_active_par' is not of type"
                            " 'bool'.")

        # Log that reading has been finished
        logger.info("Finished reading pipeline parameters.")

    # This function controls how n_eval_samples is calculated
    def _get_n_eval_samples(self, emul_i):
        """
        This function calculates the total amount of emulator evaluation
        samples at a given emulator iteration `emul_i` from the
        `n_eval_samples` provided during class initialization.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.

        Returns
        -------
        n_eval_samples : int
            Number of emulator evaluation samples.

        """

        # Calculate n_eval_samples
        return(emul_i*self._n_eval_samples*self._modellink._par_dim)

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
            raise TypeError("Input argument 'root_dir' must be a string!")

        # Check if a valid working directory prefix string is given
        if isinstance(prefix, str):
            self._prefix = prefix
            prefix_len = len(prefix)
        else:
            raise TypeError("Input argument 'prefix' must be a string!")

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
            raise TypeError("Input argument 'working_dir' is not a string!")

        # Obtain hdf5-file path
        if isinstance(hdf5_file, str):
            self._hdf5_file = path.join(self._working_dir, hdf5_file)
            logger.info("HDF5-file set to '%s'." % (hdf5_file))
            self._hdf5_file_name = path.join(working_dir, hdf5_file)
        else:
            raise TypeError("Input argument 'hdf5_file' is not a string!")

        # Obtain PRISM parameter file path
        if prism_file is None:
            self._prism_file = None
        elif isinstance(prism_file, str):
            if path.exists(prism_file):
                self._prism_file = prism_file
            else:
                self._prism_file = path.join(self._root_dir, prism_file)
            logger.info("PRISM parameters file set to '%s'." % (prism_file))
        else:
            raise TypeError("Input argument 'prism_file' is not a string!")

    # This function loads pipeline data
    def _load_data(self):
        """
        Loads in all the important pipeline data corresponding to emulator
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
        All relevant pipeline data corresponding to emulator iteration `emul_i`
        is loaded into memory, if not loaded already.

        """

        # Set the logger
        logger = logging.getLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing emulator data sets.")
        self._prc = [[]]
        self._impl_sam = [[]]
        self._impl_cut = [[]]
        self._cut_idx = [[]]

        if self._emulator._emul_i:
            # Open hdf5-file
            file = self._open_hdf5('r')

            # Read in the data
            for i in range(1, self._emulator._emul_i+1):
                self._prc.append(file['%s' % (i)].attrs['prc'])
                self._impl_sam.append(file['%s/impl_sam' % (i)][()])
                self._impl_cut.append(file['%s' % (i)].attrs['impl_cut'])
                self._cut_idx.append(file['%s' % (i)].attrs['cut_idx'])

            # Close hdf5-file
            self._close_hdf5(file)

    # This function saves pipeline data to hdf5
    def _save_data(self, keyword, data):
        """
        Saves the provided `data` for the specified data-type `keyword` at the
        last emulator iteration to the HDF5-file and as an data
        attribute to the current :obj:`~Pipeline` instance if required.

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
                    % (keyword, self._emulator._emul_i))

        # Open hdf5-file
        file = self._open_hdf5('r+')

        # Check what data keyword has been provided
        # IMPL_SAM
        if keyword in ('impl_sam'):
            # Check if any plausible regions have been found at all
            if(np.shape(data)[0] == 0):
                prc = 0
            else:
                prc = 1

            # Either update or save new prc and impl_sam values
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

        # IMPL_CUT
        elif keyword in ('impl_cut'):
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
            Number indicating the current emulator iteration.

        Generates
        ---------
        new_mod_set : 2D :obj:`~numpy.ndarray` object
            New array containing the model outputs of all specified model
            evaluation samples.

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
            Number indicating the current emulator iteration.

        Generates
        ---------
        active_par : 1D list
            List containing the indices of all the parameters that are active
            in the emulator iteration `emul_i`.
        active_par_data : 2D list
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
            Number indicating the current emulator iteration.

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
    def _do_impl_check(self, emul_i, uni_impl_val):
        """
        Performs an implausibility cut-off check on the provided implausibility
        values `uni_impl_val`.

        Parameters
        ----------
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
        impl_cut_val = sorted_impl_val[self._cut_idx[emul_i]]

        # Scan over all data points in this sample
        for i, val in enumerate(self._impl_cut[emul_i]):
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
            Number indicating the current emulator iteration.
        adj_exp_val, adj_var_val : 1D array_like
            The adjusted expectation and variance values to calculate the
            univeriate implausibility for.

        Returns
        -------
        uni_impl_val : 1D :obj:`~numpy.ndarray` object
            Univariate implausibility value for every data point.

        """

        # Log that univariate implausibility value is calculated
        logger = logging.getLogger('EMUL_IMPL')
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
        that are created by the model provided with the :obj:`~ModelLink`
        instance. This method tries to call the :func:`~ModelLink.get_md_var`
        method, and assumes a default model discrepancy variance of 1/6th the
        data value if it cannot be called.

        Parameters
        ----------
        emul_i : int
            Number indicating the current emulator iteration.

        Returns
        -------
        var_md : 1D :obj:`~numpy.ndarray` object
            Variance of the model discrepancy.

        """

        # Initialize model discrepancy variance
        md_var = np.zeros(self._emulator._n_data[emul_i])

        # Obtain md variances
        # Try to use the user-defined md variances
        try:
            md_var +=\
                self._modellink.get_md_var(emul_i,
                                           self._emulator._data_idx[emul_i])
        # If it was not user-defined, use a default value
        except NotImplementedError:
            # Use factor 2 difference on 2 sigma as acceptable
            # Imagine that 2 sigma range is given if lower and upper are factor
            # 2 apart. This gives that sigma must be 1/6th of the data value
            md_var += pow(np.array(self._emulator._data_val[emul_i])/6, 2)

        # Return it
        return(md_var)

    # This function reads in the impl_cut list from the PRISM parameters file
    def _get_impl_cut(self, emul_i):
        """
        Reads in the impl_cut list from the PRISM parameters file.

        """

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
        self.impl_cut = list(float(impl_cut) for impl_cut in impl_cut_str)


# %% VISIBLE CLASS METHODS
    # This function analyzes the emulator and determines the plausible regions
    def analyze(self, emul_i=None):
        """
        Analyzes the emulator system at the specified emulator iteration
        `emul_i`.

        """

        # Check emul_i
        if emul_i is None:
            emul_i = self._emulator._emul_i
        elif not(emul_i == self._emulator._emul_i):
            raise RequestError("Reanalysis of the emulator system is only "
                               "possible on the last emulator iteration "
                               "created (%s)!" % (self._emulator._emul_i))

        # Begin logging
        logger = logging.getLogger('ANALYZE')
        logger.info("Analyzing emulator system at iteration %s." % (emul_i))

        # Create an emulator evaluation sample set
        eval_sam_set = self._get_eval_sam_set(emul_i)

        # Obtain number of samples
        n_samples = eval_sam_set.shape[0]

        # Get the impl_cut list
        self._get_impl_cut(emul_i)

        # Create empty list holding indices of samples that pass the impl_check
        impl_idx = []

        # Calculate expectation, variance, implausibility for these samples
        for i in range(n_samples):
            for j in range(1, emul_i+1):
                # Obtain implausibility
                adj_val = self._emulator._evaluate(j, eval_sam_set[i])
                uni_impl_val = self._get_uni_impl(j, *adj_val)

                # Do implausibility cut-off check
                if self._do_impl_check(j, uni_impl_val)[0] is False:
                    break
            else:
                impl_idx.append(i)

        # Save the results
        self._save_data('impl_sam', eval_sam_set[impl_idx])

        # Log that analysis has been finished
        logger.info("Finished evaluation sample set analysis.")

        # Display details about current state of pipeline
        self.details(emul_i)

    # This function constructs a specified iteration of the emulator system
    def construct(self, emul_i=None):
        """
        Constructs the emulator at the specified emulator iteration `emul_i`.

        Using an emulator iteration that has been constructed before, will
        remove that and all following iterations, and reconstruct the specified
        iteration. Using `emul_i` = 1 will also make the class read in the
        parameters from the parameter file again.

        If a constructed emulator iteration shows signs that something is
        wrong or the emulator cannot be optimized anymore, it will not save
        this data in the emulator construction HDF5-file, but rather makes an
        emulator final HDF5-file of the previous emulator iteration and prints
        out a warning of this event both in the console and in the log-file.

        Optional
        --------
        emul_i : int or None. Default: None
            If an existing constructor file was provided during class
            initialization, the iteration of the emulator corresponding to
            `emul_i` will be used automatically.
            If *None*, the last iteration of the currently loaded emulator will
            be used.

        Generates
        ---------
        A new group with the emulator iteration value as its name, in the
        loaded emulator construction file, containing emulator data.
        The data included is required to either construct the next iteration or
        to create the projection hypercube.

        If applicable, an emulator final HDF5-file called 'emulator_final.hdf5'
        (default) containing the same data as the last folder in
        'emulator.hdf5'. This file is write-protected when used in this class.
        The creation of this file will also automatically trigger the creation
        of the projection hypercube.

        """

        # Log that a new emulator iteration is being constructed
        logger = logging.getLogger('CONSTRUCT')

        # Set emul_i correctly
        if emul_i is None:
            emul_i = self._emulator._emul_i+1

        # Log that construction of emulator iteration is being started
        logger.info("Starting construction of emulator iteration %s."
                    % (emul_i))

        # Check emul_i and act accordingly
        if(emul_i == 1):
            # Create a new emulator system
            self._emulator._create_new_emulator()

            # Reload the data
            self._load_data()

            # Create initial set of model evaluation samples
            add_sam_set = lhd(self._n_sam_init, self._modellink._par_dim,
                              self._modellink._par_rng, 'fixed',
                              self._criterion)

        else:
            # Check if a new emulator iteration can be constructed
            if not self._prc[emul_i-1]:
                raise RequestError("No plausible regions were found in "
                                   "previous emulator iteration. Construction "
                                   "is not possible!")

            # Make the emulator prepare for a new iteration
            result = self._emulator._prepare_new_iteration(emul_i)

            # Make sure the correct pipeline data is loaded in
            if result:
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

        # Analyze the emulator system
        self.analyze(emul_i)

    # This function allows one to obtain the emulator details/properties
    def details(self, emul_i=None):
        """
        Prints the details/properties of the currently loaded emulator at given
        emulator iteration `emul_i`.

        Optional
        --------
        emul_i : int or None. Default: None
            If an existing constructor file was provided during class
            initialization, the iteration of the emulator corresponding to
            `emul_i` will be used automatically.
            If *None*, the last iteration of the currently loaded emulator will
            be used.

        """

        # Define details logger
        logger = logging.getLogger("DETAILS")
        logger.info("Collecting details about current emulator file.")

        # Check what kind of hdf5-file was provided
        try:
            emul_i = self._emulator._get_emul_i(emul_i)
        except RequestError:
            return
        else:
            n_impl_sam = len(self._impl_sam[emul_i])
            n_eval_samples = self._get_n_eval_samples(emul_i)

            # Open hdf5-file
            file = self._open_hdf5('r')

            try:
                file['%s/proj_hcube' % (emul_i)]
            except KeyError:
                proj = 0
            else:
                proj = 1

        # Close hdf5-file
        self._close_hdf5(file)

        # Print details about hdf5-file provided
        width = 30
        print("\n")
        print("DETAILS CURRENT PIPELINE INSTANCE")
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
        else:
            print("{0: <{1}}\t{2}".format("Projection available?", width,
                                          "Yes"))
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
        for i in range(self._modellink._par_dim):
            print("%s: %s" % (self._modellink._par_names[i],
                              self._modellink._par_rng[i]))
        print("="*width)
