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
from ._internal import RequestError, move_logger, start_logger
from .emulator import Emulator
from .modellink import ModelLink

# All declaration
__all__ = ['Pipeline']


# %% PIPELINE CLASS DEFINITION
# TODO: Write pydocs
class Pipeline(object):
    """
    Defines the :class:`~Pipeline` class of the PRISM package.

    """

    def __init__(self, model_link, root_dir=None, working_dir=None,
                 prefix='emul_', emulator_hdf5_file='emulator.hdf5',
                 prism_file='prism.txt'):
        """
        Initialize an instance of the :class:`~Pipeline` class.

        Parameters
        ----------
        model_link : :obj:`~ModelLink` object
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
        emulator_hdf5_file : str. Default: 'emulator.hdf5'
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
        self._get_paths(root_dir, working_dir, prefix, emulator_hdf5_file,
                        prism_file)

        # Move logger to working directory
        move_logger(self._working_dir)

        # Initialize Emulator class
        self._emulator = Emulator(self._emulator_hdf5_file)


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
    def emulator_hdf5_file(self):
        """
        Absolute path to the loaded emulator HDF5-file.

        """

        return(self._emulator_hdf5_file)

    @property
    def emulator_hdf5_file_name(self):
        """
        Name of loaded emulator HDF5-file.

        """

        return(self._emulator_hdf5_file_name)

    @property
    def prism_file(self):
        """
        Absolute path to PRISM parameters file.

        """

        return(self._prism_file)

    @property
    def model_link(self):
        """
        The :obj:`~ModelLink` instance provided during Pipeline initialization.

        """

        return(self._model_link)

    @model_link.setter
    def model_link(self, model_link):
        """
        Sets the `model_link` property.

        Parameters
        ----------
        model_link : :obj:`~ModelLink` object
            Instance of the :class:`~ModelLink` class that links the emulated
            model to this :obj:`~Pipeline` object.
            The provided :obj:`~ModelLink` object must be the same as the one
            used to construct the loaded emulator system.

        """

        # Check if a subclass of the ModelLink class has been provided
        if not isinstance(model_link, ModelLink):
            raise TypeError("Input argument 'model_link' must be an instance "
                            "of the ModelLink class!")

        # If an existing emulator system is loaded, check if classes are equal
        if self._emulator._emul_load:
            if(model_link.__class__.__name__ != self._model_link_name):
                raise InputError("Provided ModelLink subclass %s is not equal "
                                 "to ModelLink subclass %s used for emulator "
                                 "construction!"
                                 % (model_link.__class__.__name__,
                                    self._model_link_name))
        # If not, set the name
        else:
            self._model_link_name = model_link.__class__.__name__

        # Set ModelLink class
        self._model_link = model_link

    @property
    def model_link_name(self):
        """
        Name of the :obj:`~ModelLink` instance provided during Pipeline
        initialization.

        """

        return(self._model_link_name)

    @property
    def emulator(self):
        """
        The :obj:`~Emulator` instance created during Pipeline initialization.

        """

        return(self._emulator)


# %% GENERAL CLASS METHODS
    # Obtains the paths for the root directory, working directory, emulator
    # hdf5-file and prism parameters file
    def _get_paths(self, root_dir, working_dir, prefix, emulator_hdf5_file,
                   prism_file):
        """
        Obtains the path for the root directory, working directory, HDF5-file
        and parameters file for the Pipeline.

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
        emulator_hdf5_file : str
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
        The absolute paths to the root directory, working directory, emulator
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
        if isinstance(emulator_hdf5_file, str):
            self._emulator_hdf5_file = path.join(self._working_dir,
                                                 emulator_hdf5_file)
            logger.info("HDF5-file set to '%s'." % (emulator_hdf5_file))
            self._emulator_hdf5_file_name = path.join(working_dir,
                                                      emulator_hdf5_file)
        else:
            raise TypeError("Input argument 'emulator_hdf5_file' is not a "
                            "string!")

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
