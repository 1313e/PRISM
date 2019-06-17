# -*- coding: utf-8 -*-

"""
Pipeline
========
Provides the definition of the main class of the *PRISM* package, the
:class:`~Pipeline` class.

"""


# %% IMPORTS
# Built-in imports
from ast import literal_eval
from contextlib import contextmanager
from inspect import isclass
import logging
import os
from os import path
import sys
from textwrap import dedent
from time import time

# Package imports
from e13tools import InputError, ShapeError
from e13tools.math import nCr
from e13tools.sampling import lhd
from e13tools.utils import (convert_str_seq, delist, docstring_append,
                            docstring_copy, docstring_substitute,
                            get_outer_frame, raise_error, raise_warning)
from mpi4pyd import MPI
from mpi4pyd.MPI import get_HybridComm_obj
import numpy as np
from numpy.random import normal, random
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._docstrings import (call_emul_i_doc, call_model_doc_s,
                               call_model_doc_m, def_par_doc, ext_mod_set_doc,
                               ext_real_set_doc_d, ext_real_set_doc_s,
                               ext_sam_set_doc, impl_cut_doc, paths_doc_d,
                               paths_doc_s, save_data_doc_p, set_par_doc,
                               std_emul_i_doc, user_emul_i_doc)
from prism._internal import (RequestError, RequestWarning, check_vals,
                             getCLogger, get_PRISM_File, getRLogger,
                             move_logger, np_array, set_base_logger)
from prism._projection import Projection
from prism.emulator import Emulator

# All declaration
__all__ = ['Pipeline']


# %% PIPELINE CLASS DEFINITION
# TODO: Allow user to switch between emulation and modelling
# TODO: Implement multivariate implausibilities
# TODO: Think of a way to allow no ModelLink instance to be provided.
# This could be done with a DummyLink, but md_var is then uncallable.
# OPTIMIZE: Use the Numba package to speed up certain calculations?
class Pipeline(Projection, object):
    """
    Defines the :class:`~Pipeline` class of the *PRISM* package.

    The :class:`~Pipeline` class is the main user class of the *PRISM* package
    and provides a user-friendly environment that gives access to all
    operations within the package.

    """

    @docstring_substitute(paths=paths_doc_d)
    def __init__(self, modellink_obj, *, root_dir=None, working_dir=None,
                 prefix=None, prism_par=None, emul_type=None, comm=None,
                 **kwargs):
        """
        Initialize an instance of the :class:`~Pipeline` class.

        Parameters
        ----------
        modellink_obj : :obj:`~prism.modellink.ModelLink` object
            Instance of the :class:`~prism.modellink.ModelLink` subclass
            that links the emulated model to this :obj:`~Pipeline` instance.

        Optional
        --------
        %(paths)s
        prism_par : array_like, dict, str or None. Default: None
            A dict containing the values for the *PRISM* parameters that need
            to be changed from their default values.
            If array_like, dict(`prism_par`) must generate a dict with the
            correct lay-out.
            If str, the string is the absolute or relative path to the file
            that contains the keys in the first column and the dict values in
            the second column. If a relative path is given, its path must be
            relative to `root_dir` or the current directory.
            If *None*, no changes will be made to the default parameters.
        emul_type : :class:`~prism.emulator.Emulator` subclass or None.\
            Default: None
            The type of :class:`~prism.emulator.Emulator` to use in this
            :obj:`~Pipeline` instance. If *None*, use the default emulator
            instead.
        comm : :obj:`~mpi4pyd.MPI.Intracomm` object or None. Default: None
            The MPI intra-communicator to use in this :obj:`~Pipeline`
            instance or :obj:`MPI.COMM_WORLD` if `comm` is *None*.

        """

        # Obtain MPI communicator, ranks and sizes
        self._comm = get_HybridComm_obj(comm)
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        # Set statuses
        self._is_controller = 1 if not self._rank else 0
        self._is_worker = 1 if self._rank else 0
        self._worker_mode = 0
        self._do_logging = 1

        # Controller obtaining paths and preparing logging system
        if self._is_controller:
            # Start logging
            set_base_logger()
            logger = getCLogger('PIPELINE')
            logger.info("")

            # Initialize class
            logger = getCLogger('INIT')
            logger.info("Initializing Pipeline class.")

            # Obtain paths
            self._get_paths(root_dir, working_dir, prefix)

            # Move logger to working directory and restart it
            move_logger(self._working_dir)

        # Remaining workers obtain paths from controller
        else:
            # Obtain paths
            logger = getCLogger('INIT')
            self._get_paths(root_dir, working_dir, prefix)

        # MPI Barrier
        self._comm.Barrier()

        # Start logger for workers as well
        if self._is_worker:
            set_base_logger(path.join(self._working_dir, 'prism_log.log'))

        # Check if deprecated prism_file was provided
        if 'prism_file' in kwargs.keys():
            # Assign prism_file to prism_par
            prism_par = kwargs['prism_file']

            # Raise a FutureWarning
            if self._is_controller:
                warn_msg = ("Input argument 'prism_file' is deprecated since "
                            "v1.1.2 in favor of 'prism_par'. It will be "
                            "removed entirely in v1.2.0.")
                raise_warning(warn_msg, FutureWarning, logger, stacklevel=2)

        # Read in the provided parameter dict/file
        self._read_parameters(prism_par)

        # Initialize Emulator class
        # If emul_type is None, use default emulator
        if emul_type is None:
            Emulator(self, modellink_obj)

        # Else if emul_type is a subclass of Emulator, try to initialize it
        elif isclass(emul_type) and issubclass(emul_type, Emulator):
            try:
                emul_type(self, modellink_obj)
                self._emulator
                self._modellink
            except Exception as error:
                err_msg = ("Input argument 'emul_type' is invalid! (%s)"
                           % (error))
                raise_error(err_msg, InputError, logger)

        # If anything else is given, it is invalid
        else:
            err_msg = ("Input argument 'emul_type' is invalid (%r)!"
                       % (emul_type))
            raise_error(err_msg, InputError, logger)

        # Let everybody set the pipeline parameters
        self._set_parameters()

        # Compile pre-defined code snippets
        self._compile_code_snippets()

        # Let controller load in the data
        if self._is_controller:
            self._load_data()

        # Print out the details of the current state of the pipeline
        self.details()

    # Allows one to call one full loop of the PRISM pipeline
    @docstring_substitute(emul_i=call_emul_i_doc)
    def __call__(self, emul_i=None, *, force=False):
        """
        Calls the :meth:`~construct` method to start the construction of the
        given iteration of the emulator and creates the projection figures
        right afterward if this construction was successful.

        Optional
        --------
        %(emul_i)s
        force : bool. Default: False
            Controls what to do if the specified emulator iteration `emul_i`
            already (partly) exists.
            If *False*, finish construction of the specified iteration or skip
            it if already finished.
            If *True*, reconstruct the specified iteration entirely.

        """

        # Perform construction
        self.construct(emul_i, force=force)

        # Perform projection
        self.project()

    # Define the representation of a Pipeline object
    # TODO: Find out if there is a way to represent an MPI intra-communicator
    def __repr__(self):
        # Get path to current working directory, make all paths relative to it
        pwd = os.getcwd()

        # Make empty list holding representations of all input arguments
        str_repr = []

        # Add the ModelLink representation
        str_repr.append(repr(self._modellink))

        # Add the root_dir representation if it is not default
        if(path.splitdrive(self._root_dir)[0].lower() !=
           path.splitdrive(pwd)[0].lower()):
            rel_root_path = self._root_dir
        else:
            rel_root_path = path.relpath(self._root_dir, pwd)

        if(rel_root_path != '.'):
            str_repr.append("root_dir=%r" % (rel_root_path))

        # Add the working_dir representation
        str_repr.append("working_dir=%r" % (path.relpath(self._working_dir,
                                                         self._root_dir)))

        # Add the prism_par representation if it is not default
        if self._prism_dict:
            # Make empty default dict
            default_dict = sdict()

            # Make empty list of prism_par representations
            par_repr = []

            # Add the default parameter dicts to it
            default_dict.update(self._get_default_parameters())
            default_dict.update(self._emulator._get_default_parameters())
            default_dict.update(self._Projection__get_default_parameters())

            # Loop over all items in prism_dict and check if it is default
            for key, val in self._prism_dict.items():
                # Convert key to lowercase and val to string
                key = key.lower()
                val = str(val)

                # Check if this key exists in default_dict
                if key in default_dict.keys():
                    # Check if the corresponding values are not the same
                    if(val != default_dict[key]):
                        # Add this parameter to the prism_par_repr list
                        par_repr.append("%r: %r" % (key, val))

            # Convert par_repr from list to dict and add it to str_repr
            str_repr.append("prism_par={%s}" % (", ".join(map(str, par_repr))))

        # Add the emul_type representation if it is not default
        emul_repr = "%s.%s" % (self._emulator.__class__.__module__,
                               self._emulator.__class__.__name__)
        if(emul_repr != 'prism.emulator._emulator.Emulator'):
            str_repr.append("emul_type=%s" % (emul_repr))

        # Return representation
        return("Pipeline(%s)" % (", ".join(str_repr)))

    # %% CLASS PROPERTIES
    # MPI properties
    @property
    def comm(self):
        """
        :obj:`~mpi4pyd.MPI.Intracomm`: The MPI intra-communicator that is used
        in this :obj:`~Pipeline` instance. By default, this is
        :obj:`MPI.COMM_WORLD`.

        """

        return(self._comm)

    @property
    def rank(self):
        """
        int: The rank of this MPI process in :attr:`~comm`. If no MPI is used,
        this is always 0.

        """

        return(self._rank)

    @property
    def size(self):
        """
        int: The number of MPI processes in :attr:`~comm`. If no MPI is used,
        this is always 1.

        """

        return(self._size)

    @property
    def is_controller(self):
        """
        bool: Whether or not this MPI process is a controller rank. If no MPI
        is used, this is always *True*.

        """

        return(bool(self._is_controller))

    @property
    def is_worker(self):
        """
        bool: Whether or not this MPI process is a worker rank. If no MPI is
        used, this is always *False*.

        """

        return(bool(self._is_worker))

    @property
    @contextmanager
    def worker_mode(self):
        """
        :obj:`~contextlib._GeneratorContextManager`: Special context manager
        within which all code is executed in worker mode. In worker mode, all
        worker ranks are continuously listening for calls from the controller
        rank made with :meth:`~_make_call`.

        Note that all code within the context manager is executed by all ranks,
        with the worker ranks executing it after the controller rank exited.
        It is therefore advised to use an if-statement inside to make sure only
        the controller rank executes the code.

        Using this context manager allows for easier use of *PRISM* in
        combination with serial/OpenMP codes (like MCMC methods).

        """

        # Make logger
        logger = getCLogger('WORKER_M')

        # Workers start listening for calls
        self._listen_for_calls()

        # Log that workers are now listening
        logger.info("Workers are now listening for calls.")

        # Execute code block within context manager
        yield

        # Make workers stop listening for calls
        self._make_call(None)

        # Log that workers are no longer listening for calls
        logger.info("Workers are no longer listening for calls.")

        # MPI Barrier
        self._comm.Barrier()

    # Pipeline details
    @property
    def do_logging(self):
        """
        bool: Whether or not to save all logging messages. If *False*, all
        logging messages of level :attr:`~logging.INFO` and below are ignored.
        It also enables/disables the progress bar for making projections.

        """

        return(bool(self._do_logging))

    @do_logging.setter
    def do_logging(self, flag):
        # Make logger
        logger = getRLogger('DO_LOGGING')

        # Check if provided value is a bool
        flag = check_vals(flag, 'do_logging', 'bool')

        # If flag and do_logging are the same, skip
        if flag is self._do_logging:
            pass
        # If logging is turned on, log this and turn off logging
        elif not flag:
            logging.root.manager.loggerDict['prism'].setLevel(logging.INFO+1)
            logger.warning("Logging messages of level %i (INFO) and below are "
                           "now ignored." % (logging.INFO))
        # If logging is turned off, turn it on and log this
        else:
            logging.root.manager.loggerDict['prism'].setLevel(logging.DEBUG)
            logger.warning("Logging messages are no longer ignored.")
        self._do_logging = flag

    @property
    def root_dir(self):
        """
        str: Absolute path to the root directory.

        """

        return(self._root_dir)

    @property
    def working_dir(self):
        """
        str: Absolute path to the working directory.

        """

        return(self._working_dir)

    @property
    def hdf5_file(self):
        """
        str: Absolute path to the loaded master HDF5-file.

        """

        return(self._hdf5_file)

    @property
    def prism_dict(self):
        """
        dict: Dictionary containing all *PRISM* parameters that were provided
        during :class:`~Pipeline` initialization.

        """

        return(self._prism_dict)

    @property
    def File(self):
        """
        :class:`~h5py.File`: Custom :class:`~h5py.File` class that has added
        logging and automatically uses :attr:`~hdf5_file` as the HDF5-file to
        open.

        """

        return(self._File)

    @property
    def modellink(self):
        """
        :obj:`~prism.modellink.ModelLink`: The
        :obj:`~prism.modellink.ModelLink` instance provided during
        :class:`~Pipeline` initialization.

        """

        return(self._modellink)

    @property
    def emulator(self):
        """
        :obj:`~prism.emulator.Emulator`: The :obj:`~prism.emulator.Emulator`
        instance created during :class:`~Pipeline` initialization.

        """

        return(self._emulator)

    @property
    def code_objects(self):
        """
        dict of code objects: Collection of pre-compiled built-in code snippets
        that are used in the :meth:`~_evaluate_sam_set` method.

        """

        return(self._code_objects)

    # Pipeline settings
    @property
    def criterion(self):
        """
        str, float or None: Value indicating which criterion to use in the
        :func:`~e13tools.sampling.lhd` function.

        """

        return(self._criterion)

    @criterion.setter
    def criterion(self, criterion):
        # Make logger
        logger = getRLogger('CHECK')

        # If criterion is None, save it as such
        if criterion is None:
            self._criterion = None

        # If criterion is a bool, raise error
        elif criterion is True or criterion is False:
            err_msg = ("Input argument 'criterion' does not accept values of "
                       "type 'bool'!")
            raise_error(err_msg, TypeError, logger)

        # If anything else is given, it must be a float or string
        else:
            # Try to use criterion to check validity
            try:
                lhd(3, 2, criterion=criterion)
            except Exception as error:
                err_msg = ("Input argument 'criterion' is invalid! (%s)"
                           % (error))
                raise_error(err_msg, InputError, logger)
            else:
                self._criterion = criterion

    @property
    def do_active_anal(self):
        """
        bool: Whether or not to do an active parameters analysis during the
        construction of the emulator systems.

        """

        return(bool(self._do_active_anal))

    @do_active_anal.setter
    def do_active_anal(self, do_active_anal):
        self._do_active_anal = check_vals(do_active_anal, 'do_active_anal',
                                          'bool')

    @property
    def freeze_active_par(self):
        """
        bool: Whether or not previously active parameters always stay active if
        possible.

        """

        return(bool(self._freeze_active_par))

    @freeze_active_par.setter
    def freeze_active_par(self, freeze_active_par):
        self._freeze_active_par = check_vals(freeze_active_par,
                                             'freeze_active_par', 'bool')

    @property
    def pot_active_par(self):
        """
        list of str: The potentially active parameters. Only parameters from
        this list can become active during the active parameters analysis.
        If :attr:`~do_active_anal` is *False*, all parameters in this list will
        be active.

        """

        return([self._modellink._par_name[i] for i in self._pot_active_par])

    @pot_active_par.setter
    def pot_active_par(self, pot_active_par):
        # Make logger
        logger = getRLogger('CHECK')

        # If pot_active_par is None, save all model parameters
        if pot_active_par is None:
            self._pot_active_par = np_array(range(self._modellink._n_par))

        # If pot_active_par is a bool, raise error
        elif pot_active_par is True or pot_active_par is False:
            err_msg = ("Input argument 'pot_active_par' does not accept values"
                       "of type 'bool'!")
            raise_error(err_msg, TypeError, logger)

        # If anything else is given, it must be a sequence of model parameters
        else:
            # Convert the given sequence to an array of indices
            self._pot_active_par = np_array(self._modellink._get_model_par_seq(
                pot_active_par, 'pot_active_par'))

    @property
    def n_sam_init(self):
        """
        int: Number of evaluation samples used to construct the first iteration
        of the emulator systems.

        """

        return(self._n_sam_init)

    @n_sam_init.setter
    def n_sam_init(self, n_sam_init):
        self._n_sam_init = check_vals(n_sam_init, 'n_sam_init', 'int', 'pos')

    @property
    def base_eval_sam(self):
        """
        int: Base number of emulator evaluations used to analyze the emulator
        systems. This number is scaled up by the number of model parameters and
        the current emulator iteration to generate the true number of emulator
        evaluations (:attr:`~n_eval_sam`).

        """

        return(self._base_eval_sam)

    @base_eval_sam.setter
    def base_eval_sam(self, base_eval_sam):
        self._base_eval_sam = check_vals(base_eval_sam, 'base_eval_sam', 'int',
                                         'pos')

    @property
    def impl_cut(self):
        """
        list of float: The non-wildcard univariate implausibility cut-off
        values for an emulator iteration. Setting it with the reduced
        implausibility cut-off list will change the values of :attr:`~cut_idx`
        and this property at the last emulator iteration.

        """

        return(self._impl_cut)

    @impl_cut.setter
    def impl_cut(self, impl_cut):
        # Only the controller can set this value
        if self._is_controller:
            # Make logger
            logger = getRLogger('CHECK')

            # Get last emul_i
            emul_i = self._emulator._emul_i

            # Raise error if emul_i is 0
            if not emul_i:
                err_msg = "Emulator is not built yet!"
                raise_error(err_msg, RequestError, logger)

            # Do some logging
            logger.info("Checking compatibility of provided implausibility "
                        "cut-off values.")

            # Check if analyze() is calling this setter
            caller_frame = get_outer_frame(self.analyze)

            # Check if the last iteration has already been analyzed
            try:
                if self._n_eval_sam[emul_i]:
                    # If so, check if this setter was called by analyze()
                    if caller_frame is None:
                        # If not, raise error
                        err_msg = ("Pipeline property 'impl_cut' can only be "
                                   "set if the current emulator iteration has "
                                   "not been analyzed yet!")
                        raise_error(err_msg, RequestError, logger)

            # If n_eval_sam at emul_i does not exist, no analyze has been done
            except IndexError:
                pass

            # Check if provided impl_cut contains solely non-negative floats
            impl_cut = check_vals(impl_cut, 'impl_cut', 'float', 'nneg')

            # Complete the impl_cut list
            # Loop over all values in impl_cut except the first
            for i in range(1, len(impl_cut)):
                # If value is zero, take on the value of the previous cut-off
                if not impl_cut[i]:
                    impl_cut[i] = impl_cut[i-1]

                # If the value is higher than the previous value, it is invalid
                elif(impl_cut[i-1] != 0 and impl_cut[i] > impl_cut[i-1]):
                    err_msg = ("Cut-off %i is higher than cut-off %i "
                               "(%f > %f)!"
                               % (i, i-1, impl_cut[i], impl_cut[i-1]))
                    raise_error(err_msg, ValueError, logger)

            # Get the index identifying where the first real impl_cut is
            red_impl_cut = impl_cut[:self._emulator._n_data_tot[emul_i]]
            for i, impl in enumerate(red_impl_cut):
                if(impl != 0):
                    cut_idx = i
                    break
            # If the loop ends normally, there were only wildcards
            else:
                err_msg = ("No non-wildcard implausibility cut-off was "
                           "provided!")
                raise_error(err_msg, ValueError, logger)

            # Obtain impl_cut
            impl_cut = np_array(impl_cut)[cut_idx:]

            # Set impl_par
            try:
                self._impl_cut[emul_i] = impl_cut
                self._cut_idx[emul_i] = cut_idx
            except IndexError:
                self._impl_cut.append(impl_cut)
                self._cut_idx.append(cut_idx)

    # Analysis results/properties
    @property
    def cut_idx(self):
        """
        int: The index of the first non-wildcard in a list of implausibility
        values. This is equivalent to the number of wildcards leading the
        cut-off values in :attr:`~impl_cut`.

        """

        return(self._cut_idx)

    @property
    def n_eval_sam(self):
        """
        int: The number of evaluation samples used to analyze an emulator
        iteration of the emulator systems. The number of plausible evaluation
        samples is stored in :attr:`~n_impl_sam`.
        It is zero if the specified iteration has not been analyzed yet.

        """

        return(self._n_eval_sam)

    @property
    def n_impl_sam(self):
        """
        int: Number of model evaluation samples that passed the implausibility
        checks during the analysis of an emulator iteration.
        It is zero if the specified iteration has not been analyzed yet or has
        no plausible samples.

        """

        return(self._n_impl_sam)

    @property
    def impl_sam(self):
        """
        :obj:`~numpy.ndarray`: The model evaluation samples that will be added
        to the next emulator iteration.

        """

        return(self._impl_sam)

    # %% GENERAL CLASS METHODS
    # Function that locks workers into listening for controller calls
    def _listen_for_calls(self):
        """
        All worker ranks in the :attr:`~comm` communicator start listening for
        calls from the corresponding controller rank and will attempt to
        execute the received message. Listening for calls continues until
        :attr:`~_worker_mode` is set to *False*.

        This method is automatically initialized and finalized when using the
        :attr:`~worker_mode` context manager.

        """

        # Set worker_mode to 1
        self._worker_mode = 1

        # All workers start listening for calls
        if self._is_worker:
            while self._worker_mode:
                exec_fn, args, kwargs = self._comm.bcast([], 0)
                if exec_fn is None:
                    self._worker_mode = 0
                elif isinstance(exec_fn, str):
                    attrs = exec_fn.split('.')
                    obj = self
                    for attr in attrs:
                        obj = getattr(obj, attr)
                    obj(*args, **kwargs)
                else:
                    exec_fn(*args, **kwargs)

    # Function that sends a code string to all workers and executes it
    def _make_call(self, exec_fn, *args, **kwargs):
        """
        Send the provided `exec_fn` to all worker ranks, if they are
        listening for calls, and tell them to execute it using the provided
        `args` and `kwargs`. All ranks that call this function will execute
        `exec_fn` as well.

        If used within the :attr:`~worker_mode` context manager, this function
        should only be called by the controller. If not, it should be called by
        all ranks that must execute `exec_fn`.

        Parameters
        ----------
        exec_fn : str, callable or None
            If string, a callable attribute of this :obj:`~Pipeline` instance
            or a callable object that the workers should execute if not.
            If *None*, the workers stop listening for calls instead (disables
            worker mode).
        args : tuple
            Positional arguments that need to be provided to `exec_fn`.
        kwargs : dict
            Keyword arguments that need to be provided to `exec_fn`.

        """

        # Send received exec_code to all workers if they are listening
        if self._worker_mode and self._is_controller:
            self._comm.bcast([exec_fn, args, kwargs], 0)

        # Execute exec_fn as well
        if exec_fn is None:
            self._worker_mode = 0
        elif isinstance(exec_fn, str):
            attrs = exec_fn.split('.')
            obj = self
            for attr in attrs:
                obj = getattr(obj, attr)
            return(obj(*args, **kwargs))
        else:
            return(exec_fn(*args, **kwargs))

    # This function evaluates the model for a given set of evaluation samples
    # TODO: If not MPI_call, all ranks evaluate part of sam_set simultaneously?
    # Requires check/flag that model can be evaluated in multiple instances
    # TODO: If not MPI_call, should OMP_NUM_THREADS be temporarily unset?
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _evaluate_model(self, emul_i, sam_set, data_idx):
        """
        Evaluates the model for provided evaluation sample set `sam_set` at
        given data points `data_idx`.

        This method automatically distributes the samples according to the
        various flags set in the :class:`~prism.modellink.ModelLink` subclass.

        Parameters
        ----------
        %(emul_i)s
        sam_set : 1D or 2D array_like
            Parameter/sample set to evaluate in the model.
        data_idx : list of tuples
            The list of data identifiers for which the model is requested to
            return the corresponding data values.

        Returns
        -------
        mod_set : 2D :obj:`~numpy.ndarray` object of shape ``(n_sam, n_data)``
            Array containing the data values corresponding to the requested
            data points.

        """

        # Log that evaluation of model samples is started
        logger = getCLogger('MODEL')
        logger.info("Evaluating model samples.")

        # Make sure that sam_set is at least 2D and a NumPy array
        sam_set = np_array(sam_set, ndmin=2)

        # Check who needs to call the model
        if self._is_controller or self._modellink._MPI_call:
            # Request all evaluation samples at once
            if self._modellink._multi_call:
                mod_set = self._multi_call_model(emul_i, sam_set, data_idx)

            # Request evaluation samples one-by-one
            else:
                # Initialize mod_set
                mod_set = np.zeros([sam_set.shape[0], self._modellink._n_data])

                # Loop over all requested evaluation samples
                for i, par_set in enumerate(sam_set):
                    mod_set[i] = self._call_model(emul_i, par_set, data_idx)

        # If workers did not call model, give them a dummy mod_set
        else:
            mod_set = []

        # Log that evaluation is completed and return mod_set
        logger.info("Finished evaluating model samples.")
        return(mod_set)

    # Function obtaining the model output for a given set of parameter values
    @docstring_append(call_model_doc_s)
    def _call_model(self, emul_i, par_set, data_idx):
        # Make sure par_set is at least 1D and a NumPy array
        sam = np_array(par_set, ndmin=1)

        # Log that model is being called
        logger = getCLogger('CALL_MODEL')
        logger.info("Calling model at parameters %s." % (sam))

        # Obtain model output
        mod_out = self._modellink.call_model(
            emul_i=emul_i,
            par_set=sdict(zip(self._modellink._par_name, sam)),
            data_idx=data_idx)

        # If mod_out is a dict, convert it to a NumPy array
        if isinstance(mod_out, dict):
            mod_out = np_array([mod_out[idx] for idx in data_idx])

        # Log that calling model has been finished
        logger.info("Model returned %s." % (mod_out))

        # Return it
        return(np_array(mod_out))

    # Function containing the model output for a given set of parameter samples
    @docstring_append(call_model_doc_m)
    def _multi_call_model(self, emul_i, sam_set, data_idx):
        # Make sure sam_set is at least 2D and a NumPy array
        sam_set = np_array(sam_set, ndmin=2)

        # Log that model is being multi-called
        logger = getCLogger('CALL_MODEL')
        logger.info("Multi-calling model for sample set of size %i."
                    % (sam_set.shape[0]))

        # Obtain set of model outputs
        mod_set = self._modellink.call_model(
            emul_i=emul_i,
            par_set=sdict(zip(self._modellink._par_name, sam_set.T)),
            data_idx=data_idx)

        # If mod_set is a dict, convert it to a NumPy array
        if isinstance(mod_set, dict):
            mod_set = np_array([mod_set[idx] for idx in data_idx]).T

        # Log that multi-calling model has been finished
        logger.info("Finished model multi-call.")

        # Return it
        return(np_array(mod_set))

    # This function reads in the parameters from the provided parameters
    def _read_parameters(self, prism_par):
        """
        Reads in all parameters in the provided `prism_par` and saves them as
        a dict in the current :obj:`~Pipeline` instance.

        """

        # Make a logger
        logger = getCLogger('INIT')
        logger.info("Reading in PRISM parameters.")

        # If no PRISM parameters were provided
        if prism_par is None:
            prism_dict = sdict()

        # If a PRISM parameter file was provided
        elif isinstance(prism_par, str):
            # Check if prism_par was given as an absolute path
            if path.exists(prism_par):
                prism_par = path.abspath(prism_par)
            # If not, check if it was relative to root_dir
            elif path.exists(path.join(self._root_dir, prism_par)):
                prism_par = path.join(self._root_dir, prism_par)
            # If not either, it is invalid
            else:
                err_msg = ("Input argument 'prism_par' is a non-existing path"
                           " (%r)!" % (prism_par))
                raise_error(err_msg, OSError, logger)

            # Read in the contents of the given prism_par file
            prism_par = np.genfromtxt(prism_par, dtype=(str), delimiter=':',
                                      autostrip=True)

            # Make sure that prism_par is 2D
            prism_par = np_array(prism_par, ndmin=2)

            # Make prism_dict
            prism_dict = sdict(prism_par)

        # If a PRISM parameters dict was provided
        elif isinstance(prism_par, dict):
            # Save parameters dict
            prism_dict = sdict(prism_par)

        # If anything else is given
        else:
            # Check if it can be converted to a dict
            try:
                prism_dict = sdict(prism_par)
            except Exception:
                err_msg = ("Input argument 'prism_par' cannot be converted to "
                           "type 'dict'!")
                raise_error(err_msg, TypeError, logger)

        # Save prism_dict
        self._prism_dict = prism_dict

        # Log again
        logger.info("Finished reading in PRISM parameters.")

    # This function returns default pipeline parameters
    @docstring_append(def_par_doc.format("pipeline"))
    def _get_default_parameters(self):
        # Create parameter dict with default parameters
        par_dict = {'n_sam_init': '500',
                    'base_eval_sam': '800',
                    'impl_cut': '[0, 4.0, 3.8, 3.5]',
                    'criterion': "None",
                    'do_active_anal': 'True',
                    'freeze_active_par': 'True',
                    'pot_active_par': 'None'}

        # Return it
        return(sdict(par_dict))

    # Set the parameters that were read in from the provided parameter file
    # TODO: May want to use configparser.Configparser for this
    @docstring_append(set_par_doc.format("Pipeline"))
    def _set_parameters(self):
        # Log that the pipeline parameters are being set
        logger = getCLogger('INIT')
        logger.info("Setting pipeline parameters.")

        # Obtaining default pipeline parameter dict
        par_dict = self._get_default_parameters()

        # Add the read-in prism dict to it
        par_dict.update(self._prism_dict)

        # More logging
        logger.info("Checking compatibility of provided pipeline parameters.")

        # GENERAL
        # Set number of starting samples
        self.n_sam_init = convert_str_seq(par_dict['n_sam_init'])[0]

        # Set base number of emulator evaluation samples
        self.base_eval_sam = convert_str_seq(par_dict['base_eval_sam'])[0]

        # Convert criterion to a string
        criterion = str(par_dict['criterion'])

        # If criterion is None, True or False, try to save it as such
        if criterion.lower() in ('none', 'true', 'false'):
            self.criterion = literal_eval(criterion.capitalize())

        # If anything else is given, it must be an int, float or string
        else:
            self.criterion = convert_str_seq(criterion)[0]

        # Set the bool determining whether to do an active parameters analysis
        self.do_active_anal = par_dict['do_active_anal']

        # Set the bool determining whether active parameters stay active
        self.freeze_active_par = par_dict['freeze_active_par']

        # Convert pot_active_par to a string
        pot_active_par = str(par_dict['pot_active_par'])

        # If pot_active_par is None, True or False, try to save it as such
        if pot_active_par.lower() in ('none', 'true', 'false'):
            self.pot_active_par = literal_eval(pot_active_par.capitalize())

        # If anything else is given, save it
        else:
            self.pot_active_par = pot_active_par

        # Log that setting has been finished
        logger.info("Finished setting pipeline parameters.")

    # This function controls how n_eval_samples is calculated
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_n_eval_sam(self, emul_i):
        """
        This function calculates the total number of emulator evaluation
        samples at a given emulator iteration `emul_i` from
        :attr:`~base_eval_sam`.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        n_eval_sam : int
            Total number of emulator evaluation samples.

        """

        # Calculate n_eval_sam and return it
        return(emul_i*self._base_eval_sam*self._modellink._n_par)

    # Obtains the paths for the root directory, working directory, pipeline
    # hdf5-file and prism parameters file
    # TODO: Start supporting and using pathlib.Path?
    @docstring_substitute(paths=paths_doc_s)
    def _get_paths(self, root_dir, working_dir, prefix):
        """
        Obtains the path for the root directory, working directory and
        parameters file for *PRISM*.

        Parameters
        ----------
        %(paths)s

        Generates
        ---------
        The absolute paths to the root directory, working directory, emulator
        master HDF5-file and *PRISM* parameters file.

        """

        # Set logging system
        logger = getCLogger('INIT')
        logger.info("Obtaining related directory and file paths.")

        # Controller obtaining the paths
        if self._is_controller:
            # Obtain root directory path
            # If one did not specify a root directory, set it to default
            if root_dir is None:
                self._root_dir = path.abspath('.')
                logger.info("No root directory specified, set to %r."
                            % (self._root_dir))

            # If one specified a root directory, use it
            elif isinstance(root_dir, str):
                self._root_dir = path.abspath(root_dir)
                logger.info("Root directory set to %r." % (self._root_dir))

                # Check if this directory already exists and create it if not
                try:
                    os.mkdir(self._root_dir)
                except OSError:
                    pass
                else:
                    logger.info("Root directory did not exist, created it.")

            # If anything else is given, it is invalid
            else:
                err_msg = "Input argument 'root_dir' is invalid!"
                raise_error(err_msg, InputError, logger)

            # Check if a valid working directory prefix string is given
            if prefix is None:
                prefix_scan = ''
                prefix_new = 'prism_'
            elif isinstance(prefix, str):
                prefix_scan = prefix
                prefix_new = prefix
            else:
                err_msg = "Input argument 'prefix' is not of type 'str'!"
                raise_error(err_msg, TypeError, logger)

            # Obtain working directory path
            # If one did not specify a working directory, obtain it
            if working_dir in (None, False):
                logger.info("No working directory specified, trying to load "
                            "last one created.")
                dirnames = next(os.walk(self._root_dir))[1]
                emul_dirs = []

                # Check which directories in the root_dir satisfy the default
                # naming scheme of the emulator directories
                for dirname in dirnames:
                    # If the prefix is the same as the scan prefix
                    if dirname.startswith(prefix_scan):
                        # Obtain full path to this directory
                        dir_path = path.join(self._root_dir, dirname)

                        # Check if this directory contains a 'prism_log.log'
                        if 'prism_log.log' in os.listdir(dir_path):
                            # Obtain creation time and append to emul_dirs
                            ctime = path.getctime(dir_path)
                            emul_dirs.append([dir_path, ctime])

                # Sort list of emul_dirs on creation time
                emul_dirs.sort(key=lambda x: x[1], reverse=True)

                # If no working directory exists, create a new one
                if not len(emul_dirs):
                    working_dir = ''.join([prefix_new, '0'])
                    self._working_dir = path.join(self._root_dir, working_dir)
                    os.mkdir(self._working_dir)
                    logger.info("No working directories found, created %r."
                                % (working_dir))

                # If working directories exist, load last one created
                else:
                    self._working_dir = emul_dirs[0][0]
                    logger.info("Working directories found, set to %r."
                                % (path.basename(self._working_dir)))

            # If one requested a new working directory
            elif working_dir is True:
                # Obtain list of working directories that satisfy naming scheme
                dirnames = next(os.walk(self._root_dir))[1]
                n_dirs = 0

                # Check if there are any directories with the same prefix
                for dirname in dirnames:
                    if dirname.startswith(prefix_scan):
                        # Obtain full path to this directory
                        dir_path = path.join(self._root_dir, dirname)

                        # Check if this directory contains a 'prism_log.log'
                        if 'prism_log.log' in os.listdir(dir_path):
                            n_dirs += 1

                # Check if working directories already exist with the same
                # prefix and append a number to the name if this is the case
                while True:
                    working_dir = path.join(self._root_dir,
                                            ''.join([prefix_new, str(n_dirs)]))
                    try:
                        os.mkdir(working_dir)
                    except OSError:
                        n_dirs += 1
                    else:
                        break

                # Save path to new working directory
                self._working_dir = working_dir
                logger.info("New working directory requested, created %r."
                            % (path.basename(working_dir)))

            # If one specified a working directory, use it
            elif isinstance(working_dir, str):
                self._working_dir = path.join(self._root_dir, working_dir)
                logger.info("Working directory set to %r." % (working_dir))

                # Check if this directory already exists and create it if not
                try:
                    os.mkdir(self._working_dir)
                except OSError:
                    pass
                else:
                    logger.info("Working directory did not exist, created it.")

            # If one provided an integer, raise warning about it
            elif isinstance(working_dir, int):
                err_msg = ("Input argument 'working_dir' cannot be of type "
                           "int!")
                raise_error(err_msg, TypeError, logger)

            # If anything else is given, it is invalid
            else:
                err_msg = "Input argument 'working_dir' is invalid!"
                raise_error(err_msg, InputError, logger)

            # Obtain hdf5-file path
            self._hdf5_file = path.join(self._working_dir, 'prism.hdf5')

        # Workers get dummy paths
        else:
            self._root_dir = None
            self._working_dir = None
            self._hdf5_file = None

        # Broadcast paths to workers
        self._root_dir = self._comm.bcast(self._root_dir, 0)
        self._working_dir = self._comm.bcast(self._working_dir, 0)
        self._hdf5_file = self._comm.bcast(self._hdf5_file, 0)

        # Generate custom File class using the path to the master HDF5-file
        self._File = get_PRISM_File(self._hdf5_file)

    # This function generates mock data and loads it into ModelLink
    # TODO: Find a way to use mock data without changing ModelLink properties
    def _get_mock_data(self, mock_par):
        """
        Generates mock data and loads it into the
        :obj:`~prism.modellink.ModelLink` object that was provided
        during class initialization.
        This function overwrites the
        :class:`~prism.modellink.ModelLink` properties holding the
        parameter estimates, data values and data errors.

        Parameters
        ----------
        mock_par : 1D array_like or None
            If 1D array_like, use the provided parameter estimates to create
            the mock data. If *None*, a random parameter set will be generated
            as parameter estimates.

        Generates
        ---------
        Overwrites the corresponding
        :class:`~prism.modellink.ModelLink` class properties with the
        generated values.

        """

        # Start logger
        logger = getCLogger('MOCK_DATA')

        # Log new mock_data being created
        logger.info("Generating mock data for new emulator.")

        # Controller only
        if self._is_controller:
            # If mock parameter estimates were provided, use them
            if mock_par is not None:
                self._modellink._par_est = mock_par.tolist()

            # Else, generate mock parameter estimates
            else:
                self._modellink._par_est = self._modellink._to_par_space(
                    random(self._modellink._n_par)).tolist()

        # Controller broadcasting new parameter estimates to workers
        self._modellink._par_est = self._comm.bcast(self._modellink._par_est,
                                                    0)

        # Obtain non-default model data values
        data_val = self._evaluate_model(0, self._modellink._par_est,
                                        self._modellink._data_idx)

        # Controller only
        if self._is_controller:
            # Set non-default model data values
            self._emulator._data_val[0] = data_val[0].tolist()

            # Use model discrepancy variance as model data errors
            md_var = self._get_md_var(0, self._modellink._par_est)
            err = np.sqrt(md_var).tolist()
            self._modellink._data_err = err

            # Add model data errors as noise to model data values
            noise = normal(size=self._modellink._n_data)
            for i in range(self._modellink._n_data):
                # If value space is linear
                if(self._modellink._data_spc[i] == 'lin'):
                    if(noise[i] > 0):
                        noise[i] *= err[i][0]
                    else:
                        noise[i] *= err[i][1]

                # If value space is log10
                elif(self._modellink._data_spc[i] == 'log10'):
                    if(noise[i] > 0):
                        noise[i] = np.log10((pow(10, err[i][0])-1)*noise[i]+1)
                    else:
                        noise[i] =\
                            np.log10((pow(10, -1*err[i][1])-1)*-1*noise[i]+1)

                # If value space is ln
                elif(self._modellink._data_spc[i] == 'ln'):
                    if(noise[i] > 0):
                        noise[i] = np.log((pow(np.e, err[i][0])-1)*noise[i]+1)
                    else:
                        noise[i] =\
                            np.log((pow(np.e, -1*err[i][1])-1)*-1*noise[i]+1)

                # If value space is anything else
                else:
                    raise NotImplementedError

            # Add noise to the data values
            self._modellink._data_val =\
                (self._emulator._data_val[0]+noise).tolist()

        # Logger
        logger.info("Generated mock data.")

        # Broadcast updated modellink properties to workers
        self._modellink._data_val = self._comm.bcast(self._modellink._data_val,
                                                     0)
        self._modellink._data_err = self._comm.bcast(self._modellink._data_err,
                                                     0)

    # This function loads pipeline data
    def _load_data(self):
        """
        Loads in all the important pipeline data into memory for the controller
        rank.
        If it is detected that the last emulator iteration has not been
        analyzed yet, the implausibility analysis parameters are read in from
        the *PRISM* parameters file and temporarily stored in memory.

        Generates
        ---------
        All relevant pipeline data up to the last emulator iteration is loaded
        into memory.

        """

        # Set the logger
        logger = getCLogger('LOAD_DATA')

        # Initialize all data sets with empty lists
        logger.info("Initializing pipeline data sets.")
        self._n_impl_sam = [[]]
        self._impl_cut = [[]]
        self._cut_idx = [[]]
        self._n_eval_sam = [[]]
        self._impl_sam = []

        # If an emulator currently exists, load in all data
        if self._emulator._emul_i:
            # Open hdf5-file
            with self._File('r', None) as file:
                # Read in the data up to the last emulator iteration
                for i in range(1, self._emulator._emul_i+1):
                    # Get this emulator
                    emul = file['%i' % (i)]

                    # Check if analysis has been carried out (only if i=emul_i)
                    try:
                        self._impl_cut.append(emul.attrs['impl_cut'])

                    # If not, use prism_dict to set the impl_par
                    except KeyError:
                        self._set_impl_par(None)

                    # If so, load the cut_idx as well
                    else:
                        self._cut_idx.append(emul.attrs['cut_idx'])

                    # Load the remaining data (which is always available)
                    finally:
                        self._n_impl_sam.append(emul.attrs['n_impl_sam'])
                        self._n_eval_sam.append(emul.attrs['n_eval_sam'])

                # Read in the samples that survived the implausibility check
                self._impl_sam = emul['impl_sam'][()]
                self._impl_sam.dtype = float

        # Log that loading has been finished
        logger.info("Finished loading pipeline data sets.")

    # This function saves pipeline data to hdf5
    @docstring_substitute(save_data=save_data_doc_p)
    def _save_data(self, data_dict):
        """
        Saves a given data dict ``{keyword: data}`` at the last emulator
        iteration to the HDF5-file and as an data attribute to the current
        :obj:`~Pipeline` instance.

        %(save_data)s

        """

        # Do some logging
        logger = getRLogger('SAVE_DATA')

        # Obtain last emul_i
        emul_i = self._emulator._emul_i

        # Open hdf5-file
        with self._File('r+', None) as file:
            # Obtain the dataset this data needs to be saved to
            data_set = file['%i' % (emul_i)]

            # Loop over entire provided data dict
            for keyword, data in data_dict.items():
                # Log what data is being saved
                logger.info("Saving %r data at iteration %i to HDF5."
                            % (keyword, emul_i))

                # Check what data keyword has been provided
                # IMPL_PAR
                if(keyword == 'impl_par'):
                    # Save impl_par data
                    self._impl_cut[emul_i] = data['impl_cut']
                    self._cut_idx[emul_i] = data['cut_idx']
                    data_set.attrs['impl_cut'] = data['impl_cut']
                    data_set.attrs['cut_idx'] = data['cut_idx']

                # IMPL_SAM
                elif(keyword == 'impl_sam'):
                    # Check if any plausible regions have been found at all
                    n_impl_sam = np.shape(data)[0]

                    # Convert data to a compound data set
                    dtype = [(n, float) for n in self._modellink._par_name]
                    data_c = data.copy()
                    data_c.dtype = dtype

                    # Check if impl_sam data has been saved before
                    try:
                        self._n_impl_sam[emul_i] = n_impl_sam
                    except IndexError:
                        data_set.create_dataset('impl_sam', data=data_c)
                        self._n_impl_sam.append(n_impl_sam)
                    else:
                        del data_set['impl_sam']
                        data_set.create_dataset('impl_sam', data=data_c)
                    finally:
                        self._impl_sam = data
                        data_set.attrs['n_impl_sam'] = n_impl_sam

                # N_EVAL_SAM
                elif(keyword == 'n_eval_sam'):
                    # Check if n_eval_sam has been saved before
                    try:
                        self._n_eval_sam[emul_i] = data
                    except IndexError:
                        self._n_eval_sam.append(data)
                    finally:
                        data_set.attrs['n_eval_sam'] = data

                # INVALID KEYWORD
                else:
                    err_msg = "Invalid keyword argument provided!"
                    raise_error(err_msg, ValueError, logger)

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

        Dict variables
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
        logger = getCLogger('STATISTICS')
        logger.info("Saving statistics to HDF5.")

        # Open hdf5-file
        with self._File('r+', None) as file:
            # Loop over all statistics in stat_dict and save them
            for keyword, (value, unit) in stat_dict.items():
                file['%i/statistics' % (emul_i)].attrs[keyword] =\
                    [str(value).encode('ascii', 'ignore'),
                     unit.encode('ascii', 'ignore')]

    # This function evaluates and distributes the model evaluations samples
    # This is function 'k'
    @docstring_substitute(emul_i=std_emul_i_doc, ext_sam=ext_sam_set_doc,
                          ext_mod=ext_mod_set_doc)
    def _get_iteration_data(self, emul_i, sam_set, ext_sam_set, ext_mod_set):
        """
        Obtains the model realization data for given emulator iteration
        `emul_i` by evaluating the provided `sam_set` in the model and
        distributing model outputs to the correct emulator systems.

        Parameters
        ----------
        %(emul_i)s
        sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the model evaluation samples.
        %(ext_sam)s
        %(ext_mod)s

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
        logger = getCLogger('MODEL_REAL')
        logger.info("Obtaining model realization data for emulator iteration "
                    "%i." % (emul_i))

        # Save the current time
        start_time = time()

        # Obtain number of samples
        n_sam = np.shape(sam_set)[0]

        # Flatten the corresponding data_idx_to_core on the controller
        if self._is_controller:
            data_idx_flat = []
            data_idx_len = []
            for data_idx_rank in self._emulator._data_idx_to_core[emul_i]:
                data_idx_rank = delist(data_idx_rank)
                data_idx_len.append(len(data_idx_rank))
                data_idx_flat.extend(data_idx_rank)

        # Use dummy data_idx_flat on workers
        else:
            data_idx_flat = None

        # For sake of consistency, broadcast data_idx_flat to workers
        data_idx_flat = self._comm.bcast(data_idx_flat, 0)

        # If there are any samples in sam_set, evaluate them in the model
        if n_sam:
            mod_set = self._evaluate_model(emul_i, sam_set, data_idx_flat)

            # Transpose obtained mod_set on controller
            if self._is_controller:
                mod_set = mod_set.T

        # Controller processing the received data values
        if self._is_controller:
            # Get end time
            end_time = time()-start_time

            # Check if ext_real_set and/or sam_set were provided
            if ext_sam_set.shape[0] and n_sam:
                sam_set = np.concatenate([sam_set, ext_sam_set], axis=0)
                mod_set = np.concatenate([mod_set, ext_mod_set], axis=1)
                use_ext_real_set = 1
            elif ext_sam_set.shape[0]:
                sam_set = ext_sam_set
                mod_set = ext_mod_set
                use_ext_real_set = 1
            else:
                use_ext_real_set = 0

            # Sent the specific mod_set parts to the corresponding workers
            logger.info("Distributing model realization data to corresponding "
                        "emulator systems.")
            s_idx = 0
            for rank, n_data in enumerate(data_idx_len):
                # Controller data must be saved last
                if not rank:
                    pass

                # Send the remaining parts to the workers
                else:
                    self._comm.send([sam_set, mod_set[s_idx:s_idx+n_data]],
                                    dest=rank, tag=777+rank)

                # Save which data parts have already been sent
                s_idx += n_data

            # MPI Barrier to make sure that workers have saved their data
            self._comm.Barrier()

            # Save controller data
            self._emulator._save_data(emul_i, None, {
                'mod_real_set': {
                    'sam_set': sam_set,
                    'mod_set': mod_set[0:data_idx_len[0]],
                    'use_ext_real_set': use_ext_real_set}})

        # Workers waiting for controller to send them their data values
        else:
            sam_set, mod_set = self._comm.recv(source=0, tag=777+self._rank)

            # Save all the data to the specific hdf5-files
            for i, lemul_s in enumerate(self._emulator._active_emul_s[emul_i]):
                self._emulator._save_data(emul_i, lemul_s, {
                    'mod_real_set': {
                        'mod_set': mod_set[i]}})

            # Save sam_set data to memory
            self._emulator._sam_set[emul_i] = sam_set
            self._emulator._n_sam[emul_i] = np.shape(sam_set)[0]

            # MPI Barrier to let controller know data was saved
            self._comm.Barrier()

        # Controller finishing up
        if self._is_controller:
            # Log that this is finished
            eval_rate = end_time/n_sam if n_sam else 0
            msg = ("Finished obtaining and distributing model realization data"
                   " in %#.3g seconds, averaging %#.3g seconds per model "
                   "evaluation." % (end_time, eval_rate))
            self._save_statistics(emul_i, {
                'tot_model_eval_time': ['%#.3g' % (end_time), 's'],
                'avg_model_eval_time': ['%#.3g' % (eval_rate), 's'],
                'MPI_comm_size_model': ['%i' % (self._size), '']})
            logger.info(msg)
            print(msg)

        # MPI Barrier
        self._comm.Barrier()

    # This function generates a large Latin Hypercube sample set to analyze
    # the emulator at
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_eval_sam_set(self, emul_i):
        """
        Generates an emulator evaluation sample set to be used for analyzing an
        emulator iteration. Currently uses the :func:`~e13tools.sampling.lhd`
        function.

        Parameters
        ----------
        %(emul_i)s

        Returns
        -------
        eval_sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing the evaluation samples.

        """

        # Log about this
        logger = getCLogger('EVAL_SAMS')

        # Obtain number of samples
        n_eval_sam = self._get_n_eval_sam(emul_i)

        # Create array containing all samples for analyzing the emulator
        logger.info("Creating emulator evaluation sample set with size %i."
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
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _do_impl_check(self, emul_i, uni_impl_val):
        """
        Performs an implausibility cut-off check on the provided implausibility
        values `uni_impl_val` at emulator iteration `emul_i`.

        Parameters
        ----------
        %(emul_i)s
        uni_impl_val : 1D array_like
            Array containing all univariate implausibility values corresponding
            to a certain parameter set for all data points.

        Returns
        -------
        result : bool
            1 if check was successful, 0 if it was not.
        impl_cut_val : float
            Implausibility value at the first real implausibility cut-off.

        """

        # Sort impl_val to compare with the impl_cut list
        sorted_impl_val = np.flip(np.sort(
            uni_impl_val, axis=-1), axis=-1)[self._cut_idx[emul_i]:]

        # Save the implausibility value at the first real cut-off
        impl_cut_val = sorted_impl_val[0]

        # Scan over all data points in this sample
        for impl_val, cut_val in zip(sorted_impl_val, self._impl_cut[emul_i]):
            # If impl_val is not below impl_cut, break
            if(impl_val > cut_val):
                return(0, impl_cut_val)
        else:
            # If for-loop ended in a normal way, the check was successful
            return(1, impl_cut_val)

    # This function calculates the univariate implausibility values
    # This is function 'I²(x)'
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_uni_impl(self, emul_i, par_set, adj_exp_val, adj_var_val):
        """
        Calculates the univariate implausibility values at a given emulator
        iteration `emul_i` for specified expectation and variance values
        `adj_exp_val` and `adj_var_val`, corresponding to given `par_set`.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the univariate
            implausibility values for. Only used to pass to the
            :meth:`~prism.modellink.ModelLink.get_md_var` method.
        adj_exp_val, adj_var_val : 1D array_like
            The adjusted expectation and variance values to calculate the
            univariate implausibility for.

        Returns
        -------
        uni_impl_val : 1D :obj:`~numpy.ndarray` object
            Univariate implausibility value for all requested emulator systems.

        """

        # Obtain model discrepancy variance
        md_var = self._get_md_var(emul_i, par_set)

        # Determine the active emulator systems for this iteration
        active_emul_s = self._emulator._active_emul_s[emul_i]

        # Initialize empty univariate implausibility
        uni_impl_val_sq = np.zeros(len(active_emul_s))

        # Calculate the univariate implausibility values
        for i, emul_s in enumerate(active_emul_s):
            # Use the upper errors by default
            err_idx = 0

            # If adj_exp_val < data_val, use the lower error instead
            if(adj_exp_val[i] < self._emulator._data_val[emul_i][emul_s]):
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
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _get_md_var(self, emul_i, par_set):
        """
        Retrieves the model discrepancy variances, which includes all variances
        that are created by the model provided by the
        :obj:`~prism.modellink.ModelLink` instance. This method tries
        to call the :meth:`~prism.modellink.ModelLink.get_md_var`
        method, and assumes a default model discrepancy variance of ``1/6th``
        the data value if it cannot be called. If the data value space is not
        linear, then this default value is calculated such to reflect that.

        Parameters
        ----------
        %(emul_i)s
        par_set : 1D :obj:`~numpy.ndarray` object
            Model parameter value set to calculate the model discrepancy
            variances for.

        Returns
        -------
        var_md : 2D :obj:`~numpy.ndarray` object
            Variance of the model discrepancy.

        """

        # Obtain md variances
        # Try to use the user-defined md variances
        try:
            md_var = self._modellink.get_md_var(
                emul_i=emul_i,
                par_set=sdict(zip(self._modellink._par_name, par_set)),
                data_idx=delist(self._emulator._data_idx[emul_i]))

        # If it was not user-defined, use a default value
        except NotImplementedError:
            # Use factor 2 difference on 2 sigma as acceptable
            # Imagine that 2 sigma range is given if lower and upper are factor
            # 2 apart. This gives that sigma must be 1/6th of the data value

            # Create empty md_var list
            md_var = []

            # Loop over all data points and check their values spaces
            for data_val, data_spc in zip(
                    delist(self._emulator._data_val[emul_i]),
                    delist(self._emulator._data_spc[emul_i])):
                # If value space is linear, take 1/6th of the data value
                if(data_spc == 'lin'):
                    md_var.append([pow(data_val/6, 2)]*2)

                # If value space is log10, take log10(7/6) and log10(5/6)
                elif(data_spc == 'log10'):
                    md_var.append([0.0044818726418455815,
                                   0.006269669725654501])

                # If value space is ln, take ln(7/6) and ln(5/6)
                elif(data_spc == 'ln'):
                    md_var.append([0.023762432091205918, 0.03324115007177121])

                # If value space is anything else
                else:
                    raise NotImplementedError

            # Make sure that md_var is a NumPy array
            md_var = np_array(md_var, ndmin=2)

        # If it was user-defined, check if the values are compatible
        else:
            # If md_var is a dict, convert it to a NumPy array
            if isinstance(md_var, dict):
                data_idx = self._emulator._data_idx[emul_i]
                md_var = np_array([md_var[idx] for idx in data_idx])

            # Make sure that md_var is a NumPy array
            md_var = np_array(md_var)

            # If single values were given, duplicate them
            if(md_var.ndim == 1):
                md_var = np_array([md_var]*2).T

        # Return it
        return(md_var)

    # This function sets the impl_cut list from read-in PRISM parameters file
    # TODO: Make impl_cut dynamic
    @docstring_substitute(impl_cut=impl_cut_doc)
    def _set_impl_par(self, impl_cut):
        """
        Sets the :attr:`~impl_cut` and :attr:`~cut_idx` properties for
        implausibility evaluations using the read-in *PRISM* parameters file
        and the provided `impl_cut`.

        Parameters
        ----------
        impl_cut : list of float or None
            Incomplete, shortened impl_cut-offs list to be used during the
            analysis of this emulator iteration. If *None*, use the read-in
            *PRISM* parameters file instead.

        Generates
        ---------
        %(impl_cut)s

        """

        # Do some logging
        logger = getCLogger('INIT')
        logger.info("Setting implausibility cut-off values.")

        # If impl_cut is None, set the impl_par the standard way
        if impl_cut is None:
            # Obtaining default pipeline parameter dict
            par_dict = self._get_default_parameters()

            # Add the read-in prism dict to it
            par_dict.update(self._prism_dict)

            # Remove auxiliary characters from impl_cut
            impl_cut = convert_str_seq(par_dict['impl_cut'])

        # Set the impl_par
        self.impl_cut = impl_cut

        # Finish logging
        logger.info("Finished setting implausibility cut-off values.")

    # This function processes an externally provided real_set
    # TODO: Additionally allow for an old PRISM master file to be provided
    @docstring_substitute(ext_set=ext_real_set_doc_s, ext_sam=ext_sam_set_doc,
                          ext_mod=ext_mod_set_doc)
    def _get_ext_real_set(self, ext_real_set):
        """
        Processes an externally provided model realization set `ext_real_set`,
        containing the used sample set and the corresponding data value set.

        Parameters
        ----------
        %(ext_set)s

        Returns
        -------
        %(ext_sam)s
        %(ext_mod)s


        """

        # If no ext_real_set is provided, return empty arrays without logging
        if ext_real_set is None:
            return(np_array([]), np_array([]))

        # Do some logging
        logger = getCLogger('INIT')
        logger.info("Processing externally provided model realization set.")

        # If a list is given
        if isinstance(ext_real_set, list):
            # Check if ext_real_set contains 2 elements
            if(len(ext_real_set) != 2):
                err_msg = "Input argument 'ext_real_set' is not of length 2!"
                raise_error(err_msg, ShapeError, logger)

            # Try to extract ext_sam_set and ext_mod_set
            try:
                ext_sam_set, ext_mod_set = ext_real_set
            except Exception as error:
                err_msg = ("Input argument 'ext_real_set' is invalid! (%s)"
                           % (error))
                raise_error(err_msg, InputError, logger)

        # If a dict is given
        elif isinstance(ext_real_set, dict):
            # Check if ext_real_set contains correct keys
            if 'sam_set' not in ext_real_set.keys():
                err_msg = ("Input argument 'ext_real_set' does not contain key"
                           " 'sam_set'!")
                raise_error(err_msg, KeyError, logger)
            if 'mod_set' not in ext_real_set.keys():
                err_msg = ("Input argument 'ext_real_set' does not contain key"
                           " 'mod_set'!")
                raise_error(err_msg, KeyError, logger)

            # Try to extract ext_sam_set and ext_mod_set
            try:
                ext_sam_set = ext_real_set['sam_set']
                ext_mod_set = ext_real_set['mod_set']
            except Exception as error:
                err_msg = ("Input argument 'ext_real_set' is invalid! (%s)"
                           % (error))
                raise_error(err_msg, InputError, logger)

        # If anything else is given, it is invalid
        else:
            err_msg = "Input argument 'ext_real_set' is invalid!"
            raise_error(err_msg, InputError, logger)

        # Check ext_sam_set and ext_mod_set
        ext_sam_set = self._modellink._check_sam_set(ext_sam_set,
                                                     'ext_sam_set')
        ext_mod_set = self._modellink._check_mod_set(ext_mod_set,
                                                     'ext_mod_set')

        # Make sure that ext_sam_set and ext_mod_set are 2D
        ext_sam_set = np_array(ext_sam_set, ndmin=2)
        ext_mod_set = np_array(ext_mod_set, ndmin=2)

        # Check if both arrays contain the same number of samples
        if not(ext_sam_set.shape[0] == ext_mod_set.shape[0]):
            err_msg = ("External sample and model output sets do not contain "
                       "the same number of samples (%i != %i)!"
                       % (ext_sam_set.shape[0], ext_mod_set.shape[0]))
            raise_error(err_msg, ShapeError, logger)

        # Log that processing has been finished
        logger.info("Finished processing externally provided model realization"
                    " set of size %i." % (ext_sam_set.shape[0]))

        # If all checks are passed, return ext_sam_set and ext_mod_set
        return(ext_sam_set, ext_mod_set.T)

    # This function compiles pre-defined built-in code snippets and saves them
    def _compile_code_snippets(self):
        """
        Compiles all pre-defined built-in code snippets to code objects and
        saves them to :attr:`~code_objects`. These code objects are used for
        performing standard operations in the :meth:`~_evaluate_sam_set`
        method.

        """

        # Log that code snippets are being compiled
        logger = getCLogger('INIT')
        logger.info("Compiling built-in code snippets.")

        # Define dict of code snippet tuples
        code_objects = {}

        # ANALYZE
        # Define the various code snippets
        pre_code = compile("", '<string>', 'exec')
        eval_code = compile("", '<string>', 'exec')
        anal_code = compile("", '<string>', 'exec')
        post_code = compile("self.results = sam_set[sam_idx]", '<string>',
                            'exec')
        exit_code = compile("", '<string>', 'exec')

        # Combine code snippets into a tuple and add to dict
        code_objects['analyze'] = (pre_code, eval_code, anal_code, post_code,
                                   exit_code)

        # EVALUATE
        # Define the various code snippets
        # We have to use lists here to account for n_data differing with emul_i
        pre_code = compile(dedent("""
            adj_exp_val = [[] for _ in range(n_sam)]
            adj_var_val = [[] for _ in range(n_sam)]
            uni_impl_val_list = [[] for _ in range(n_sam)]
            emul_i_stop = np.zeros([n_sam])
            """), '<string>', 'exec')
        eval_code = compile(dedent("""
            adj_exp_val[sam_idx[j]] = adj_val[0]
            adj_var_val[sam_idx[j]] = adj_val[1]
            uni_impl_val_list[sam_idx[j]] = uni_impl_vals[j]
            """), '<string>', 'exec')
        anal_code = compile("emul_i_stop[sam_idx[j]] = i", '<string>', 'exec')
        post_code = compile(dedent("""
            adj_exp_val = self._comm.gather(np_array(adj_exp_val), 0)
            adj_var_val = self._comm.gather(np_array(adj_var_val), 0)
            uni_impl_val_list = self._comm.gather(np_array(uni_impl_val_list),
                                                  0)
            """), '<string>', 'exec')
        exit_code = compile(dedent("""
            adj_exp_val = np.concatenate(adj_exp_val, axis=1)
            adj_var_val = np.concatenate(adj_var_val, axis=1)
            uni_impl_val = np.concatenate(uni_impl_val_list, axis=1)
            self.results = (adj_exp_val, adj_var_val, uni_impl_val,
                            emul_i_stop, impl_check)
            """), '<string>', 'exec')

        # Combine code snippets into a tuple and add to dict
        code_objects['evaluate'] = (pre_code, eval_code, anal_code, post_code,
                                    exit_code)

        # HYBRID
        # Define the various code snippets
        pre_code = compile("lnprior = 0", '<string>', 'exec')
        eval_code = compile("", '<string>', 'exec')
        anal_code = compile(dedent("""
            lnprior = (np.log(1-impl_cut_val/self._impl_cut[i][0]) if
                       impl_check_val else -np.infty)
            """), '<string>', 'exec')
        post_code = compile(dedent("""
            lnprior = self._comm.bcast(lnprior, 0)
            self.results = (sam_set[sam_idx], lnprior)
            """), '<string>', 'exec')
        exit_code = compile("", '<string>', 'exec')

        # Combine code snippets into a tuple and add to dict
        code_objects['hybrid'] = (pre_code, eval_code, anal_code, post_code,
                                  exit_code)

        # PROJECT
        # Define the various code snippets
        pre_code = compile("impl_cut = np.zeros([n_sam])", '<string>', 'exec')
        eval_code = compile("", '<string>', 'exec')
        anal_code = compile("impl_cut[sam_idx[j]] = impl_cut_val", '<string>',
                            'exec')
        post_code = compile("", '<string>', 'exec')
        exit_code = compile("self.results = (impl_check, impl_cut)",
                            '<string>', 'exec')

        # Combine code snippets into a tuple
        code_objects['project'] = (pre_code, eval_code, anal_code, post_code,
                                   exit_code)

        # Save code_objects dict to memory
        self._code_objects = code_objects

        # Log again
        logger.info("Finished compiling code snippets.")

    # This function evaluates given sam_set in the emulator using code snippets
    @docstring_substitute(emul_i=std_emul_i_doc)
    def _evaluate_sam_set(self, emul_i, sam_set, exec_code):
        """
        Evaluates a provided set of emulator evaluation samples `sam_set` at a
        given emulator iteration `emul_i`.
        The provided tuple of code snippets `exec_code` are executed using
        Python's :func:`~exec` function at specific points during the analysis.

        Parameters
        ----------
        %(emul_i)s
        sam_set : 2D :obj:`~numpy.ndarray` object
            Array containing model parameter value sets to be evaluated in all
            emulator systems in emulator iteration `emul_i`.
        exec_code : {'analyze', 'evaluate', 'hybrid', 'project'} or tuple
            Tuple of five code snippets ``(pre_code, eval_code, anal_code,
            post_code, exit_code)`` to be executed at specific points during
            the analysis.
            If string, use one of the built-in tuples in :attr:`~code_objects`
            instead.

        Other parameters
        ----------------
        pre_code : str or code object
            Code snippet to be executed before the evaluation of `sam_set`
            starts.
        eval_code : str or code object
            Code snippet to be executed after the evaluation of each sample in
            `sam_set`.
        anal_code : str or code object
            Code snippet to be executed after the analysis of each sample in
            `sam_set`. This code snippet is only executed by the controller.
        post_code : str or code object
            Code snippet to be executed after the evaluation of `sam_set` ends.
        exit_code : str or code object
            Code snippet to be executed before returning the results of the
            evaluation of `sam_set`. This code snippet is only executed by the
            controller.

        Returns
        -------
        results : object
            The object that is assigned to :attr:`~results`, which is defaulted
            to *None* if no code snippet changes it. Preferably, the execution
            of `post_code` and/or `exit_code` modifies :attr:`~results`. All
            MPI ranks return it.

        Notes
        -----
        If any of the code snippets is provided as a string, it will be
        compiled into a code object before starting the evaluation.

        """

        # Determine number of samples
        n_sam = sam_set.shape[0]

        # Start logging
        logger = getCLogger('EVAL_SS')
        logger.info("Starting evaluation of sample set of size %i." % (n_sam))

        # Obtain code snippets
        if isinstance(exec_code, str):
            # If string is provided, use built-in tuple of code snippets
            pre_code, eval_code, anal_code, post_code, exit_code =\
                self._code_objects[exec_code]
        else:
            # If not, use provided code snippets
            pre_code, eval_code, anal_code, post_code, exit_code = exec_code

            # Compile any code snippets that were provided as a string
            if isinstance(pre_code, str):
                pre_code = compile(pre_code, '<string>', 'exec')
            if isinstance(eval_code, str):
                eval_code = compile(eval_code, '<string>', 'exec')
            if isinstance(anal_code, str):
                anal_code = compile(anal_code, '<string>', 'exec')
            if isinstance(post_code, str):
                post_code = compile(post_code, '<string>', 'exec')
            if isinstance(exit_code, str):
                exit_code = compile(exit_code, '<string>', 'exec')

        # Make a filled bool list containing which samples are plausible
        impl_check = np.ones(n_sam, dtype=bool)

        # Make a list of plausible sample indices
        sam_idx_full = np_array(range(n_sam))
        sam_idx = sam_idx_full[impl_check]

        # Mark all samples as plausible
        eval_sam_set = sam_set

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
                logger.info("Analyzing evaluation sample set of size %i in "
                            "emulator iteration %i." % (n_sam, i))

                # Make empty uni_impl_vals list
                uni_impl_vals = np.zeros([n_sam, self._emulator._n_data[i]])

                # Loop over all still plausible samples in sam_set
                for j, par_set in enumerate(eval_sam_set):
                    # Evaluate par_set
                    adj_val = self._emulator._evaluate(i, par_set)

                    # Calculate univariate implausibility value
                    uni_impl_vals[j] = self._get_uni_impl(i, par_set, *adj_val)

                    # Execute the eval_code snippet
                    exec(eval_code)

                # Gather the results on the controller after evaluating
                uni_impl_vals_list = self._comm.gather(uni_impl_vals, 0)

                # Controller performs implausibility analysis
                if self._is_controller:
                    # Convert uni_impl_vals_list to an array
                    uni_impl_vals_array =\
                        np.concatenate(uni_impl_vals_list, axis=1)

                    # Perform implausibility cutoff check on all elements
                    for j, uni_impl_val in enumerate(uni_impl_vals_array):
                        impl_check_val, impl_cut_val =\
                            self._do_impl_check(i, uni_impl_val)

                        # Modify impl_check with obtained impl_check_val
                        impl_check[sam_idx[j]] = impl_check_val

                        # Execute the anal_code snippet
                        exec(anal_code)

                    # Modify sam_idx with those that are still plausible
                    sam_idx = sam_idx_full[impl_check]

                # MPI Barrier
                self._comm.Barrier()

                # Broadcast the updated sam_idx to the workers
                sam_idx = self._comm.bcast(sam_idx, 0)

                # Check that sam_idx is still holding plausible samples
                if not len(sam_idx):
                    # If not, stop analysis
                    break
                else:
                    # If so, determine which samples are still plausible
                    eval_sam_set = sam_set[sam_idx]

        # If any other emulator type is used
        else:
            raise NotImplementedError

        # Execute the post_code snippet
        exec(post_code)

        # Log that analysis is finished
        logger.info("Finished analyzing evaluation sample set.")

        # Execute the exit_code snippet on the controller
        if self._is_controller:
            exec(exit_code)

        # Retrieve the results and delete the attribute
        results = self.results
        del self.results

        # Return the results
        return(results)

    # %% VISIBLE CLASS METHODS
    # This function analyzes the emulator and determines the plausible regions
    def analyze(self, *, impl_cut=None):
        """
        Analyzes the emulator at the last emulator iteration for a large number
        of emulator evaluation samples. All samples that survive the
        implausibility checks set by the provided `impl_cut`, are used in the
        construction of the next emulator iteration.

        Optional
        --------
        impl_cut : list of float or None. Default: None
            Incomplete, shortened impl_cut-offs list to be used during the
            analysis of this emulator iteration.
            If *None*, the currently set implausibility cut-off values
            (:attr:`~impl_cut`) will be used if this is the first analysis or
            the 'impl_cut' value in :attr:`~prism_dict` is used if this is a
            reanalysis.

        Generates
        ---------
        impl_sam : 2D :obj:`~numpy.ndarray` object
            Array containing all emulator evaluation samples that survived the
            implausibility checks.

        """

        # Get last emul_i
        emul_i = self._emulator._get_emul_i(None, True)

        # Begin logging
        logger = getCLogger('ANALYZE')
        logger.info("Analyzing emulator iteration %i." % (emul_i))

        # Controller checking whether this iteration can still be (re)analyzed
        if self._is_controller:
            # Try to access the ccheck of the next iteration
            try:
                self._emulator._ccheck[emul_i+1]
            # If it does not exist, current iteration can be analyzed
            except IndexError:
                pass
            # If it does exist, check if mod_real_set has been evaluated
            else:
                # If it has been evaluated, reanalysis is not possible
                if 'mod_real_set' not in self._emulator._ccheck[emul_i+1]:
                    err_msg = ("Construction of next emulator iteration (%i) "
                               "has already been started. Reanalysis of the "
                               "current iteration is not possible."
                               % (emul_i+1))
                    raise_error(err_msg, RequestError, logger)

        # Controller obtaining the emulator evaluation sample set
        if self._is_controller:
            # Save current time
            start_time1 = time()

            # Set the impl_cut list if analyzed before or impl_cut is not None
            # This is to keep the impl_par set by the user if initial analyze
            if impl_cut is not None or self._n_eval_sam[emul_i]:
                self._set_impl_par(impl_cut)

            # Save impl_par to hdf5
            self._save_data({
                    'impl_par': {
                        'impl_cut': self._impl_cut[emul_i],
                        'cut_idx': self._cut_idx[emul_i]}})

            # Create an emulator evaluation sample set
            eval_sam_set = self._get_eval_sam_set(emul_i)
            n_eval_sam = eval_sam_set.shape[0]

        # Remaining workers get dummy eval_sam_set
        else:
            eval_sam_set = []

        # Broadcast eval_sam_set to workers
        eval_sam_set = self._comm.bcast(eval_sam_set, 0)

        # Save current time again
        start_time2 = time()

        # Analyze eval_sam_set
        impl_sam = self._evaluate_sam_set(emul_i, eval_sam_set, 'analyze')

        # Controller finishing up
        if self._is_controller:
            # Obtain some timers
            end_time = time()
            time_diff_total = end_time-start_time1
            time_diff_eval = end_time-start_time2

            # Calculate the number of plausible samples left
            n_impl_sam = len(impl_sam)

            # Raise warning if no plausible samples were found
            if not n_impl_sam:
                warn_msg = ("No plausible regions were found. Constructing the"
                            " next iteration will not be possible.")
                raise_warning(warn_msg, RequestWarning, logger, 2)

            # Raise warning if n_impl_sam is less than n_cross_val
            elif(n_impl_sam < self._emulator._n_cross_val):
                warn_msg = ("Number of plausible samples is lower than the "
                            "number of cross validations used during "
                            "regression (%i < %i). Constructing the next "
                            "iteration will not be possible."
                            % (n_impl_sam, self._emulator._n_cross_val))
                raise_warning(warn_msg, RequestWarning, logger, 2)

            # Raise warning if n_impl_sam is less than n_sam_init
            elif(n_impl_sam < self._n_sam_init):
                warn_msg = ("Number of plausible samples is lower than the "
                            "number of samples in the first iteration (%i < "
                            "%i). Constructing the next iteration might not "
                            "produce a more accurate emulator."
                            % (n_impl_sam, self._n_sam_init))
                raise_warning(warn_msg, RequestWarning, logger, 2)

            # Save the results
            self._save_data({
                'impl_sam': impl_sam,
                'n_eval_sam': n_eval_sam})

            # Save statistics about analyze time, evaluation rate, par_space
            avg_eval_rate = n_eval_sam/time_diff_eval
            par_space_rem = (n_impl_sam/n_eval_sam)*100
            self._save_statistics(emul_i, {
                'tot_analyze_time': ['%.2f' % (time_diff_total), 's'],
                'avg_emul_eval_rate': ['%.2f' % (avg_eval_rate), '1/s'],
                'par_space_remaining': ['%#.3g' % (par_space_rem), '%'],
                'MPI_comm_size_anal': ['%i' % (self._size), '']})

            # Log that analysis has been finished
            msg1 = ("Finished analysis of emulator iteration in %.2f seconds, "
                    "averaging %.2f emulator evaluations per second."
                    % (time_diff_total, n_eval_sam/time_diff_eval))
            msg2 = ("There is %#.3g%% of parameter space remaining."
                    % (par_space_rem))
            logger.info(msg1)
            logger.info(msg2)
            print(msg1)
            print(msg2)

        # Display details about current state of pipeline
        self.details()

    # This function constructs a specified iteration of the emulator system
    # TODO: Make time and RAM cost plots
    # TODO: Fix the timers for interrupted constructs
    @docstring_substitute(emul_i=call_emul_i_doc, ext_set=ext_real_set_doc_d)
    def construct(self, emul_i=None, *, analyze=True, ext_real_set=None,
                  force=False):
        """
        Constructs the emulator at the specified emulator iteration `emul_i`,
        and performs an implausibility analysis on the emulator iteration right
        afterward if requested (:meth:`~analyze`).

        Optional
        --------
        %(emul_i)s
        analyze : bool. Default: True
            Bool indicating whether or not to perform an analysis after the
            specified emulator iteration has been successfully constructed,
            which is required for constructing the next iteration.
        %(ext_set)s
        force : bool. Default: False
            Controls what to do if the specified emulator iteration `emul_i`
            already (partly) exists.
            If *False*, finish construction of the specified iteration or skip
            it if already finished.
            If *True*, reconstruct the specified iteration entirely.

        Generates
        ---------
        A new HDF5-group with the emulator iteration as its name, in the loaded
        emulator master file, containing emulator data required for this
        emulator iteration.

        Notes
        -----
        Using an emulator iteration that has been (partly) constructed before,
        will finish construction or skip it if already finished when `force` is
        *False*; or it will delete that and all following iterations, and
        reconstruct the specified iteration when `force` is *True*. Using
        `emul_i` = 1 and `force` is *True* is equivalent to reconstructing the
        entire emulator.

        If no implausibility analysis is requested, then the implausibility
        parameters are read in from the *PRISM* parameters file and temporarily
        stored in memory in order to enable the usage of the :meth:`~evaluate`
        and :meth:`~prism._projection.Projection.project` methods.

        """

        # Log that a new emulator iteration is being constructed
        logger = getCLogger('CONSTRUCT')

        # Set emul_i correctly
        emul_i = self._emulator._get_emul_i(emul_i, False)

        # Controller performing checks regarding construction progress
        if self._is_controller:
            # Save current time
            start_time = time()

            # Check if force-parameter received a bool
            force = check_vals(force, 'force', 'bool')

            # Check if iteration was interrupted or not, or if force is True
            logger.info("Checking state of emulator iteration %i." % (emul_i))
            try:
                # If force is True, reconstruct full iteration
                if force:
                    logger.info("Emulator iteration %i has been requested to "
                                "be (re)constructed." % (emul_i))
                    c_from_start = 1

                # If interrupted at start, reconstruct full iteration
                elif('mod_real_set' in self._emulator._ccheck[emul_i]):
                    logger.info("Emulator iteration %i does not contain "
                                "evaluated model realization data. Will be "
                                "constructed from start." % (emul_i))
                    c_from_start = 1

                # If already finished, skip everything
                elif(emul_i <= self._emulator._emul_i):
                    msg = ("Emulator iteration %i has already been fully "
                           "constructed. Skipping construction process."
                           % (emul_i))
                    logger.info(msg)
                    print(msg)
                    c_from_start = None

                # If interrupted midway, do not reconstruct full iteration
                else:
                    logger.info("Construction of emulator iteration %i was "
                                "interrupted. Continuing from point of "
                                "interruption." % (emul_i))
                    c_from_start = 0

            # If never constructed before, construct full iteration
            except IndexError:
                logger.info("Emulator iteration %i has not been constructed."
                            % (emul_i))
                c_from_start = 1

        # Remaining workers
        else:
            c_from_start = None

        # Check if analyze-parameter received a bool
        analyze = check_vals(analyze, 'analyze', 'bool')

        # Broadcast construct_emul_i to workers
        c_from_start = self._comm.bcast(c_from_start, 0)

        # If iteration is already finished, show the details
        if c_from_start is None:
            self.details(emul_i)
            return

        # If iteration needs to be constructed from the beginning
        if c_from_start:
            # Log that construction of emulator iteration is being started
            logger.info("Starting construction of emulator iteration %i."
                        % (emul_i))

            # Activate worker mode
            with self.worker_mode:
                if self._is_controller:
                    # If this is the first iteration
                    if(emul_i == 1):
                        # Process ext_real_set
                        ext_sam_set, ext_mod_set =\
                            self._get_ext_real_set(ext_real_set)

                        # Obtain number of externally given model realizations
                        n_ext_sam = np.shape(ext_sam_set)[0]

                        # Create a new emulator
                        self._make_call('_emulator._create_new_emulator')

                        # Reload the data
                        self._load_data()

                        # Create initial set of model evaluation samples
                        n_sam_init = max(0, self._n_sam_init-n_ext_sam)
                        if n_sam_init:
                            logger.info("Creating initial model evaluation "
                                        "sample set of size %i."
                                        % (n_sam_init))
                            add_sam_set =\
                                lhd(n_sam_init, self._modellink._n_par,
                                    self._modellink._par_rng, 'center',
                                    self._criterion, constraints=ext_sam_set)
                            logger.info("Finished creating initial sample "
                                        "set.")
                        else:
                            add_sam_set = np_array([])

                    # If this is any other iteration
                    else:
                        # Get dummy ext_real_set
                        ext_sam_set, ext_mod_set = self._get_ext_real_set(None)

                        # Check if last iteration was analyzed, do so if not
                        if not self._n_eval_sam[emul_i-1]:
                            # Analyze previous iteration
                            logger.info("Previous emulator iteration has not "
                                        "been analyzed. Performing analysis "
                                        "first.")
                            self._make_call('analyze')

                        # Check if a new emulator iteration can be constructed
                        # If no plausible regions were found
                        if not self._n_impl_sam[emul_i-1]:
                            err_msg = ("No plausible regions were found in the"
                                       " analysis of the previous emulator "
                                       "iteration. Construction is not "
                                       "possible!")
                            raise_error(err_msg, RequestError, logger)

                        # If n_impl_sam is less than n_cross_val
                        elif(self._n_impl_sam[emul_i-1] <
                             self._emulator._n_cross_val):
                            err_msg = ("Number of plausible samples found in "
                                       "the analysis of the previous iteration"
                                       " is lower than the number of cross "
                                       "validations used during regression"
                                       " (%i < %i). Construction is not "
                                       "possible!"
                                       % (self._n_impl_sam[emul_i-1],
                                          self._emulator._n_cross_val))
                            raise_error(err_msg, RequestError, logger)

                        # Make the emulator prepare for a new iteration
                        reload = self._make_call(
                            '_emulator._prepare_new_iteration', emul_i)

                        # Make sure the correct pipeline data is loaded in
                        if reload:
                            self._load_data()

                        # Obtain additional sam_set
                        add_sam_set = self._impl_sam

                    # Obtain corresponding set of model evaluations
                    self._make_call('_get_iteration_data', emul_i, add_sam_set,
                                    ext_sam_set, ext_mod_set)

        # If iteration needs to be constructed from midway
        else:
            # Log that construction of emulator iteration is continued
            logger.info("Continuing construction of emulator iteration %i."
                        % (emul_i))

        # Construct emulator iteration
        self._emulator._construct_iteration(emul_i)

        # Controller finishing up construction process
        if self._is_controller:
            # Save that emulator iteration has not been analyzed yet
            self._save_data({
                'impl_sam': np_array([]),
                'n_eval_sam': 0})
            self._set_impl_par(None)

            # Log that construction has been completed
            time_diff_total = time()-start_time
            self._save_statistics(emul_i, {
                'tot_construct_time': ['%.2f' % (time_diff_total), 's']})
            msg = ("Finished construction of emulator iteration in %.2f "
                   "seconds." % (time_diff_total))
            logger.info(msg)
            print(msg)

        # Analyze the emulator iteration if requested
        if analyze:
            self.analyze()
        # If not, show details
        else:
            self.details(emul_i)

    # This function allows one to view the pipeline details/properties
    # TODO: Allow the viewing of the entire polynomial function in SymPy
    @docstring_substitute(emul_i=user_emul_i_doc)
    def details(self, emul_i=None):
        """
        Prints the details/properties of the currently loaded :obj:`~Pipeline`
        instance at given emulator iteration `emul_i`. See ``Props`` for
        detailed descriptions of all printed properties.

        Optional
        --------
        %(emul_i)s

        Props
        -----
        Working directory
            The relative path to the working directory of the emulator starting
            at the current working directory.
        Emulator type
            The type of this emulator, corresponding to the
            :attr:`~prism.emulator.Emulator.emul_type` of the provided
            `emul_type` during :class:`~Pipeline` initialization.
            If no emulator type was provided during initialization, this is
            'default'.
        ModelLink subclass
            Name of the :class:`~prism.modellink.ModelLink` subclass
            used to construct this emulator.
        Emulation method
            Indicates the combination of regression and Gaussian emulation
            methods that have been used for this emulator.
        Mock data used?
            Whether or not mock data has been used to construct this emulator.
            If so, the printed estimates for all model parameters are the
            parameter values used to create the mock data.

        ----

        Emulator iteration
            The iteration of the emulator this details overview is about. By
            default, this is the last (partly) constructed iteration.
        Construction completed?
            Whether or not the construction of this emulator iteration is
            completed. If not, the missing components for each emulator system
            are listed and the remaining information of this iteration is not
            printed.
        Plausible regions?
            Whether or not plausible regions have been found during the
            analysis of this emulator iteration. If no analysis has been done
            yet, "N/A" will be printed.
        Projections available?
            Whether or not projections have been created for this emulator
            iteration. If projections are available and analysis has been done,
            but with different implausibility cut-offs, a "desync" note is
            added. Also prints number of available projections versus maximum
            number of projections in parentheses.

        ----

        # of model evaluation samples
            The total number of model evaluation samples used to construct all
            emulator iterations up to this iteration, with the number for every
            individual iteration in parentheses.
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
            number of model parameters defined in the used
            :class:`~prism.modellink.ModelLink` subclass.
        # of emulated data points
            The number of data points that have been emulated in this
            emulator iteration.
        # of emulator systems
            The total number of emulator systems that are required in this
            emulator. The number of active emulator systems is equal to the
            number of data points.

        ----

        Parameter space
            Lists the name, lower and upper value boundaries and estimate (if
            provided) of all model parameters defined in the used
            :class:`~prism.modellink.ModelLink` subclass. An asterisk is
            printed in front of the parameter name if this model parameter was
            considered active during the construction of this emulator
            iteration. A question mark is used instead if the construction of
            this emulator iteration is not finished.

        """

        # If emulator has not been built yet, return immediately
        if(len(self._emulator._ccheck) == 1):
            return

        # Define details logger
        logger = getCLogger("DETAILS")
        logger.info("Collecting details about current pipeline instance.")

        # Check if last emulator iteration is finished constructing
        ccheck_flag = (delist(self._emulator._ccheck[-1]) == [])

        # Gather the ccheck_flags on all ranks to see if all are finished
        ccheck_flag = self._comm.allreduce(ccheck_flag, op=MPI.MIN)

        # Obtain correct emul_i depending on the construction progress
        emul_i = self._emulator._get_emul_i(emul_i, ccheck_flag)

        # Gather ccheck information on the controller
        ccheck_list = self._comm.gather(self._emulator._ccheck[emul_i], 0)

        # Controller generating the entire details overview
        if self._is_controller:
            # Flatten the received ccheck_list
            ccheck_flat = [[] for _ in range(self._emulator._n_emul_s_tot)]
            ccheck_flat.extend(ccheck_list[0][self._emulator._n_emul_s:])
            for rank, ccheck_rank in enumerate(ccheck_list):
                for emul_s, ccheck in zip(self._emulator._emul_s_to_core[rank],
                                          ccheck_rank):
                    ccheck_flat[emul_s] = ccheck

            # Get max lengths of various strings for parameter space section
            name_len =\
                max([len(par_name) for par_name in self._modellink._par_name])
            lower_len =\
                max([len(str(i)) for i in self._modellink._par_rng[:, 0]])
            upper_len =\
                max([len(str(i)) for i in self._modellink._par_rng[:, 1]])
            est_lengths = [len('%.5f' % (i)) for i in self._modellink._par_est
                           if i is not None]
            est_len = max(est_lengths) if len(est_lengths) else 0

            # Open hdf5-file
            with self._File('r', None) as file:
                # Check if projection data is available by trying to access it
                try:
                    data_set = file['%i/proj_hcube' % (emul_i)]
                except KeyError:
                    proj = 0
                    n_proj = 0

                # If projection data is available
                else:
                    # Get the number of projections and used impl_cut-offs
                    proj = 0
                    n_proj = len(data_set.keys())
                    for hcube in data_set.values():
                        p_impl_cut = hcube.attrs['impl_cut']
                        p_cut_idx = hcube.attrs['cut_idx']

                        # Check if projections were made with the same impl_cut
                        if(len(p_impl_cut) == len(self._impl_cut[emul_i]) and
                           (p_impl_cut == self._impl_cut[emul_i]).all() and
                           p_cut_idx == self._cut_idx[emul_i]):
                            # If it was, projections are synced
                            proj = 1
                        else:
                            # If not, projections are desynced
                            proj = 2
                            break

            # Log file being closed
            logger.info("Finished collecting details about current pipeline "
                        "instance.")

            # Obtain number of model parameters
            n_par = self._modellink._n_par

            # Obtain number of data points
            n_data = self._emulator._n_data_tot[emul_i]

            # Obtain number of emulator systems
            n_emul_s = self._emulator._n_emul_s_tot

            # Save if requested emulator iteration is fully constructed
            ccheck_flag = (delist(ccheck_flat) == [])

            # Determine the relative path to the working directory
            pwd = os.getcwd()
            if(path.splitdrive(self._working_dir)[0].lower() !=
               path.splitdrive(pwd)[0].lower()):
                working_dir_rel_path = self._working_dir
            else:
                working_dir_rel_path = path.relpath(self._working_dir, pwd)

            # Set width of first details column
            width = 31

            # PRINT DETAILS
            # HEADER
            print("\nPIPELINE DETAILS")
            print("="*width)

            # GENERAL
            print("\nGENERAL")
            print("-"*width)

            # General details about loaded emulator
            print("{0: <{1}}\t'{2}'".format("Working directory", width,
                                            working_dir_rel_path))
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
            if ccheck_flag:
                # Determine the number of (active) parameters
                n_active_par = len(self._emulator._active_par[emul_i])

                # Calculate the maximum number of projections
                n_proj_max = n_active_par
                n_proj_max += nCr(n_active_par, 2) if(n_par > 2) else 0

                # Flag details
                print("{0: <{1}}\t{2}".format("Construction completed?", width,
                                              "Yes"))
                if not self._n_eval_sam[emul_i]:
                    print("{0: <{1}}\t{2}".format("Plausible regions?", width,
                                                  "N/A"))
                else:
                    print("{0: <{1}}\t{2}".format(
                        "Plausible regions?", width,
                        "Yes" if self._n_impl_sam[emul_i] else "No"))
                if not proj:
                    print("{0: <{1}}\t{2}".format("Projections available?",
                                                  width, "No"))
                else:
                    print("{0: <{1}}\t{2} ({3}/{4}){5}".format(
                        "Projections available?", width,
                        "Yes", n_proj, n_proj_max,
                        "" if proj == 1 else ", desynced"))
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
                    print("{0: <{1}}\t{2:#.3g}%".format(
                        "% of parameter space remaining", width,
                        (self._n_impl_sam[emul_i] /
                         self._n_eval_sam[emul_i])*100))
                print("{0: <{1}}\t{2}/{3}".format(
                    "# of active/total parameters", width,
                    n_active_par, n_par))
                print("{0: <{1}}\t{2}".format("# of emulated data points",
                                              width, n_data))
                print("{0: <{1}}\t{2}".format("# of emulator systems",
                                              width, n_emul_s))

            # If not, print which components are still missing
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
                    "'active_par_data'?", width-4, "No (%s)" % (ccheck_i) if
                    len(ccheck_i) else "Yes"))

                # Check if all regression processes have been done if requested
                if self._emulator._method.lower() in ('regression', 'full'):
                    ccheck_i = [i for i in range(n_emul_s) if
                                'regression' in ccheck_flat[i]]
                    print("  - {0: <{1}}\t{2}".format(
                        "'regression'?", width-4, "No (%s)" % (ccheck_i) if
                        len(ccheck_i) else "Yes"))

                # Check if all covariance matrices have been determined
                ccheck_i = [i for i in range(n_emul_s) if
                            'cov_mat' in ccheck_flat[i]]
                print("  - {0: <{1}}\t{2}".format(
                    "'cov_mat'?", width-4, "No (%s)" % (ccheck_i) if
                    len(ccheck_i) else "Yes"))

                # Check if all exp_dot_terms have been determined
                ccheck_i = [i for i in range(n_emul_s) if
                            'exp_dot_term' in ccheck_flat[i]]
                print("  - {0: <{1}}\t{2}".format(
                    "'exp_dot_term'?", width-4, "No (%s)" % (ccheck_i) if
                    len(ccheck_i) else "Yes"))
            print("-"*width)

            # PARAMETER SPACE
            print("\nPARAMETER SPACE")
            print("-"*width)

            # Define string format if no par_ests are provided
            str_format1 = "{6}{0: <{1}}: [{2: >{3}}, {4: >{5}}]"

            # Define string format if this par_est was provided
            str_format2 = "{8}{0: <{1}}: [{2: >{3}}, {4: >{5}}] ({6: >{7}.5f})"

            # Define string format if this par_est was not provided
            str_format3 = "{8}{0: <{1}}: [{2: >{3}}, {4: >{5}}] ({6:->{7}})"

            # Print details about every model parameter in parameter space
            for i in range(n_par):
                # Determine what string to use for the active flag
                if not ccheck_flag:
                    active_str = "?"
                elif i in self._emulator._active_par[emul_i]:
                    active_str = "*"
                else:
                    active_str = " "

                # Check if par_est is given and use correct string formatting
                if not est_len:
                    print(str_format1.format(
                        self._modellink._par_name[i], name_len,
                        self._modellink._par_rng[i, 0], lower_len,
                        self._modellink._par_rng[i, 1], upper_len,
                        active_str))
                elif self._modellink._par_est[i] is not None:
                    print(str_format2.format(
                        self._modellink._par_name[i], name_len,
                        self._modellink._par_rng[i, 0], lower_len,
                        self._modellink._par_rng[i, 1], upper_len,
                        self._modellink._par_est[i], est_len, active_str))
                else:
                    print(str_format3.format(
                        self._modellink._par_name[i], name_len,
                        self._modellink._par_rng[i, 0], lower_len,
                        self._modellink._par_rng[i, 1], upper_len,
                        "", est_len, active_str))

            # FOOTER
            print("="*width)

            # Flush the console
            sys.stdout.flush()

        # MPI Barrier
        self._comm.Barrier()

    # This function allows the user to evaluate a given sam_set in the emulator
    # TODO: Rewrite entire function to enable clarity about what is returned
    @docstring_substitute(emul_i=user_emul_i_doc)
    def evaluate(self, sam_set, emul_i=None):
        """
        Evaluates the given model parameter sample set `sam_set` up to given
        emulator iteration `emul_i`.
        The output of this function depends on the number of dimensions in
        `sam_set`. The output is always provided on the controller rank.

        Parameters
        ----------
        sam_set : 1D or 2D array_like or dict
            Array containing model parameter value sets to be evaluated in the
            emulator up to emulator iteration `emul_i`.

        Optional
        --------
        %(emul_i)s

        Returns (if 2D `sam_set`)
        -------------------------
        impl_check : list of bool
            List of bool indicating whether or not the given samples passed the
            implausibility check at the given emulator iteration `emul_i`.
        emul_i_stop : list of int
            List containing the last emulator iterations at which the given
            samples are still within the plausible region of the emulator.
        adj_exp_val : 2D :obj:`~numpy.ndarray` object
            Array containing the adjusted expectation values for all given
            samples.
        adj_var_val : 2D :obj:`~numpy.ndarray` object
            Array containing the adjusted variance values for all given
            samples.
        uni_impl_val : 2D :obj:`~numpy.ndarray` object
            Array containing the univariate implausibility values for all given
            samples.

        Prints (if 1D `sam_set`)
        ------------------------
        emul_i_stop : int
            Last emulator iteration at which the given sample is still within
            the plausible region of the emulator.
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
        the values are used that were read in when the emulator was loaded or
        that have been set by the user.

        """

        # Get emulator iteration
        emul_i = self._emulator._get_emul_i(emul_i, True)

        # Do some logging
        logger = getCLogger('EVALUATE')
        logger.info("Evaluating emulator iteration %i for provided set of "
                    "model parameter samples." % (emul_i))

        # Check if sam_set was provided as a dict
        if isinstance(sam_set, dict):
            # Make sure that sam_set is a SortedDict
            sam_set = sdict(sam_set)

            # Return it to normal
            sam_set = np_array(sam_set.values()).T

        # Controller checking the contents of sam_set
        if self._is_controller:
            # Check sam_set
            sam_set = self._modellink._check_sam_set(sam_set, 'sam_set')

            # If ndim == 1, convert to 2D array and print output later
            if(sam_set.ndim == 1):
                sam_set = np_array(sam_set, ndmin=2)
                print_output = 1
            # If ndim == 2, return output later
            else:
                print_output = 0

        # The workers make sure that sam_set is also two-dimensional
        else:
            sam_set = np_array(sam_set, ndmin=2)

        # MPI Barrier
        self._comm.Barrier()

        # Analyze sam_set
        results = self._evaluate_sam_set(emul_i, sam_set, 'evaluate')

        # Do more logging
        logger.info("Finished evaluating emulator.")

        # Controller finishing up
        if self._is_controller:
            # Extract data arrays from results
            adj_exp_val, adj_var_val, uni_impl_val, emul_i_stop, impl_check =\
                results

            # If print_output is True, print the results
            if print_output:
                # Print results
                pr_str = "Evaluation results of %s" % (sam_set[0])
                print(pr_str)
                print("-"*len(pr_str))
                if impl_check[0]:
                    print("Plausible? Yes")
                else:
                    print("Plausible? No")
                print("emul_i_stop = %i" % (emul_i_stop[0]))
                print("adj_exp_val = %s" % (adj_exp_val[0]))
                print("adj_var_val = %s" % (adj_var_val[0]))
                print("sigma_val = %s" % (np.sqrt(adj_var_val[0])))
                print("uni_impl_val = %s" % (uni_impl_val[0]))

            # Else, return the results
            else:
                # MPI Barrier for controller
                self._comm.Barrier()

                # Return results
                return({'impl_check': impl_check.tolist(),
                        'emul_i_stop': emul_i_stop,
                        'adj_exp_val': adj_exp_val,
                        'adj_var_val': adj_var_val,
                        'uni_impl_val': uni_impl_val})

        # MPI Barrier
        self._comm.Barrier()

    # This function simply executes self.__call__
    @docstring_copy(__call__)
    def run(self, emul_i=None, *, force=False):
        self(emul_i, force=force)
