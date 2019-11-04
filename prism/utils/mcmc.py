# -*- coding: utf-8 -*-

"""
MCMC
====
Provides several functions that allow for *PRISM* to be connected more easily
to MCMC routines.

"""


# %% IMPORTS
# Built-in imports
from inspect import isfunction
import warnings

# Package imports
from e13tools import InputError
from e13tools.sampling import lhd
from e13tools.utils import docstring_substitute
import numpy as np
from numpy.random import multivariate_normal
from sortedcontainers import SortedDict as sdict
from tqdm import tqdm

# PRISM imports
from prism._docstrings import user_emul_i_doc
from prism._internal import RequestError, check_vals, np_array
from prism._pipeline import Pipeline

# All declaration
__all__ = ['get_hybrid_lnpost_fn', 'get_walkers']


# %% FUNCTION DEFINITIONS
# This function returns a hybrid version of the lnpost function
@docstring_substitute(emul_i=user_emul_i_doc)
def get_hybrid_lnpost_fn(lnpost_fn, pipeline_obj, *, emul_i=None,
                         unit_space=False, impl_prior=True, par_dict=False):
    """
    Returns a function definition ``hybrid_lnpost(par_set, *args, **kwargs)``.

    This `hybrid_lnpost()` function can be used to calculate the natural
    logarithm of the posterior probability, which analyzes a given `par_set`
    first in the provided `pipeline_obj` at iteration `emul_i` and passes it to
    `lnpost_fn` if it is plausible.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    lnpost_fn : function
        Function definition that needs to be called if the provided `par_set`
        is plausible in iteration `emul_i` of `pipeline_obj`. The used call
        signature is ``lnpost_fn(par_set, *args, **kwargs)``. All MPI ranks
        will call this function unless called within the
        :attr:`~prism.Pipeline.worker_mode` context manager.
    pipeline_obj : :obj:`~prism.Pipeline` object
        The instance of the :class:`~prism.Pipeline` class that needs
        to be used for determining the validity of the proposed sampling step.

    Optional
    --------
    %(emul_i)s
    unit_space : bool. Default: False
        Bool determining whether or not `par_set` will be given in unit space.
    impl_prior : bool. Default: True
        Bool determining whether or not the `hybrid_lnpost()` function should
        use the implausibility values of a given `par_set` as an additional
        prior.
    par_dict : bool. Default: False
        Bool determining whether or not `par_set` will be an array_like
        (*False*) or a dict (*True*).

    Returns
    -------
    hybrid_lnpost : function
        Definition of the function ``hybrid_lnpost(par_set, *args, **kwargs)``.

    See also
    --------
    :func:`~get_walkers`
        Analyzes proposed `init_walkers` and returns valid `p0_walkers`.

    :attr:`~prism.Pipeline.worker_mode`
        Special context manager within which all code is executed in worker
        mode.

    Note
    ----
    The input arguments `unit_space` and `par_dict` state in what form
    `par_set` will be provided to the `hybrid_lnpost()` function, such that it
    can be properly converted to the format used in :class:`~prism.Pipeline`.
    The `par_set` that is passed to `lnpost_fn` is unchanged.

    Warning
    -------
    Calling this function factory will disable all regular logging in
    `pipeline_obj` (:attr:`~prism.Pipeline.do_logging` set to *False*), in
    order to avoid having the same message being logged every time
    `hybrid_lnpost()` is called.

    """

    # Check if lnpost_fn is a function
    if not isfunction(lnpost_fn):
        raise InputError("Input argument 'lnpost_fn' is not a callable "
                         "function definition!")

    # Make abbreviation for pipeline_obj
    pipe = pipeline_obj

    # Check if provided pipeline_obj is an instance of the Pipeline class
    if not isinstance(pipe, Pipeline):
        raise TypeError("Input argument 'pipeline_obj' must be an instance of "
                        "the Pipeline class!")

    # Check if the provided pipeline_obj uses a default emulator
    if(pipe._emulator._emul_type != 'default'):
        raise InputError("Input argument 'pipeline_obj' does not use a default"
                         " emulator!")

    # Get emulator iteration
    emul_i = pipe._emulator._get_emul_i(emul_i)

    # Check if unit_space is a bool
    unit_space = check_vals(unit_space, 'unit_space', 'bool')

    # Check if impl_prior is a bool
    impl_prior = check_vals(impl_prior, 'impl_prior', 'bool')

    # Check if par_dict is a bool
    par_dict = check_vals(par_dict, 'par_dict', 'bool')

    # Disable PRISM logging
    pipe.do_logging = False

    # Define hybrid_lnpost function
    def hybrid_lnpost(par_set, *args, **kwargs):
        """
        Calculates the natural logarithm of the posterior probability of
        `par_set` using the provided function `lnpost_fn`, in addition to
        constraining it first with the emulator defined in the `pipeline_obj`.

        This function needs to be called by all MPI ranks unless called within
        the :attr:`~prism.Pipeline.worker_mode` context manager.

        Parameters
        ----------
        par_set : 1D array_like or dict
            Sample to calculate the posterior probability for. This sample is
            first analyzed in `pipeline_obj` and only given to `lnpost_fn` if
            it is plausible. If `par_dict` is *True*, this is a dict.
        args : positional arguments
            Positional arguments that need to be passed to `lnpost_fn`.
        kwargs : keyword arguments
            Keyword arguments that need to be passed to `lnpost_fn`.

        Returns
        -------
        lnp : float
            The natural logarithm of the posterior probability of `par_set`, as
            determined by `lnpost_fn` if `par_set` is plausible. If
            `impl_prior` is *True*, `lnp` is calculated as `lnprior` +
            `lnpost_fn()`, with `lnprior` the natural logarithm of the first
            implausibility cut-off value of `par_set` scaled with its maximum.

        """

        # If par_dict is True, convert par_set to a NumPy array
        if par_dict:
            sam = np_array(sdict(par_set).values(), ndmin=2)
        else:
            sam = np_array(par_set, ndmin=2)

        # If unit_space is True, convert par_set to par_space
        if unit_space:
            sam = pipe._modellink._to_par_space(sam)

        # Check if par_set is within parameter space and return -inf if not
        par_rng = pipe._modellink._par_rng
        if not ((par_rng[:, 0] <= sam[0])*(sam[0] <= par_rng[:, 1])).all():
            return(-np.infty)

        # Check what sampling is requested and analyze par_set
        if impl_prior:
            impl_sam, lnprior = pipe._make_call('_evaluate_sam_set', emul_i,
                                                sam, 'hybrid')
        else:
            impl_sam = pipe._make_call('_evaluate_sam_set', emul_i, sam,
                                       'analyze')
            lnprior = 0

        # If par_set is plausible, call lnpost_fn
        if len(impl_sam):
            return(lnprior+lnpost_fn(par_set, *args, **kwargs))

        # If par_set is not plausible, return -inf
        else:
            return(-np.infty)

    # Check if model in ModelLink can be single-called, raise warning if not
    if pipe._is_controller and not pipe._modellink._single_call:
        warn_msg = ("ModelLink bound to provided Pipeline object solely "
                    "requests multi-calls. Using MCMC may not be possible.")
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    # Return hybrid_lnpost function definition
    return(hybrid_lnpost)


# This function returns a set of valid MCMC walkers
@docstring_substitute(emul_i=user_emul_i_doc)
def get_walkers(pipeline_obj, *, emul_i=None, init_walkers=None,
                req_n_walkers=None, unit_space=False, lnpost_fn=None,
                **kwargs):
    """
    Analyzes proposed `init_walkers` and returns plausible `p0_walkers`.

    Analyzes sample set `init_walkers` in the provided `pipeline_obj` at
    iteration `emul_i` and returns all samples that are plausible to be used as
    starting positions for MCMC walkers. The provided samples and returned
    walkers should be/are given in unit space if `unit_space` is *True*.

    If `init_walkers` is *None*, returns :attr:`~prism.Pipeline.impl_sam`
    instead if it is available.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    pipeline_obj : :obj:`~prism.Pipeline` object
        The instance of the :class:`~prism.Pipeline` class that needs to be
        used for determining the plausibility of the proposed starting
        positions.

    Optional
    --------
    %(emul_i)s
    init_walkers : 2D array_like, dict, int or None. Default: None
        Sample set of proposed initial MCMC walker positions. All plausible
        samples in `init_walkers` will be returned.
        If int, generate an LHD of provided size and return all plausible
        samples.
        If *None*, return :attr:`~prism.Pipeline.impl_sam` corresponding to
        iteration `emul_i` instead.
    req_n_walkers : int or None. Default: None
        The minimum required number of plausible starting positions that should
        be returned. If *None*, all plausible starting positions in
        `init_walkers` are returned instead.

        .. versionadded:: 1.2.0
    unit_space : bool. Default: False
        Bool determining whether or not the provided samples and returned
        walkers are given in unit space.
    lnpost_fn : function or None. Default: None
        If function, call :func:`~get_hybrid_lnpost_fn` using `lnpost_fn` and
        the same values for `pipeline_obj`, `emul_i` and `unit_space`, and
        return the resulting function definition `hybrid_lnpost()`. Any
        additionally provided `kwargs` are also passed to it.

    Returns
    -------
    n_walkers : int
        Number of returned MCMC walkers. Note that this number can be higher
        than `req_n_walkers` if not *None*.
    p0_walkers : 2D :obj:`~numpy.ndarray` object or dict
        Array containing plausible starting positions of valid MCMC walkers.
        If `init_walkers` was provided as a dict, `p0_walkers` will be a dict.
    hybrid_lnpost : function (if `lnpost_fn` is a function)
        The function returned by :func:`~get_hybrid_lnpost_fn` using
        `lnpost_fn`, `pipeline_obj`, `emul_i`, `unit_space` and `kwargs` as the
        input values.

    See also
    --------
    :func:`~get_hybrid_lnpost_fn`
        Returns a function definition ``hybrid_lnpost(par_set, *args,
        **kwargs)``.

    :attr:`~prism.Pipeline.worker_mode`
        Special context manager within which all code is executed in worker
        mode.

    Notes
    -----
    If `init_walkers` is *None* and emulator iteration `emul_i` has not been
    analyzed yet, a :class:`~prism._internal.RequestError` will be raised.

    If `req_n_walkers` is not *None*, a custom Metropolis-Hastings sampling
    algorithm is used to generate the required number of starting positions.
    All plausible samples in `init_walkers` are used as the start of every MCMC
    chain. Note that if the number of plausible samples in `init_walkers` is
    small, it is possible that the returned `p0_walkers` are not spread out
    properly over parameter space.

    """

    # Make abbreviation for pipeline_obj
    pipe = pipeline_obj

    # Check if provided pipeline_obj is an instance of the Pipeline class
    if not isinstance(pipe, Pipeline):
        raise TypeError("Input argument 'pipeline_obj' must be an instance of "
                        "the Pipeline class!")

    # Check if the provided pipeline_obj uses a default emulator
    if(pipe._emulator._emul_type != 'default'):
        raise InputError("Input argument 'pipeline_obj' does not use a default"
                         " emulator!")

    # Get emulator iteration
    emul_i = pipe._emulator._get_emul_i(emul_i)

    # If req_n_walkers is not None, check if it is an integer
    if req_n_walkers is not None:
        req_n_walkers = check_vals(req_n_walkers, 'req_n_walkers', 'int',
                                   'pos')

    # Check if unit_space is a bool
    unit_space = check_vals(unit_space, 'unit_space', 'bool')

    # Assume that walkers are not to be returned as a dict
    walker_dict = False

    # Check if lnpost_fn is None and try to get hybrid_lnpost function if not
    if lnpost_fn is not None:
        try:
            hybrid_lnpost =\
                get_hybrid_lnpost_fn(lnpost_fn, pipe, emul_i=emul_i,
                                     unit_space=unit_space, **kwargs)
        except InputError:
            raise InputError("Input argument 'lnpost_fn' is invalid!")

    # If init_walkers is None, use impl_sam of emul_i
    if init_walkers is None:
        # Controller checking if emul_i has already been analyzed
        if pipe._is_controller:
            # If iteration has not been analyzed, raise error
            if not pipe._n_eval_sam[emul_i]:
                raise RequestError("Emulator iteration %i has not been "
                                   "analyzed yet!" % (emul_i))
            # If iteration is last iteration, init_walkers is current impl_sam
            elif(emul_i == pipe._emulator._emul_i):
                init_walkers = pipe._impl_sam
            # If iteration is not last, init_walkers is previous impl_sam
            else:
                init_walkers = pipe._emulator._sam_set[emul_i+1]

            # Make sure to make a copy of init_walkers to avoid modifications
            init_walkers = init_walkers.copy()

        # Broadcast init_walkers to workers as p0_walkers
        p0_walkers = pipe._comm.bcast(init_walkers, 0)

    # If init_walkers is not None, use provided samples or LHD size
    else:
        # Controller checking if init_walkers is valid
        if pipe._is_controller:
            # If init_walkers is an int, create LHD of provided size
            if isinstance(init_walkers, int):
                # Check if provided integer is positive
                n_sam = check_vals(init_walkers, 'init_walkers', 'pos')

                # Create LHD of provided size
                init_walkers = lhd(n_sam, pipe._modellink._n_par,
                                   pipe._modellink._par_rng, 'center',
                                   pipe._criterion, 100)

            # If init_walkers is not an int, it must be array_like or dict
            else:
                # If init_walkers is provided as a dict, convert it
                if isinstance(init_walkers, dict):
                    # Make sure that init_walkers is a SortedDict
                    init_walkers = sdict(init_walkers)

                    # Convert it to normal
                    init_walkers = np_array(init_walkers.values()).T

                    # Return p0_walkers as a dict
                    walker_dict = True

                # Make sure that init_walkers is a NumPy array
                init_walkers = np_array(init_walkers, ndmin=2)

                # If unit_space is True, convert init_walkers to par_space
                if unit_space:
                    init_walkers = pipe._modellink._to_par_space(init_walkers)

                # Check if init_walkers is valid
                init_walkers = pipe._modellink._check_sam_set(init_walkers,
                                                              'init_walkers')

        # Broadcast init_walkers to workers
        init_walkers = pipe._comm.bcast(init_walkers, 0)

        # Analyze init_walkers and save them as p0_walkers
        p0_walkers = pipe._evaluate_sam_set(emul_i, init_walkers, 'analyze')

    # Check if init_walkers is not empty and raise error if it is
    if not p0_walkers.shape[0]:
        raise InputError("Input argument 'init_walkers' contains no plausible "
                         "samples!")

    # If req_n_walkers is not None, use MH MCMC to find all required walkers
    if req_n_walkers is not None:
        n_walkers, p0_walkers = _do_mh_walkers(pipe, p0_walkers, req_n_walkers)
    else:
        p0_walkers = np.unique(p0_walkers, axis=0)
        n_walkers = p0_walkers.shape[0]

    # Check if p0_walkers needs to be converted
    if unit_space:
        p0_walkers = pipe._modellink._to_unit_space(p0_walkers)

    # Check if p0_walkers needs to be returned as a dict
    if walker_dict:
        p0_walkers = sdict(zip(pipe._modellink._par_name, p0_walkers.T))

    # Check if hybrid_lnpost was requested and return it as well if so
    if lnpost_fn is not None:
        return(n_walkers, p0_walkers, hybrid_lnpost)
    else:
        return(n_walkers, p0_walkers)


# %% HIDDEN FUNCTION DEFINITIONS
# This function uses MH sampling to find req_n_walkers initial positions
def _do_mh_walkers(pipeline_obj, p0_walkers, req_n_walkers):
    """
    Generates `req_n_walkers` unique starting positions for MCMC walkers using
    Metropolis-Hastings sampling and the provided `pipeline_obj` and
    `p0_walkers`.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    pipeline_obj : :obj:`~prism.Pipeline` object
        The instance of the :class:`~prism.Pipeline` class that needs to be
        used for determining the validity of a proposed sampling step.
    p0_walkers : 2D :obj:`~numpy.ndarray` object
        Sample set of starting positions for the MCMC chains.
    req_n_walkers : int
        The minimum required number of unique MCMC walker positions that must
        be returned.

    Returns
    -------
    n_walkers : int
        Number of unique MCMC walker positions that are returned.
    walkers : 2D :obj:`~numpy.ndarray` object
        Array containing all unique MCMC walker positions.

    Note
    ----
    Executing this function will temporarily disable all regular logging in the
    provided :obj:`~prism.Pipeline` object. If logging was enabled before this
    function was executed, it will be enabled again afterward.

    """

    # Make abbreviation for pipeline_obj
    pipe = pipeline_obj

    # Define function to check if proposed sam_set should be accepted
    def advance_chain(sam_set):
        # Make sure that sam_set is 2D
        sam_set = np_array(sam_set, ndmin=2)

        # Check if sam_set is within parameter space and reject if not
        par_rng = pipe._modellink._par_rng
        accept = ((par_rng[:, 0] <= sam_set)*(sam_set <= par_rng[:, 1])).all(1)

        # Evaluate all non-rejected samples and accept if plausible
        emul_i = pipe._emulator._emul_i
        accept[accept] = pipe._make_call('_evaluate_sam_set', emul_i,
                                         sam_set[accept], 'project')[0]

        # Return which samples should be accepted or rejected
        return(accept)

    # Initialize array of final walkers
    n_walkers = p0_walkers.shape[0]
    walkers = np.empty([req_n_walkers+n_walkers-1, pipe._modellink._n_par])
    walkers[:n_walkers] = p0_walkers

    # Check if logging is currently turned on
    was_logging = bool(pipe.do_logging)

    # Make sure that logging is turned off
    pipe.do_logging = False

    # Use worker mode
    with pipe.worker_mode:
        if pipe._is_controller:
            # Initialize progress bar
            pbar = tqdm(desc="Finding walkers", total=req_n_walkers,
                        initial=n_walkers, disable=not was_logging,
                        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                    "[Time: {elapsed}]"))

            # Keep searching parameter space until req_n_walkers are found
            while(n_walkers < req_n_walkers):
                # Calculate the covariance matrix of all walkers
                cov = np.cov(walkers[:n_walkers].T)

                # Create set of proposed walkers
                new_walkers = np.apply_along_axis(multivariate_normal, 1,
                                                  p0_walkers, cov)

                # Check which proposed walkers should be accepted
                accept = advance_chain(new_walkers)
                acc_walkers = new_walkers[accept]
                n_accepted = sum(accept)

                # Replace current walkers with accepted walkers
                p0_walkers[accept] = acc_walkers

                # Update final walkers array
                walkers[n_walkers:n_walkers+n_accepted] = acc_walkers
                n_walkers += n_accepted

                # Update progress bar
                pbar.update(min(req_n_walkers, n_walkers)-pbar.n)

            # Close progress bar
            pbar.close()

    # Turn logging back on if it used to be on
    pipe.do_logging = was_logging

    # Broadcast walkers to all workers
    walkers = pipe._comm.bcast(np.unique(walkers[:req_n_walkers], axis=0), 0)
    n_walkers = walkers.shape[0]

    # Return n_walkers, walkers
    return(n_walkers, walkers)
