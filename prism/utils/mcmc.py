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

# PRISM imports
from prism._docstrings import user_emul_i_doc
from prism._internal import RequestError, check_vals, np_array
from prism._pipeline import Pipeline

# All declaration
__all__ = ['get_lnpost_fn', 'get_walkers']


# %% FUNCTION DEFINITIONS
# This function returns a specialized version of the lnpost function
@docstring_substitute(emul_i=user_emul_i_doc)
def get_lnpost_fn(ext_lnpost, pipeline_obj, *, emul_i=None, unit_space=True,
                  hybrid=True):
    """
    Returns a function definition ``get_lnpost(par_set, *args, **kwargs)``.

    This `get_lnpost` function can be used to calculate the natural logarithm
    of the posterior probability, which analyzes a given `par_set` first in the
    provided `pipeline_obj` at iteration `emul_i` and passes it to the
    `ext_lnpost` function if it is plausible.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    ext_lnpost : function
        Function definition that needs to be called if the provided `par_set`
        is plausible in iteration `emul_i` of `pipeline_obj`. The used call
        signature is ``ext_lnpost(par_set, *args, **kwargs)``. All MPI ranks
        will call this function unless called within the
        :attr:`~prism.Pipeline.worker_mode` context manager.
    pipeline_obj : :obj:`~prism.Pipeline` object
        The instance of the :class:`~prism.Pipeline` class that needs
        to be used for determining the validity of the proposed sampling step.

    Optional
    --------
    %(emul_i)s
    unit_space : bool. Default: True
        Bool determining whether or not the provided sample will be given in
        unit space.
    hybrid : bool. Default: True
        Bool determining whether or not the `get_lnpost` function should
        use the implausibility values of a given `par_set` as an additional
        prior.

    Returns
    -------
    get_lnpost : function
        Definition of the function ``get_lnpost(par_set, *args, **kwargs)``.


    See also
    --------
    :func:`~get_walkers`
        Analyzes proposed `init_walkers` and returns valid `p0_walkers`.

    :attr:`~prism.Pipeline.worker_mode`
        Special context manager within which all code is executed in worker
        mode.

    Warning
    -------
    Calling this function factory will disable all regular logging in
    `pipeline_obj` (:attr:`~prism.Pipeline.do_logging` set to
    *False*), in order to avoid having the same message being logged every time
    `get_lnpost` is called.

    """

    # Check if ext_lnpost is a function
    if not isfunction(ext_lnpost):
        raise InputError("Input argument 'ext_lnpost' is not a callable "
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
    emul_i = pipe._emulator._get_emul_i(emul_i, True)

    # Check if unit_space is a bool
    unit_space = check_vals(unit_space, 'unit_space', 'bool')

    # Check if hybrid is a bool
    hybrid = check_vals(hybrid, 'hybrid', 'bool')

    # Disable PRISM logging
    pipe.do_logging = False

    # Define get_lnpost function
    def get_lnpost(par_set, *args, **kwargs):
        """
        Calculates the natural logarithm of the posterior probability of
        `par_set` using the provided function `ext_lnpost`, in addition to
        constraining it first with the emulator defined in the `pipeline_obj`.

        This function needs to be called by all MPI ranks unless called within
        the :attr:`~prism.Pipeline.worker_mode` context manager.

        Parameters
        ----------
        par_set : 1D array_like
            Sample to calculate the posterior probability for. This sample is
            first analyzed in `pipeline_obj` and only given to `ext_lnpost` if
            it is plausible.
        args : tuple
            Positional arguments that need to be passed to the `ext_lnpost`
            function.
        kwargs : dict
            Keyword arguments that need to be passed to the `ext_lnpost`
            function.

        Returns
        -------
        lnp : float
            The natural logarithm of the posterior probability of `par_set`, as
            determined by the `ext_lnpost` function if `par_set` is plausible.
            If `hybrid` is *True*, `lnp` is calculated as `lnprior` +
            `ext_lnpost()`, with `lnprior` the natural logarithm of the first
            implausibility cut-off value of `par_set` scaled with its maximum.

        """

        # If unit_space is True, convert par_set to par_space
        if unit_space:
            sam = pipe._modellink._to_par_space(par_set)
        else:
            sam = par_set

        # Check if par_set is within parameter space and return -inf if not
        par_rng = pipe._modellink._par_rng
        if not ((par_rng[:, 0] <= sam)*(sam <= par_rng[:, 1])).all():
            return(-np.infty)

        # Check what sampling is requested and analyze par_set
        if hybrid:
            impl_sam, lnprior = pipe._make_call('_evaluate_sam_set', emul_i,
                                                np_array(sam, ndmin=2),
                                                'hybrid')
        else:
            impl_sam = pipe._make_call('_evaluate_sam_set', emul_i,
                                       np_array(sam, ndmin=2), 'analyze')
            lnprior = 0

        # If par_set is plausible, call ext_lnpost
        if len(impl_sam):
            return(lnprior+ext_lnpost(par_set, *args, **kwargs))

        # If par_set is not plausible, return -inf
        else:
            return(-np.infty)

    # Check if model in ModelLink can be single-called, raise warning if not
    if pipe._is_controller and not pipe._modellink._single_call:
        warn_msg = ("ModelLink bound to provided Pipeline object solely "
                    "requests multi-calls. Using MCMC may not be possible.")
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    # Return get_lnpost function definition
    return(get_lnpost)


# This function returns a set of valid MCMC walkers
@docstring_substitute(emul_i=user_emul_i_doc)
def get_walkers(pipeline_obj, *, emul_i=None, init_walkers=None,
                unit_space=True, ext_lnpost=None, **kwargs):
    """
    Analyzes proposed `init_walkers` and returns valid `p0_walkers`.

    Analyzes sample set `init_walkers` in the provided `pipeline_obj` at
    iteration `emul_i` and returns all samples that are plausible to be used as
    MCMC walkers. The provided samples and returned walkers should be/are given
    in unit space if `unit_space` is *True*.

    If `init_walkers` is *None*, returns
    :attr:`~prism.Pipeline.impl_sam` instead if it is available.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    pipeline_obj : :obj:`~prism.Pipeline` object
        The instance of the :class:`~prism.Pipeline` class that needs
        to be used for determining the validity of the proposed walkers.

    Optional
    --------
    %(emul_i)s
    init_walkers : 2D array_like, int or None. Default: None
        Sample set of proposed initial MCMC walker positions. All plausible
        samples in `init_walkers` will be returned.
        If int, generate an LHD of provided size and return all plausible
        samples.
        If *None*, return :attr:`~prism.Pipeline.impl_sam`
        corresponding to iteration `emul_i` instead.
    unit_space : bool. Default: True
        Bool determining whether or not the provided samples and returned
        walkers are given in unit space.
    ext_lnpost : function or None. Default: None
        If function, call :func:`~get_lnpost_fn` function factory using
        `ext_lnpost` and the same values for `pipeline_obj`, `emul_i` and
        `unit_space`, and return the resulting function definition
        `get_lnpost`. Any additionally provided `kwargs` are also passed to it.

    Returns
    -------
    n_walkers : int
        Number of returned MCMC walkers.
    p0_walkers : 2D :obj:`~numpy.ndarray` object
        Array containing starting positions of valid MCMC walkers.
    get_lnpost : function (if `ext_lnpost` is a function)
        The function returned by :func:`~get_lnpost_fn` function factory using
        `ext_lnpost`, `pipeline_obj`, `emul_i`, `unit_space` and `kwargs` as
        the input values.

    See also
    --------
    :func:`~get_lnpost_fn`
        Returns a function definition ``get_lnpost(par_set, *args, **kwargs)``.

    :attr:`~prism.Pipeline.worker_mode`
        Special context manager within which all code is executed in worker
        mode.

    Notes
    -----
    If `init_walkers` is *None* and emulator iteration `emul_i` has not been
    analyzed yet, a :class:`~prism._internal.RequestError` will be raised.

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
    emul_i = pipe._emulator._get_emul_i(emul_i, True)

    # Check if unit_space is a bool
    unit_space = check_vals(unit_space, 'unit_space', 'bool')

    # Check if ext_lnpost is None and try to obtain lnpost function if not
    if ext_lnpost is not None:
        try:
            lnpost_fn = get_lnpost_fn(ext_lnpost, pipe, emul_i=emul_i,
                                      unit_space=unit_space, **kwargs)
        except InputError:
            raise InputError("Input argument 'ext_lnpost' is invalid!")

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

            # If init_walkers is not an int, it must be array_like
            else:
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

    # Calculate n_walkers
    n_walkers = len(p0_walkers)

    # Check if p0_walkers is not empty
    if not n_walkers:
        raise InputError("Input argument 'init_walkers' contains no plausible "
                         "samples!")

    # Check if p0_walkers needs to be converted
    if unit_space:
        p0_walkers = pipe._modellink._to_unit_space(p0_walkers)

    # Check if lnpost_fn was requested and return it as well if so
    if ext_lnpost is not None:
        return(n_walkers, p0_walkers, lnpost_fn)
    else:
        return(n_walkers, p0_walkers)
