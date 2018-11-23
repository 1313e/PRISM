# -*- coding: utf-8 -*-

"""
MCMC
====
Provides several functions that allow for *PRISM* to be connected easily to
MCMC routines.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from inspect import isfunction
import sys
import warnings

# Package imports
from e13tools import InputError
from e13tools.sampling import lhd
import numpy as np

# PRISM imports
from .._docstrings import user_emul_i_doc
from .._internal import (RequestError, check_vals, docstring_substitute)
from ..pipeline import Pipeline

# All declaration
__all__ = ['get_lnpost_fn', 'get_walkers']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% FUNCTION DEFINITIONS
# This function returns a specialized version of the lnpost function
@docstring_substitute(emul_i=user_emul_i_doc)
def get_lnpost_fn(ext_lnpost, pipeline_obj, emul_i=None, unit_space=True):
    """
    Returns a function definition ``lnpost(par_set, *args, **kwargs)``.

    This ``lnpost`` function can be used to calculate the natural logarithm of
    the posterior probability, which analyzes a given `par_set` first in the
    provided `pipeline_obj` at iteration `emul_i` and passes it to the
    `ext_lnpost` function if it is plausible.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    ext_lnpost : function
        Function definition that needs to be called if the provided `par_set`
        is plausible in iteration `emul_i` of `pipeline_obj`. The used call
        signature is ``ext_lnpost(par_set, *args, **kwargs)``. All MPI ranks
        will call this function unless
        :attr:`~prism.pipeline.Pipeline.is_listening` is *True*.
    pipeline_obj : :obj:`~prism.pipeline.Pipeline` object
        The instance of the :class:`~prism.pipeline.Pipeline` class that needs
        to be used for determining the validity of the proposed sampling step.

    Optional
    --------
    %(emul_i)s
    unit_space : bool. Default: True
        Bool determining whether or not the provided sample will be given in
        unit space.

    Returns
    -------
    lnpost : function
        Definition of the function ``lnpost(par_set, *args, **kwargs)``.


    See also
    --------
    :func:`~get_walkers`: Analyzes proposed `init_walkers` and returns valid \
        `p0_walkers`.

    Warning
    -------
    Calling this function factory will disable all regular logging in
    `pipeline_obj` (:attr:`~prism.pipeline.Pipeline.do_logging` set to
    *False*), in order to avoid having the same message being logged every time
    `lnpost` is called.

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

    # Get emulator iteration
    emul_i = pipe._emulator._get_emul_i(emul_i, True)

    # Check if unit_space is a bool
    unit_space = check_vals(unit_space, 'unit_space', 'bool')

    # Disable PRISM logging
    pipe.do_logging = False

    # Define lnpost function
    def lnpost(par_set, *args, **kwargs):
        """
        Calculates the natural logarithm of the posterior probability of
        `par_set` using the provided function `ext_lnpost`, in addition to
        constraining it first with the emulator defined in the `pipeline_obj`.

        This function needs to be called by all MPI ranks if
        :attr:`~prism.pipeline.Pipeline.is_listening` is *False*.

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
            determined by the emulator in `pipeline_obj` and the `ext_lnpost`
            function.

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

        # Analyze par_set
        impl_sam = pipe._make_call('_get_impl_sam', emul_i,
                                   np.array(sam, ndmin=2))

        # If par_set is plausible, call ext_lnpost
        if impl_sam.size:
            return(ext_lnpost(par_set, *args, **kwargs))

        # If par_set is not plausible, return -inf
        else:
            return(-np.infty)

    # Check if model in ModelLink can be single-called, raise warning if not
    if pipe._is_controller and not pipe._modellink._single_call:
        warn_msg = ("ModelLink bound to provided Pipeline object solely "
                    "requests multi-calls. Using MCMC may not be possible.")
        warnings.warn(warn_msg, stacklevel=2)

    # Return lnpost function definition
    return(lnpost)


# This function returns a set of valid MCMC walkers
@docstring_substitute(emul_i=user_emul_i_doc)
def get_walkers(pipeline_obj, emul_i=None, init_walkers=None, unit_space=True,
                lnpost_fn=None):
    """
    Analyzes proposed `init_walkers` and returns valid `p0_walkers`.

    Analyzes sample set `init_walkers` in the provided `pipeline_obj` at
    iteration `emul_i` and returns all samples that are plausible to be used as
    MCMC walkers. The provided samples and returned walkers should be/are given
    in unit space if `unit_space` is *True*.

    If `init_walkers` is *None*, returns
    :attr:`~prism.pipeline.Pipeline.impl_sam` instead if it is available.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    pipeline_obj : :obj:`~prism.pipeline.Pipeline` object
        The instance of the :class:`~prism.pipeline.Pipeline` class that needs
        to be used for determining the validity of the proposed walkers.

    Optional
    --------
    %(emul_i)s
    init_walkers : 2D array_like, int or None. Default: None
        Sample set of proposed initial MCMC walker positions. All plausible
        samples in `init_walkers` will be returned.
        If int, generate an LHD of provided size and return all plausible
        samples.
        If *None*, return :attr:`~prism.pipeline.Pipeline.impl_sam`
        corresponding to iteration `emul_i` instead.
    unit_space : bool. Default: True
        Bool determining whether or not the provided samples and returned
        walkers are given in unit space.
    lnpost_fn : function or None. Default: None
        If function, call :func:`~get_lnpost_fn` function factory using
        `lnpost_fn` as the `ext_lnpost` input argument and the same values for
        `pipeline_obj`, `emul_i` and `unit_space`, and return the resulting
        function definition ``lnpost``.

    Returns
    -------
    n_walkers : int
        Number of returned MCMC walkers.
    p0_walkers : 2D :obj:`~numpy.ndarray` object
        Array containing starting positions of valid MCMC walkers.
    lnpost : function (if `lnpost_fn` is a function)
        The function returned by :func:`~get_lnpost_fn` function factory using
        `lnpost_fn`, `pipeline_obj`, `emul_i` and `unit_space` as the input
        values.

    See also
    --------
    :func:`~get_lnpost_fn`: Returns a function definition \
        ``lnpost(par_set, *args, **kwargs)``.

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

    # Get emulator iteration
    emul_i = pipe._emulator._get_emul_i(emul_i, True)

    # Check if unit_space is a bool
    unit_space = check_vals(unit_space, 'unit_space', 'bool')

    # Check if lnpost_fn is a function or None
    if not isfunction(lnpost_fn) and lnpost_fn is not None:
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

        # Broadcast init_walkers to workers as p0_walkers
        p0_walkers = pipe._comm.bcast(init_walkers, 0)

    # If init_walkers is not None, use provided samples or LHD size
    else:
        # Controller checking if init_walkers is valid
        if pipe._is_controller:
            # If init_walkers is an int, create LHD of provided size
            if isinstance(init_walkers, (int, np.integer)):
                # Check if provided integer is positive
                n_sam = check_vals(init_walkers, 'init_walkers', 'pos')

                # Create LHD of provided size
                init_walkers = lhd(n_sam, pipe._modellink._n_par,
                                   pipe._modellink._par_rng, 'center',
                                   pipe._criterion, 100)

            # If init_walkers is not an int, it must be array_like
            else:
                # Make sure that init_walkers is a NumPy array
                init_walkers = np.array(init_walkers, ndmin=2)

                # If unit_space is True, convert init_walkers to par_space
                if unit_space:
                    init_walkers = pipe._modellink._to_par_space(init_walkers)

                # Check if init_walkers is valid
                init_walkers = pipe._modellink._check_sam_set(init_walkers,
                                                              'init_walkers')

        # Broadcast init_walkers to workers
        init_walkers = pipe._comm.bcast(init_walkers, 0)

        # Analyze init_walkers and save them as p0_walkers
        p0_walkers = pipe._get_impl_sam(emul_i, init_walkers)

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
    if lnpost_fn is not None:
        return(n_walkers, p0_walkers,
               get_lnpost_fn(lnpost_fn, pipe, emul_i, unit_space))
    else:
        return(n_walkers, p0_walkers)