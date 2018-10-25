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
import sys

# Package imports
from e13tools.core import ShapeError
import numpy as np

# PRISM imports
from .._docstrings import user_emul_i_doc
from .._internal import (RequestError, check_vals, docstring_substitute,
                         exec_code_anal)
from ..pipeline import Pipeline

# All declaration
__all__ = ['get_walkers', 'lnpost']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% FUNCTION DEFINITIONS
# This function returns a set of valid MCMC walkers
@docstring_substitute(emul_i=user_emul_i_doc)
def get_walkers(pipeline_obj, emul_i=None, init_walkers=None, unit_space=True,
                lnpost_fn=False):
    """
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
    init_walkers : 2D array_like or None. Default: None
        Sample set of proposed initial MCMC walker positions. All plausible
        samples in `init_walkers` will be returned.
        If *None*, return :attr:`~prism.pipeline.Pipeline.impl_sam
        corresponding to iteration `emul_i` instead.
    unit_space : bool. Default: True
        Bool determining whether or not the provided samples and returned
        walkers are given in unit space.
    lnpost_fn : bool. Default: False
        If *True*, a specialized version of the :func:`~lnpost` function is
        returned as well.

    Returns
    -------
    n_walkers : int
        Number of returned MCMC walkers.
    p0_walkers : 2D :obj:`~numpy.ndarray` object
        Array containing starting positions of valid MCMC walkers.
    lnpost_s (if `lnpost_fn` is *True*) : :func:`~lnpost`
        A specialized version of the :func:`~lnpost` function that by default
        uses the same values for `pipeline_obj`, `emul_i` and `unit_space` as
        provided to this function.

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

    # Check if lnpost_fn is a bool
    lnpost_fn = check_vals(lnpost_fn, 'lnpost_fn', 'bool')

    # If init_walkers is None, use impl_sam of emul_i
    if init_walkers is None:
        # Controller checking if emul_i has already been analyzed
        if pipe._is_controller:
            # If iteration has not been analyzed, raise error
            if not pipe._n_eval_sam[emul_i]:
                raise RequestError("Emulator iteration %s has not been "
                                   "analyzed yet!" % (emul_i))
            # If iteration is last iteration, init_walkers is current impl_sam
            elif(emul_i == pipe._emulator._emul_i):
                init_walkers = pipe._impl_sam
            # If iteration is not last, init_walkers is previous impl_sam
            else:
                init_walkers = pipe._emulator._sam_set[emul_i+1]

        # Broadcast init_walkers to workers as p0_walkers
        p0_walkers = pipe._comm.bcast(init_walkers, 0)

    # If init_walkers is not None, use provided samples
    else:
        # Controller checking if init_walkers is valid
        if pipe._is_controller:
            # Make sure that init_walkers is a NumPy array
            init_walkers = np.array(init_walkers, ndmin=2)

            # Check if init_walkers is two-dimensional
            if not(init_walkers.ndim == 2):
                raise ShapeError("Input argument 'init_walkers' has more than "
                                 "two dimensions (%s)!" % (init_walkers.ndim))

            # Check if init_walkers has correct shape
            if not(init_walkers.shape[1] == pipe._modellink._n_par):
                raise ShapeError("Input argument 'init_walkers' has incorrect "
                                 "number of parameters (%s != %s)!"
                                 % (init_walkers.shape[1],
                                    pipe._modellink._n_par))

            # Check if init_walkers solely contains floats
            init_walkers = check_vals(init_walkers, 'init_walkers', 'float')

            # If unit_space is True, convert init_walkers to par_space
            if unit_space:
                init_walkers = pipe._modellink._to_par_space(init_walkers)

        # Broadcast init_walkers to workers
        init_walkers = pipe._comm.bcast(init_walkers, 0)

        # Analyze init_walkers and save them as p0_walkers
        p0_walkers = pipe._analyze_sam_set(emul_i, init_walkers,
                                           *exec_code_anal)

    # Check if p0_walkers needs to be converted
    if unit_space:
        p0_walkers = pipe._modellink._to_unit_space(p0_walkers)

    # Check if lnpost_fn was requested and return it as well if so
    if lnpost_fn:
        # Define specialized version of lnpost
        def lnpost_s(par_set, ext_lnpost, pipeline_obj=pipe, emul_i=emul_i,
                     unit_space=unit_space, **kwargs):
            """
            Calculates the natural logarithm of the posterior probability of
            `par_set` using the provided function `ext_lnpost`, in addition to
            constraining it first with the emulator defined in the
            `pipeline_obj`.

            This is a specialized version of :func:`~lnpost`, where
            `pipeline_obj`, `emul_i` and `unit_space` are defaulted to the
            values used in the call of :func:`~get_walkers` that created this
            function. See the docs of :func:`~lnpost` for a more extensive
            description.

            """

            # Call lnpost using specific default values
            return(lnpost(par_set, ext_lnpost, pipeline_obj, emul_i,
                          unit_space, **kwargs))

        return(len(p0_walkers), p0_walkers, lnpost_s)
    else:
        return(len(p0_walkers), p0_walkers)


# This function constrains the default lnpost function by PRISM
@docstring_substitute(emul_i=user_emul_i_doc)
def lnpost(par_set, ext_lnpost, pipeline_obj, emul_i=None, unit_space=True,
           **kwargs):
    """
    Calculates the natural logarithm of the posterior probability of `par_set`
    using the provided function `ext_lnpost`, in addition to constraining it
    first with the emulator defined in the `pipeline_obj`.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    par_set : 1D array_like
        Sample to calculate the posterior probability for. This sample is first
        analyzed in `pipeline_obj` and only given to `ext_lnpost` if it is
        plausible.
    ext_lnpost : function
        Function definition that needs to be called if the provided `par_set`
        is plausible in iteration `emul_i` of `pipeline_obj`. The used call
        signature is ``ext_lnpost(par_set, **kwargs)``. All MPI ranks will call
        this function.
    pipeline_obj : :obj:`~prism.pipeline.Pipeline` object
        The instance of the :class:`~prism.pipeline.Pipeline` class that needs
        to be used for determining the validity of the proposed sampling step.

    Optional
    --------
    %(emul_i)s
    unit_space : bool. Default: True
        Bool determining whether or not the provided sample is given in unit
        space.
    kwargs : dict
        Dict of keyword arguments that needs to be passed to the `ext_lnpost`
        function.

    Returns
    -------
    lnp : float
        The natural logarithm of the posterior probability of `par_set`, as
        determined by the emulator in `pipeline_obj` and the `ext_lnpost`
        function.

    Warning
    -------
    This function does not have any checks in place to see if the provided
    input arguments are valid, as these checks would be performed many times
    unnecessarily. Therefore, raised exceptions due to invalid input arguments
    may be harder to track down.

    """

    # Make abbreviation for pipeline_obj
    pipe = pipeline_obj

    # Get emulator iteration
    if emul_i is None:
        emul_i = pipe._emulator._emul_i

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
    impl_sam = pipe._analyze_sam_set(emul_i, np.array(sam, ndmin=2),
                                     *exec_code_anal)

    # If par_set is plausible, call ext_lnpost
    if impl_sam.size:
        return(ext_lnpost(par_set, **kwargs))

    # If par_set is not plausible, return -inf
    else:
        return(-np.infty)
