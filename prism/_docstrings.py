# -*- coding: utf-8 -*-

"""
PRISM Docstrings
================
Contains a collection of docstrings that are reused throughout the
documentation of the various functions in the PRISM package.

"""

# %% CHECK DOCSTRINGS
check_bool_doc = \
    """Checks if provided argument `name` of `value` is a bool.
    Returns 0 or 1 if *True* and raises a :class:`~TypeError` if *False*."""

check_fin_doc = \
    """Checks if provided argument `name` of `value` is finite.
    Returns `value` if *True* and raises a :class:`~ValueError` if *False*."""

check_type_doc = \
    """Checks if provided argument `name` of `value` is %s.
    Returns `value` if *True* and raises a :class:`~TypeError` if *False*."""

check_val_doc = \
    """Checks if provided argument `name` of `value` is %s.
    Returns `value` if *True* and raises a :class:`~TypeError` or
    :class:`~ValueError` if *False*."""


# %% EMUL_I DOCSTRINGS
# Description of emul_i used in __call__/construct
call_emul_i_doc = \
     """emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the next iteration of the loaded emulator system will be
            constructed."""

# Description of emul_i used in the get_emul_i() method of the Emulator class
get_emul_i_doc = \
     """emul_i : int or None
            Number indicating the requested emulator iteration."""

# Description of emul_i used in basically all standard hidden functions
std_emul_i_doc = \
     """emul_i : int
            Number indicating the requested emulator iteration."""

# Description of emul_i used in all user functions except __call__/construct
user_emul_i_doc = \
     """emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the last iteration of the loaded emulator system will be
            used."""
