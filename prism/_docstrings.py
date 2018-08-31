# -*- coding: utf-8 -*-

"""
PRISM Docstrings
================
Contains a collection of docstrings that are reused throughout the
documentation of the various functions in the *PRISM* package.

"""

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

# %% EMUL_S DOCSTRINGS
# Description of sequence of emul_s used in all standard hidden functions
emul_s_seq_doc = \
    """emul_s_seq : list of int
           List of numbers indicating the requested emulator systems."""

# Description of emul_s used in basically all standard hidden functions
lemul_s_doc = \
    """lemul_s : int or None
           Number indicating the requested local emulator system.
           If *None*, use the master emulator file instead."""

# Description of sequence of emul_s used in all standard user functions
user_emul_s_doc = \
    """emul_s : int, list of int or None. Default: None
           Number of list of numbers indicating the requested emulator systems.
           If *None*, all emulator systems in the requested emulator iteration
           are used."""
