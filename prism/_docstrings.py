# -*- coding: utf-8 -*-

"""
PRISM Docstrings
================
Contains a collection of docstrings that are reused throughout the
documentation of the various functions in the PRISM package.

"""

# Description of emul_i used in the get_emul_i() method of the Emulator class
get_emul_i_doc = \
     """emul_i : int or None
            Number indicating the requested emulator iteration."""

# Description of emul_i used in basically all standard hidden functions
std_emul_i_doc = \
     """emul_i : int
            Number indicating the requested emulator iteration."""

# Description of emul_i used in all user functions
user_emul_i_doc = \
     """emul_i : int or None. Default: None
            If int, number indicating the requested emulator iteration.
            If *None*, the last iteration of the loaded emulator system will be
            used."""