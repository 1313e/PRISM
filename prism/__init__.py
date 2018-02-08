# -*- coding: utf-8 -*-

"""
PRISM
========
Probabilistic Regression Instrument for Simulating Models created by **1313e**.

"""


# %% IMPORTS AND DECLARATIONS
from __future__ import absolute_import, division, print_function

# Import package modules
from .version import version as __version__
from . import modellink
from .pipeline import Pipeline

__all__ = ['modellink', 'Pipeline']
