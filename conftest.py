# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
import sys

# Package imports
import matplotlib as mpl
from mpi4pyd import MPI
from py.path import local
import pytest


# Set MPL backend
mpl.use('Agg')


# %% PYTEST CUSTOM CONFIGURATION PLUGINS
# This makes the pytest report header mention the tested PRISM version
def pytest_report_header(config):
    from prism.__version__ import __version__
    return("PRISM: %s" % (__version__))


# Disable xvfb on all cores except the first
def pytest_load_initial_conftests(args):
    if MPI.COMM_WORLD.Get_rank() and sys.platform.startswith('linux'):
        args = ["--no-xvfb"] + args


# Add an attribute to PRISM stating that pytest is being used
# Also add the pep8 and incremental markers
def pytest_configure(config):
    import prism
    prism.__PYTEST = True

    config.addinivalue_line("markers", "pep8: Checks for PEP8 compliancy.")
    config.addinivalue_line("markers",
                            "incremental: Mark test suite to xfail all "
                            "remaining tests when one fails.")


# After pytest has finished, remove this attribute again
def pytest_unconfigure(config):
    import prism
    del prism.__PYTEST


# This introduces a marker that auto-fails tests if a previous one failed
def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


# This makes every marked test auto-fail if a previous one failed as well
def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("Previous test failed (%s)" % (previousfailed.name))


# %% PYTEST SETTINGS
# Set the current working directory to the temporary directory
local.get_temproot().chdir()
