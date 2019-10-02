# -*- coding: utf-8 -*-

# %% IMPORTS
# Package imports
import pytest

# PRISM imports
from prism.__version__ import __version__


# %% PYTEST CUSTOM CONFIGURATION PLUGINS
# This makes the pytest report header mention the tested PRISM version
def pytest_report_header(config):
    return("PRISM: v%s" % (__version__))


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

