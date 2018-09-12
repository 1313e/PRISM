# -*- coding: utf-8 -*-

# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
import logging
from os import path
import sys

# Package imports
from e13tools.core import InputError, ShapeError
import numpy as np
import pytest
import pytest_mpl

# PRISM imports
from .modellink.test_gaussian_link import GaussianLink2D, GaussianLink3D
from prism._internal import RequestError
from prism.emulator import Emulator
from prism.pipeline import Pipeline

# Save the path to this directory
dirpath = path.dirname(__file__)


# %% PYTEST CLASSES AND FUNCTIONS
# Custom invalid Emulator class
class InvalidEmulator(Emulator):
    def __init__(self, *args, **kwargs):
        pass


# Pytest for Pipeline class (+Emulator, +Projection)
# TODO: See if it is possible to run some methods in parallel
class Test_Pipeline(object):
    # Save paths to various files
    model_data = path.join(dirpath, 'data/data_gaussian.txt')
    prism_file = path.join(dirpath, 'data/prism.txt')
    model_parameters_2D = path.join(dirpath, 'data/parameters_gaussian_2D.txt')
    model_parameters_3D = path.join(dirpath, 'data/parameters_gaussian_3D.txt')

    # Get a list of all Pipeline properties
    pipeline_props = [prop for prop in dir(Pipeline) if
                      isinstance(getattr(Pipeline, prop), property)]

    # Get a list of all Emulator properties
    emulator_props = [prop for prop in dir(Emulator) if
                      isinstance(getattr(Emulator, prop), property)]

    # Test a 2D Gaussian model
    def test_gaussian_2D(self, tmpdir):
        # Obtain paths
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a GaussianLink2D object
        model_link = GaussianLink2D(model_data=self.model_data)

        # Create a Pipeline object
        pipe = Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=self.prism_file, emul_type='default')

        # Set MPL backend temporarily to Agg
        with pytest_mpl.plugin.switch_backend('Agg'):
            # Check if first iteration can be constructed
            pipe.construct(1, 0)

            # Check if first iteration can be analyzed
            pipe.analyze()

            # Check if first iteration can be evaluated
            pipe.evaluate([1.5, 1.5])

            # Check if first iteration can be projected
            pipe.project()

            # Check if details overview of first iteration can be given
            pipe.details()

            # Check if entire second iteration can be created
            pipe.run(2)

        # Try to access all Pipeline properties
        for prop in self.pipeline_props:
            getattr(pipe, prop)

        # Try to access all Emulator properties
        for prop in self.emulator_props:
            getattr(pipe._emulator, prop)

    # Test a 3D Gaussian model
    def test_gaussian_3D(self, tmpdir):
        # Obtain paths
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a GaussianLink3D object
        model_link = GaussianLink3D(model_data=self.model_data)

        # Create a Pipeline object
        pipe = Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=self.prism_file, emul_type='default')

        # Set MPL backend temporarily to Agg
        with pytest_mpl.plugin.switch_backend('Agg'):
            # Check if first iteration can be constructed
            pipe.construct(1, 0)

            # Check if first iteration can be analyzed
            pipe.analyze()

            # Check if first iteration can be evaluated
            pipe.evaluate([1.5, 1.5, 1.5])

            # Check if first iteration can be projected
#            pipe.project()

            # Check if details overview of first iteration can be given
            pipe.details()

            # Check if entire second iteration can be created
#            pipe.run(2)

        # Try to access all Pipeline properties
#        for prop in self.pipeline_props:
#            getattr(pipe, prop)

        # Try to access all Emulator properties
#        for prop in self.emulator_props:
#            getattr(pipe._emulator, prop)

    # Test the many versatility functions and checks in PRISM
    def test_pipeline(self, tmpdir):
        # Obtain paths
        root_dir = path.dirname(tmpdir.strpath)
        working_dir = path.basename(tmpdir.strpath)

        # Create a GaussianLink2D object
        model_link = GaussianLink2D(model_data=self.model_data)

        # Create a Pipeline object using a custom Emulator class
        pipe = Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                        prism_file=self.prism_file, emul_type=Emulator)

        # Create a Pipeline object using an invalid Emulator class
        with pytest.raises(InputError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=self.prism_file, emul_type=InvalidEmulator)

        # Create a Pipeline object using not an Emulator class
        with pytest.raises(RequestError):
            Pipeline(model_link, root_dir=root_dir, working_dir=working_dir,
                     prism_file=self.prism_file, emul_type=Pipeline)

        # Change the modellink in pipe to use multi_call
        pipe._modellink.multi_call = True
        assert pipe._modellink._multi_call

        # Do a construction using multi_called model
        pipe.construct(1, 0)
