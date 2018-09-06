#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:43:48 2018

@author: evandervelden
"""

import matplotlib as mpl
mpl.use('Agg')
from prism import Pipeline
from prism.modellink import GaussianLink
from prism._internal import rprint
from prism.projection import Projection
import numpy as np

# Define the emulator type. Currently, only 'default' is compatible.
emul_type = 'default'

# Initialize the GaussianLink subclass
# It takes optionally model parameters and comparison data
# The parameters are hard-coded in the model, so only data is provided
model_link = GaussianLink(model_parameters='data/parameters_gaussian.txt',
                          model_data='data/data_gaussian.txt')

# Initialize the Pipeline class by giving it the ModelLink object
# The remaining optional arguments indicate several paths and non-default names
pipe = Pipeline(model_link, root_dir='tests',
                working_dir='gaussian',
                prism_file='data/prism_gaussian.txt',
                emul_type=emul_type)


# %% CONSTRUCT
#pipe.project(1, (0, 1), 0, 0, 0)
#np.random.seed(0)
#pipe.analyze()

pipe.construct()

#rprint(pipe._emulator._evaluate(1, pipe._emulator._active_emul_s[1], np.array([4, 4, 4, 4])))
# Creating the emulator is as easy as calling the following
# This constructs, analyzes and projects the next iteration of the emulator
#pipe()

# If a projection is not required, then the emulator can be constructed and
# analyzed with
#pipe.construct(1, 1, None, 1)
#np.random.seed(0)
#pipe.analyze()
