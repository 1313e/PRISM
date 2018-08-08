#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:43:48 2018

@author: evandervelden
"""

import matplotlib as mpl
mpl.use('Agg')
from mpi4py import MPI
from prism import Pipeline
from prism.modellink import SineWaveLink

# Define the emulator type. Currently, only 'default' is compatible.
emul_type = 'default'

# Initialize the SineWaveLink subclass
# It takes optionally model parameters and comparison data
# The parameters are hard-coded in the model, so only data is provided
model_link = SineWaveLink(model_data='data/data_sine_wave.txt')

# Initialize the Pipeline class by giving it the ModelLink object
# The remaining optional arguments indicate several paths and non-default names
pipe = Pipeline(model_link, root_dir='tests',
                working_dir='sine_wave',
                prism_file='data/prism_sine.txt',
                emul_type=emul_type)


# %% CONSTRUCT
# Creating the emulator is as easy as calling the following
# This constructs, analyzes and projects the next iteration of the emulator
#pipe()

# If a projection is not required, then the emulator can be constructed and
# analyzed with
pipe.construct()
