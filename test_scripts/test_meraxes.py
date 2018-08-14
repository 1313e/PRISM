#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:43:48 2018

@author: evandervelden
"""

import matplotlib as mpl
mpl.use('Agg')
from mpi4py import MPI
from prism import Pipeline
from modellink.meraxes_link import MeraxesLink
import sys

# Define the emulator type. Currently, only 'default' is compatible.
emul_type = 'default'

# Initialize MeraxesLink subclass
# It uses both model parameters and comparison data files
# It also takes an extra (mandatory) argument indicating the Meraxes input file
model_link = MeraxesLink(input_file='input_meraxes.par',
                         model_parameters='data/parameters_meraxes.txt',
                         model_data='data/data_meraxes.txt')

# For use on OzSTAR, allow for a working directory to be provided
try:
    work_dir = str(sys.argv[2])
except IndexError:
    work_dir = 'meraxes'

# Initialize the Pipeline class by giving it the ModelLink object
# The remaining optional arguments indicate several paths and non-default names
pipe = Pipeline(model_link, root_dir='tests',
                working_dir=work_dir,
                prism_file='data/prism_meraxes.txt',
                emul_type=emul_type)


# %% CONSTRUCT
# For use on OzSTAR, allow for an operation to be provided
try:
    op = str(sys.argv[1])
except IndexError:
    op = 'full'

# For use on OzSTAR, allow for an iteration to be provided
try:
    emul_i = int(sys.argv[3])
except IndexError:
    emul_i = 1

# Check which operation is requested and perform it
if(op == 'construct'):
    pipe.construct(emul_i, analyze=False)
elif(op == 'analyze'):
    pipe.analyze()
elif(op == 'projection'):
    pipe.project(emul_i)
elif(op == 'full'):
    pipe(emul_i)
