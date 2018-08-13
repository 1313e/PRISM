#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:43:48 2018

@author: evandervelden
"""

import matplotlib as mpl
mpl.use('Agg')
from prism import Pipeline
from modellink.imagine_link import ImagineLink
import sys

emul_type = 'default'
hammu_model = str(sys.argv[1])
model_link = ImagineLink(hammu_model,
                         model_data='data/data_imagine_%s.txt'
                                    % (str(sys.argv[2])))

pipe = Pipeline(model_link, root_dir='tests',
                working_dir=str(sys.argv[3]),
                prism_file='data/prism_%s.txt' % (hammu_model),
                emul_type=emul_type)


# %% CONSTRUCT
pipe.construct(1)
pipe.project(1)

