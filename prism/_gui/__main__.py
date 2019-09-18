# TODO: Remove this once finished
import os
from prism._pipeline import Pipeline
from prism.modellink import PolyLink

os.chdir("../../../PRISM_Root")

data_dict = {
    -1: [2.5, 0.1],     # f(-1) = 2.5 +- 0.1
    0: [3, 0.1],        # f(0) = 3 +- 0.1
    2: [10, 0.1]}       # f(2) = 10 +- 0.1
modellink_obj = PolyLink(model_data=data_dict)
pipe = Pipeline(modellink_obj, root_dir='tests',
                working_dir='projection_gui',
                prism_par={'use_mock': True})

pipe.construct(1)
pipe.proj_res = 15
pipe.proj_depth = 75

pipe.open_gui()
