# -*- coding: utf-8 -*-

# Constant Magnetic Field ModelLink
# Compatible with Python 2.7

# %% IMPORTS
from __future__ import absolute_import, division, print_function

# ModelLink imports
from modellink import ModelLink

# Hammurapy imports
import numpy as np
import logging
from imagine.base_classes import observables_generator
from imagine.hammurapy.hammurapy import Hammurapy

logger = logging.getLogger('CMF-LINK')


# %% CLASS DEFINITION
class ConstantMagneticFieldLink(ModelLink):
    def __init__(self, *args, **kwargs):
        self.hammuc = Hammurapy_constant(
                350, 350, hammurabi_path='/home/evandervelden/'
                                         'imagine_downloads/hammurabi_3',
                conf_directory='/home/evandervelden/imagine_downloads/imagine/'
                               'imagine/hammurapy/confs')

        super(ConstantMagneticFieldLink, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = self.hammuc.constant_parameter_dict

        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        logger.info("Starting to observe parameters")

        # Create a temp working directory
        temp_folder = self.hammuc._make_temp_folder()
        logger.info('Using %s', temp_folder)

        # Create the parameter file in the temporary directory
        self.hammuc._make_parameter_file(temp_folder, model_parameters)
        logger.info('Stored parameter file')

        # Call Hammurabi
        logger.info('Calling Hammurabi')
        self.hammuc._call_hammurabi(temp_folder)
        logger.info('Hammurabi finished')

        # Read in the rotation measure map and build an observable out of it
        observables = self.hammuc._build_observables(temp_folder)

        # Delete the temporary folder
        self.hammuc._remove_folder(temp_folder)

        # Select the correct data points according to data_idx
        obs_data = []
        obs_data.extend(observables['rm_observable']['rm_map'][data_idx[:3]])
        obs_data.extend(
            observables['sync_observable']['sync_Q'][data_idx[3:6]])
        obs_data.extend(observables['sync_observable']['sync_U'][data_idx[6:]])

        # Return it
        return(obs_data)

    def get_md_var(self, emul_i, data_idx):
        super(ConstantMagneticFieldLink, self).get_md_var(emul_i=emul_i,
                                                          data_idx=data_idx)


class Hammurapy_constant(observables_generator, Hammurapy):
    def __init__(self, box_dimensions, resolution, *args, **kwargs):
        self.box_dimensions = np.empty(3)
        self.box_dimensions[:] = box_dimensions

        self.resolution = np.empty(3)
        self.resolution[:] = resolution

        super(Hammurapy_constant, self).__init__(*args, **kwargs)

        self.constant_parameter_dict = {'B_field_b0': [0, 10, 6],
                                        'B_x_const': [-1, 1, 0.7],
                                        'B_y_const': [-1, 1, 0.7],
                                        'B_z_const': [-1, 1, 0]}

    def get_default_parameters_dict(self):
        result_dict = super(Hammurapy_constant,
                            self).get_default_parameters_dict()
        result_dict['B_field_type'] = '11'
        result_dict['obs_shell_index_numb'] = '1'
        result_dict['total_shell_numb'] = '1'
        result_dict['obs_NSIDE'] = '16'
        result_dict['TE_interp'] = 'T'
        result_dict['B_field_do_random'] = 'F'
        result_dict['do_rm'] = 'T'
        result_dict['do_sync_emission'] = 'T'
        result_dict['do_dm'] = 'F'
        result_dict['do_dust'] = 'F'
        result_dict['do_tau'] = 'F'
        result_dict['do_ff'] = 'F'
        result_dict['B_field_interp'] = 'T'
        result_dict['use_B_analytic'] = 'F'
        return(result_dict)

    def _make_parameter_file(self, working_directory, constant_parameter_dict):
        dimensions = self.box_dimensions
        resolution = self.resolution

        super(Hammurapy_constant, self)._make_parameter_file(
                working_directory,
                resolution,
                dimensions,
                custom_parameters=constant_parameter_dict)
