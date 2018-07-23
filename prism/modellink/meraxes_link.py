# -*- coding: utf-8 -*-

# Meraxes ModelLink
# Compatible with Python 3.6+ (as long as mhysa is wrapped)

"""
MeraxesLink
===========
Provides the definition of the :class:`~MeraxesLink` class.


Available classes
-----------------
:class:`~MeraxesLink`
    :class:`~ModelLink` class wrapper for the semi-analytic galaxy evolution
    model *Meraxes*.

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from os import path

# Package imports
import logging
from mhysa import Mhysa
import numpy as np

# PRISM imports
from prism import ModelLink

# All declaration
__all__ = ['MeraxesLink']

logger = logging.getLogger('MERAXES')


# %% MERAXESLINK CLASS DEFINITION
class MeraxesLink(ModelLink):
    """
    :class:`~ModelLink` class wrapper for the semi-analytic galaxy evolution
    model *Meraxes*.

    """

    def __init__(self, input_file, *args, **kwargs):
        """
        Initialize an instance of the :class:`~MeraxesLink` subclass.

        Parameters
        ----------
        input_file : str
            The name of the file that contains the input parameters for
            Meraxes.

        Optional
        --------
        args : arguments
            Arguments to be provided to the :class:`~ModelLink` class.
        kwargs : keyword arguments
            Keyword arguments to be provided to the :class:`~ModelLink` class.

        """

        # Save Meraxes input file
        self._input_file = path.abspath(input_file)

        # Request multi model calls
        self.multi_call = True

        # Request MPI calls
        self.MPI_call = True

        # Inheriting ModelLink __init__()
        super(MeraxesLink, self).__init__(*args, **kwargs)

    @property
    def _default_model_parameters(self):
        par_dict = {'SfEfficiency': [0, 1, 0.05]}
        return(par_dict)

    def call_model(self, emul_i, model_parameters, data_idx):
        # Initialize PrismMhysa object
        with PrismMhysa(n_comm=1, input_file=self._input_file,
                        data_idx=data_idx) as mhysa:
            for name, est in zip(self._par_name, self._par_est):
                mhysa.add_parameter(name, est, None)

        # Get sam_set
        if mhysa.is_controller:
            sam_set = map(lambda *args: args, *model_parameters.values())
        else:
            sam_set = None

        # Evaluate Meraxes for the entire sam_set
        mhysa.multiple_runs(sam_set)

        # Finalize PrismMhysa
        mhysa.finish()

        # Return it
        if mhysa.is_controller:
            return(mhysa._data_list)

    def get_md_var(self, emul_i, data_idx):
        super(MeraxesLink, self).get_md_var(emul_i=emul_i, data_idx=data_idx)


# %% MERAXESLINK CLASS PROPERTIES
    @property
    def input_file(self):
        """
        String containing the absolute path to the input parameter file used
        for Meraxes.

        """

        return(self._input_file)


# %% PRISMMHYSA CLASS DEFINITION
class PrismMhysa(Mhysa):
    """
    A :class:`~Mhysa` subclass manipulating the outputs of Meraxes to be used
    by the :class:`~prism.Pipeline` class.

    """

    def __init__(self, data_idx, *args, **kwargs):
        """
        Initialize an instance of the :class:`~PrismMhysa` class.

        Parameters
        ----------
        data_idx : list
            List containing the user-defined data point identifiers
            corresponding to the requested data points.

        Optional
        --------
        args : arguments
            Arguments to be provided to the :class:`~Mhysa` class.
        kwargs : keyword arguments
            Keyword arguments to be provided to the :class:`~Mhysa` class.

        """

        # Transpose the data_idx list
        # OPTIMIZE: Improve this
        self._n_data = len(data_idx)
        self._data_idx = [[idx[i] for idx in data_idx]
                          for i in range(len(data_idx[0]))]

        # Make a new snaplist file
        input_data = np.genfromtxt(kwargs['input_file'], dtype=(str),
                                   delimiter=':', autostrip=True)
        snap_file = dict(input_data)['FileWithOutputSnaps']
        snap_file = path.join(path.dirname(kwargs['input_file']), snap_file)
        np.savetxt(snap_file, sorted(self._data_idx[0]), '%i')

        # Inheriting Mhysa __init__()
        super(PrismMhysa, self).__init__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        # Inheriting Mhysa __exit__()
        super(PrismMhysa, self).__exit__(*args, **kwargs)

        # Extract the required snaps
        # Define an empty list holding results
        if self.is_controller:
            self._data_list = []

    def meraxes_hook(self, snapshot, ngals):
        # Create new empty list at snapshot 0
        if(self.is_controller and snapshot == 0):
            self._data_list.append([])

        # Check if this snapshot is required by data
        if snapshot in self._data_idx[0]:
            data_id = self._data_idx[0].index(snapshot)
            gals = self.collect_global_gals(ngals)
            if self.is_controller:
                # TODO: Not all gal_props need Hubble scaling
                data = gals[self._data_idx[1][data_id]] /\
                    self.meraxes_globals.Hubble_h
                if(self._data_idx[2][data_id] == 'sum'):
                    data = data.sum()
                else:
                    raise NotImplementedError

                self._data_list[-1].append(data)

        return(0)

