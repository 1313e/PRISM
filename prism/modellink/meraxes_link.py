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
from dragons import munge
import logging
from mhysa import Mhysa
import numpy as np

# PRISM imports
from prism import ModelLink
from prism._internal import RequestError

# All declaration
__all__ = ['MeraxesLink']

logger = logging.getLogger('MERAXES')


# %% MERAXESLINK CLASS DEFINITION
class MeraxesLink(ModelLink):
    """
    :class:`~ModelLink` class wrapper for the semi-analytic galaxy evolution
    model *Meraxes*.

    This class requires the `multi` branch of *Mhysa* and the `fesc_deps`
    branch of *Meraxes* (for now).

    Formatting data_idx
    -------------------
    snap : int
        The snapshot at which the data value needs to be obtained.
    gal_prop : {'BHemissivity', 'BlackHoleMass', 'ColdGas', 'dt', 'Fesc',\
                'FescBH', 'GrossStellarMass', 'MetalsColdGas',\
                'MetalsStellarMass', 'Mvir', 'Sfr', 'StellarMass', 'Type',\
                'ghost_flag'}
        The specific galaxy property that needs to be used for the data value,
        using the names as defined by *Meraxes*.
    operation : {'smf', 'sum'}
        The specific operation that needs to be performed in order to obtain
        the data value.
    oper_info : type
        Additional information which depends on the specified `operation`. Can
        be left empty if not required.

        - 'smf': This is the :math:`\\log_{10}(M_*/{\\rm M_{\\odot}})` at\
        which :math:`\\log_{10}(\\phi/{\\rm Mpc^{-3}})` needs to be returned.

    """

    def __init__(self, input_file, *args, **kwargs):
        """
        Initialize an instance of the :class:`~MeraxesLink` subclass.

        Parameters
        ----------
        input_file : str
            The name of the file that contains the input parameters for
            *Meraxes*.

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
        # Controller returns the gathered data_list after some processing
        if mhysa.is_controller:
            # Gather the data_list
            data_list = np.array(mhysa._data_list)

            # Get the operation data_idx
            ops = np.array([idx[2] for idx in data_idx])

            # Check if any operation involved the SMF
            smf_idx = np.arange(len(data_idx))[ops == 'smf']

            # Loop over all data points that involved the SMF
            for idx in smf_idx:
                # Obtain the data phi for this data point
                data_phi = self._data_val[idx]

                # If there are any values -np.infty (no gals), penalize them
                data_list[~np.isfinite(data_list[:, idx]), idx] = data_phi-5

            # Return the data_list
            return(data_list)

        # The workers return a dummy value
        else:
            return(0)

    def get_md_var(self, emul_i, data_idx):
        super(MeraxesLink, self).get_md_var(emul_i=emul_i, data_idx=data_idx)


# %% MERAXESLINK CLASS PROPERTIES
    @property
    def input_file(self):
        """
        String containing the absolute path to the input parameter file used
        for *Meraxes*.

        """

        return(self._input_file)


# %% PRISMMHYSA CLASS DEFINITION
class PrismMhysa(Mhysa):
    """
    A :class:`~Mhysa` subclass manipulating the outputs of *Meraxes* to be used
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

        # Ignore NumPy error messages
        np.seterr(divide='ignore')

        # Extract the snap_list
        self._n_data = len(data_idx)
        self._snaps = np.array([idx[0] for idx in data_idx])
        self._data_idx = data_idx

        # Make a new snaplist file
        input_data = np.genfromtxt(kwargs['input_file'], dtype=(str),
                                   delimiter=':', autostrip=True)
        snap_file = dict(input_data)['FileWithOutputSnaps']
        snap_file = path.join(path.dirname(kwargs['input_file']), snap_file)
        np.savetxt(snap_file, sorted(self._snaps), '%i', newline=' ')

        # Inheriting Mhysa __init__()
        super(PrismMhysa, self).__init__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        # Inheriting Mhysa __exit__()
        super(PrismMhysa, self).__exit__(*args, **kwargs)

        # Extract the required snaps
        # Define an empty list holding results
        if self.is_controller:
            self._data_list = []

        # Make shortcuts for some Meraxes global variables
        self._hubble_h = self.meraxes_globals.Hubble_h
        self._volume = pow(self.meraxes_globals.BoxSize/self._hubble_h, 3) *\
            self.meraxes_globals.VolumeFactor
        self._no_hubble_scaling = ['ghost_flag', 'Type', 'FescBH', 'Fesc',
                                   'Sfr', 'BHemissivity']

    def meraxes_hook(self, snapshot, ngals):
        # Wrap entire process in a try-statement
        try:
            # Create new empty list at snapshot 0
            if(self.is_controller and snapshot == 0):
                self._data_list.append([0]*self._n_data)

            # Check if this snapshot is required by data
            snap_req = np.arange(self._n_data)[self._snaps == snapshot]
            if(snap_req.size > 0):
                # Flag that StellarMass SMF has not been made yet
                smf_flag = 0

                # Collect galaxy properties
                gals = self.collect_global_gals(ngals)

                # Controller only
                if self.is_controller:
                    # Loop over all requested data points at this snapshot
                    for data_id in snap_req:
                        # Extract the idx corresponding to this data point
                        idx = self._data_idx[data_id]

                        # Perform user-defined operation
                        # Calculate the stellar mass function
                        if(idx[2] == 'smf' and idx[1] == 'StellarMass'):
                            # If StellarMass SMF has not been created yet
                            if not smf_flag:
                                data = np.log10(gals[idx[1]][gals[idx[1]] > 0])
                                data = data+10-np.log10(self._hubble_h)
                                mf, edges = munge.mass_function(
                                    data, self._volume, 100, (0, 12), 0, 1)
                                smf_flag = 1

                            # Select required mass index
                            mass_idx = edges.searchsorted(idx[3])-1
                            data = np.log10(mf[mass_idx, 1])

                        # Take the sum of the galaxy property
                        elif(idx[2] == 'sum'):
                            data = gals[idx[1]].sum()

                            # Perform Hubble scaling if required
                            if idx[1] not in self._no_hubble_scaling:
                                data /= self._hubble_h

                        # If unknown operation is given
                        else:
                            raise RequestError("The requested operation '%s' "
                                               "is invalid!" % (idx[2]))

                        # Save resulting data value
                        self._data_list[-1][data_id] = data

        # If any errors are raised, catch them and return -1
        except Exception as error:
            logger.error(error)
            return(-1)

        # If everything went correctly, return 0
        return(0)
