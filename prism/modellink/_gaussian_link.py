# -*- coding: utf-8 -*-

# Simple Gaussian ModelLink
# Compatible with Python 3.5+

"""
GaussianLink
============
Provides the definition of the :class:`~GaussianLink` class.

"""


# %% IMPORTS
# Package imports
import numpy as np

# PRISM imports
from prism._internal import check_vals, np_array
from prism.modellink import ModelLink

# All declaration
__all__ = ['GaussianLink']


# %% CLASS DEFINITION
class GaussianLink(ModelLink):
    """
    :class:`~prism.modellink.ModelLink` class wrapper for a simple Gaussian
    model, used for testing the functionality of the *PRISM* pipeline.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`\\sum_i A_i\\exp\\left(-\\frac{(x-B_i)^2}{2C_i^2}\\right)` to
        obtain the data value.

    """

    def __init__(self, n_gaussians=1, *args, **kwargs):
        """
        Initialize an instance of the :class:`~GaussianLink` class.

        Optional
        --------
        n_gaussians : int. Default: 1
            The number of Gaussians to use for the Gaussian model in this
            instance. The resulting number of model parameters
            :attr:`~prism.modellink.ModelLink.n_par` will be
            :math:`3*n_{gaussians}`.

        """

        # Set the number of Gaussians
        self._n_gaussians = check_vals(n_gaussians, 'n_gaussians', 'pos',
                                       'int')

        # Set the name of this GaussianLink instance
        self.name = 'GaussianLink_n%i' % (self._n_gaussians)

        # Request single or multi model calls
        self.call_type = 'hybrid'

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super().__init__(*args, **kwargs)

    # %% GAUSSIANLINK CLASS PROPERTIES
    @property
    def n_gaussians(self):
        """
        int: Number of Gaussians used in this :obj:`~GaussianLink` instance.

        """

        return(self._n_gaussians)

    # %% GAUSSIANLINK CLASS METHODS
    def get_str_repr(self):
        return(['n_gaussians=%r' % (self._n_gaussians)] if(
                self._n_gaussians != 1) else [])

    def get_default_model_parameters(self):
        # Set default parameters for every Gaussian
        A = [1, 10, 5]
        B = [0, 10, 5]
        C = [0, 5, 2]

        # Create default parameters dict and return it
        par_dict = {}
        for i in range(1, self._n_gaussians+1):
            par_dict['A%i' % (i)] = list(A)
            par_dict['B%i' % (i)] = list(B)
            par_dict['C%i' % (i)] = list(C)
        return(par_dict)

    def call_model(self, emul_i, par_set, data_idx):
        par = par_set
        mod_set = [0]*len(data_idx)
        for i, idx in enumerate(data_idx):
            for j in range(1, self._n_gaussians+1):
                mod_set[i] +=\
                    par['A%i' % (j)]*np.exp(-1*((idx-par['B%i' % (j)])**2 /
                                                (2*par['C%i' % (j)]**2)))

        return(np_array(mod_set).T)

    def get_md_var(self, emul_i, par_set, data_idx):
        return(pow(0.1*np.ones_like(data_idx), 2))
