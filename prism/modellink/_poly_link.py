# -*- coding: utf-8 -*-

# Simple Polynomial ModelLink
# Compatible with Python 3.5+

"""
PolyLink
========
Provides the definition of the :class:`~PolyLink` class.

"""


# %% IMPORTS
# Package imports
import numpy as np

# PRISM imports
from prism._internal import check_vals
from prism.modellink._modellink import ModelLink

# All declaration
__all__ = ['PolyLink']


# %% CLASS DEFINITION
class PolyLink(ModelLink):
    """
    :class:`~prism.modellink.ModelLink` class wrapper for a simple polynomial
    model, used for testing the functionality of the *PRISM* pipeline.

    Formatting data_idx
    -------------------
    x : int
        The value that needs to be used for :math:`x` in the function
        :math:`\\sum_i C_ix^i` to obtain the data value.

    """

    def __init__(self, order=2, *args, **kwargs):
        """
        Initialize an instance of the :class:`~PolyLink` class.

        Optional
        --------
        order : int. Default: 2
            The polynomial order to use  for the polynomial model in this
            instance. The resulting number of model parameters
            :attr:`~prism.modellink.ModelLink.n_par` will be
            :math:`1+\\mathrm{order}`.

        """

        # Set the polynomial order
        self._order = check_vals(order, 'order', 'pos', 'int')

        # Set the name of this PolyLink instance
        self.name = 'PolyLink_p%i' % (self._order)

        # Request single or multi model calls
        self.call_type = 'hybrid'

        # Request only controller calls
        self.MPI_call = False

        # Inheriting ModelLink __init__()
        super().__init__(*args, **kwargs)

    # %% POLYLINK CLASS PROPERTIES
    @property
    def order(self):
        """
        int: Polynomial order used in this :obj:`~PolyLink` instance.

        """

        return(self._order)

    # %% POLYLINK CLASS METHODS
    def get_str_repr(self):
        return(['order=%r' % (self._order)] if(self._order != 2) else [])

    def get_default_model_parameters(self):
        # Set default coefficients for every polynomial term
        C = [0, 10, 1]

        # Create default parameters dict and return it
        par_dict = {}
        for i in range(self._order+1):
            par_dict['C%i' % (i)] = list(C)
        return(par_dict)

    def call_model(self, emul_i, par_set, data_idx):
        mod_set = np.zeros([len(data_idx), *np.shape(par_set['C0'])])
        for i, idx in enumerate(data_idx):
            for j in range(self._order+1):
                mod_set[i] += par_set['C%i' % (j)]*idx**j

        return(mod_set.T)

    def get_md_var(self, emul_i, par_set, data_idx):
        return(pow(0.1*np.ones_like(data_idx), 2))
