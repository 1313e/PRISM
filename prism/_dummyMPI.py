# -*- coding: utf-8 -*-

"""
Dummy MPI
=========
Dummy module that emulates the functionality of the :mod:`~mpi4py.MPI` module.
This is a specialized version of the `mpi_dummy` package available at
https://gitlab.mpcdf.mpg.de/ift/mpi_dummy/tree/master

"""


# %% IMPORTS
# Future imports
from __future__ import absolute_import, division, print_function

# Built-in imports
from copy import copy
import sys

# Package imports
import numpy as np

# All declaration
__all__ = ['Comm', 'Intracomm', 'COMM_WORLD', 'COMM_SELF']

# Python2/Python3 compatibility
if(sys.version_info.major >= 3):
    unicode = str


# %% COMM CLASS DEFINITION
# Make dummy Comm class
class Comm(object):
    def __init__(self):
        # Save name of this class if not saved already
        try:
            self._name
        except AttributeError:
            self.name = self.__class__.__name__

        # Save rank and size of communicator
        self._rank = 0
        self._size = 1

    # %% CLASS PROPERTIES
    @property
    def name(self):
        return(self._name)

    @name.setter
    def name(self, name):
        if isinstance(name, (str, np.string_, unicode)):
            self._name = name
        else:
            raise TypeError("Input argument 'name' is not of type 'str'!")

    @property
    def rank(self):
        return(self._rank)

    @property
    def size(self):
        return(self._size)

    # %% GENERAL CLASS METHODS
    def _get_buffer(self, buff):
        # If buff is a list or tuple, return the first element
        if isinstance(buff, (list, tuple)):
            return(buff[0])
        # Else, return buff itself
        else:
            return(buff)

    def _scatter_gather(self, sendbuf, recvbuf=None):
        # Unwrap the sending and receiving buffers
        sendbuf = self._get_buffer(sendbuf)
        recvbuf = self._get_buffer(recvbuf)

        # If no receiving buffer was supplied, return a copy of sendbuf
        if recvbuf is None:
            # Check if sendbuf has the copy()-method and use it if so
            try:
                return(sendbuf.copy())
            # Else, use the built-in copy() function
            except AttributeError:
                return(copy(sendbuf))
        # If a receiving buffer was supplied, use it
        else:
            recvbuf[:] = sendbuf
            return(recvbuf)

    # %% VISIBLE CLASS METHODS
    # TODO: Implement dummy versions of missing communication methods
    # Still missing: Alltoall and non-blocking/synchronous (I/S) methods
    def Get_name(self):
        return(self.name)

    def Get_rank(self):
        return(self.rank)

    def Get_size(self):
        return(self.size)

    def Allgather(self, sendbuf, recvbuf, *args, **kwargs):
        return(self.Gather(sendbuf, recvbuf))

    def allgather(self, sendobj, *args, **kwargs):
        return(self.gather(sendobj))

    def Allgatherv(self, sendbuf, recvbuf, *args, **kwargs):
        return(self.Gatherv(sendbuf, recvbuf))

    def Allreduce(self, sendbuf, recvbuf, *args, **kwargs):
        return(self.Reduce(sendbuf, recvbuf))

    def allreduce(self, sendobj, *args, **kwargs):
        return(self.reduce(sendobj))

    def Barrier(self):
        pass

    def barrier(self):
        pass

    def Bcast(self, buf, *args, **kwargs):
        return(buf)

    def bcast(self, obj, *args, **kwargs):
        return(obj)

    def Gather(self, sendbuf, recvbuf, *args, **kwargs):
        return(self._scatter_gather(sendbuf, recvbuf))

    def gather(self, sendobj, *args, **kwargs):
        return([sendobj])

    def Gatherv(self, sendbuf, recvbuf, *args, **kwargs):
        return(self._scatter_gather(sendbuf, recvbuf))

    def Is_intra(self):
        return(isinstance(self, Intracomm))

    def Is_inter(self):
        return(False)

    def Reduce(self, sendbuf, recvbuf, *args, **kwargs):
        return(self._scatter_gather(sendbuf, recvbuf))

    def reduce(self, sendobj, *args, **kwargs):
        if np.isscalar(sendobj):
            return(sendobj)
        else:
            return(self._scatter_gather(sendobj))

    def Scatter(self, sendbuf, recvbuf, *args, **kwargs):
        return(self._scatter_gather(sendbuf, recvbuf))

    def scatter(self, sendobj, *args, **kwargs):
        return(sendobj[0])

    def Scatterv(self, sendbuf, recvbuf, *args, **kwargs):
        return(self._scatter_gather(sendbuf, recvbuf))

    def Sendrecv(self, sendbuf, *args, **kwargs):
        return(sendbuf)

    def sendrecv(self, sendobj, *args, **kwargs):
        return(sendobj)


# %% INTRACOMM CLASS DEFINITION
# Make dummy Intracomm class
class Intracomm(Comm):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(Intracomm, self).__init__(*args, **kwargs)


# %% INITIALIZE COMM_WORLD AND COMM_SELF
COMM_WORLD = Intracomm('dummyMPI_COMM_WORLD')
COMM_SELF = Intracomm('dummyMPI_COMM_SELF')
