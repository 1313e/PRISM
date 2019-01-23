# -*- coding: utf-8 -*-

# %% IMPORTS
# Built-in imports
from os import path
from sys import platform

# Package imports
import numpy as np
import pytest

# PRISM imports
from prism._dummyMPI import Comm, Intracomm, COMM_WORLD as comm

# Save the path to this directory
dirpath = path.dirname(__file__)

# Save if this platform is Windows
win32 = platform.startswith('win')


# %% CUSTOM CLASSES
class CommTest(Comm):
    pass


# %% PYTEST CLASSES AND FUNCTIONS
# Pytest for custom Comm class
def test_CommTest():
    test_comm = CommTest()
    assert test_comm.name == 'CommTest'


# Pytest for custom Intracomm class
def test_Intracomm():
    with pytest.raises(TypeError):
        Intracomm(1)


# Pytest for COMM_WORLD instance
class Test_COMM_WORLD(object):
    array = np.array([1, 2, 3, 4, 5])
    buffer = np.empty_like(array)

    def test_props(self):
        assert comm.Get_name() == 'dummyMPI_COMM_WORLD'
        assert comm.Get_size() == 1
        assert comm.Get_rank() == 0

    def test_Allgather(self):
        comm.Allgather(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.allgather(self.array)[0] == self.array).all()
        comm.Allgatherv(self.array, self.buffer)
        assert (self.buffer == self.array).all()

    def test_Allreduce(self):
        comm.Allreduce(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.allreduce(self.array) == self.array).all()
        assert (comm.allreduce(1) == 1)
        assert comm.reduce([tuple(self.array.tolist())]) ==\
            tuple(self.array.tolist())

    def test_Barrier(self):
        comm.Barrier()
        comm.barrier()

    def test_Bcast(self):
        comm.Bcast(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.bcast(self.array) == self.array).all()

    def test_Gather(self):
        comm.Gather(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.gather(self.array)[0] == self.array).all()
        comm.Gatherv(self.array, self.buffer)
        assert (self.buffer == self.array).all()

    def test_Is_intra(self):
        assert comm.Is_intra()

    def test_Is_inter(self):
        assert not comm.Is_inter()

    def test_Reduce(self):
        comm.Reduce(self.array, self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.reduce(self.array) == self.array).all()
        assert (comm.reduce(1) == 1)

    def test_Scatter(self):
        comm.Scatter([self.array], self.buffer)
        assert (self.buffer == self.array).all()
        assert (comm.scatter([self.array]) == self.array).all()
        comm.Scatterv([self.array], self.buffer)
        assert (self.buffer == self.array).all()

    def test_Sendrecv(self):
        assert (comm.Sendrecv(self.array) == self.array).all()
        assert (comm.sendrecv(self.array) == self.array).all()
