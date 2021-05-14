import h5py
import numpy as np
from mpi4py import MPI

class H5FlowGenerator(object):
    EMPTY = slice(0,0)

    def __init__(self, name, classname, data_manager, input_filename=None, start_position=None, end_position=None, **params):
        self.name = name
        self.classname = classname
        self.data_manager = data_manager
        self.input_filename = input_filename
        self.start_position = start_position
        self.end_position = end_position

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def __iter__(self):
        return self

    def __next__(self):
        # run next function
        next_name, next_slice = self.next()

        # check if all are empty slices
        slices = self.comm.allgather(next_slice)
        if all([sl.stop - sl.start == 0 for sl in slices]):
            raise StopIteration

        return next_name, next_slice

    def next(self):
        '''
            Generate a new slice into the data file. To end loop, return an
            empty slice (``H5FlowGenerator.EMPTY``).

            :returns: ``<dataset name>``, ``<slice>``
        '''
        raise NotImplementedError
