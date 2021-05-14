import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class H5FlowStage(object):
    def __init__(self, **params):
        self.rank = rank
        self.size = size

        self.source = params.get('source')
        self.name = params.get('name')
        self.classname = params.get('classname')
        self.data_manager = params.get('data_manager')
        self.requires = params.get('requires', list())

    def init(self):
        pass

    def process(self, source_slice, data):
        raise NotImplementedError

    def run(self, source_slice):
        data = self.load_data(source_slice) # loads all required data + references
        self.process(source_slice, data)

    def load_data(self, source_slice):
        data = dict()
        data[self.source] = self.data_manager.get_dset(self.source)[source_slice]

        for linked_name in self.requires:
            linked_dset = self.data_manager.get_dset(linked_name)
            data[linked_name] = [linked_dset[ref] for ref in self.data_manager.get_ref(self.source, linked_name)[source_slice]]
        return data
