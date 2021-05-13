import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class H5FlowStage(object):
    def __init__(self, **params):
        self.source = params.get('source')
        self.name = params.get('name')
        self.classname = params.get('classname')

        self.output_file = params.get('output_file')

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
        data[self.source] = self.output_file.get_dset(self.source)[source_slice]

        for linked_name in self.requires:
            linked_dset = self.output_file.get_dset(linked_name)
            data[linked_name] = [linked_dset[ref] for ref in self.output_file.get_ref(self.source, linked_name)[source_slice]]
        return data
