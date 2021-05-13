import h5py
import numpy as np

from ..data import open_file, get_dset, get_ref

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
        data[self.source] = get_dset(self.output_file, self.source)[source_slice]

        for linked_name in self.requires:
            linked_dset = get_dset(self.output_file, linked_name)
            data[self.linked_name] = [linked_dset[ref] for ref in get_ref(self.output_file, self.source, linked_name)[source_slice]]

        return data
