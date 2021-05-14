import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class H5FlowStage(object):
    '''
        Base class for loop stage

    '''
    def __init__(self, name, classname, data_manager, **params):
        self.name = name
        self.classname = classname
        self.data_manager = data_manager
        self.requires = params.get('requires', list())

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def init(self):
        '''
            Called once before starting the loop. Used to create datasets and
            set file meta-data

            :returns: ``None``
        '''
        pass

    def run(self, source_name, source_slice):
        '''
            Called once per ``source_slice`` provided by the ``h5flow``
            generator.

            :param source_name: path to the source dataset group

            :param source_slice: a 1D slice into the source dataset

            :returns: ``None``
        '''
        pass

    def load(self, source_name, source_slice):
        '''
            Load and dereference "required" data associated with a given source
            - first loads the data subset of ``source_name`` specified by the
            ``source_slice``. Then loops over the datasets in ``self.requires``
            and loads data from ``source_name -> required_name`` references.

            :param source_name: a path to the source dataset group

            :param source_slice: a 1D slice into the source dataset

            :returns: ``dict`` of ``<name> : <data subset>``
        '''
        data = dict()
        data[source_name] = self.data_manager.get_dset(source_name)[source_slice]
        for linked_name in self.requires:
            linked_dset = self.data_manager.get_dset(linked_name)
            data[linked_name] = [linked_dset[ref] for ref in self.data_manager.get_ref(source_name, linked_name)[source_slice]]
        return data
