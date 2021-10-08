import os
import shutil
import logging
import h5py

from h5flow import H5FLOW_MPI
from h5flow.core import H5FlowGenerator
from h5flow.data import H5FlowDataManager


class H5FlowDatasetLoopGenerator(H5FlowGenerator):
    '''
        Default dataset looping generator

        First copies input file to output file. Then slices up the dataset
        defined by ``dset_name`` into ``chunk_size`` chunks, separated by MPI rank.

        For some example use cases, the default configuration declaration::

            flow:
                source: <group name>/<dataset group name>
                stages: [...]

        will auto chunk the dataset given by ``<group name>/<dataset group name>``.
        But the manual chunk size specification::

            flow:
                source: input
                stages: [...]

            input:
                classname: H5FlowDatasetLoopGenerator
                dset_name: <group name>/<dataset group name>
                params:
                    chunk_size: <num_rows, opt>
        will chunk the same dataset, but into chunks of ``<num_rows>``.

    '''
    class_version = '0.0.0'

    def __init__(self, **params):
        super(H5FlowDatasetLoopGenerator, self).__init__(**params)

        self.chunk_size = params.get('chunk_size', 'auto')

        if self.input_filename is None:
            raise RuntimeError('must specify an input filename!')

        self.iteration = 0

        self.copy(self.input_filename, self.data_manager.filepath)

    def init(self):
        super(H5FlowDatasetLoopGenerator, self).init()
        self.setup_slices()

    def next(self):
        if self.iteration >= len(self.slices):
            curr_slice = H5FlowGenerator.EMPTY
        else:
            curr_slice = self.slices[self.iteration]
        self.iteration += 1
        return curr_slice

    def __len__(self):
        return len(self.slices)

    def setup_slices(self):
        '''
            Initialize slices for loop

        '''
        # Get the dataset that we will loop over
        dset = self.data_manager.get_dset(self.dset_name)

        self.start_position = self.start_position if self.start_position is not None else 0
        self.end_position = min(self.end_position, len(dset)) if self.end_position is not None else len(dset)

        if self.chunk_size == 'auto':
            # in auto mode, use the default chunk size in the hdf5 file
            self.chunk_size = dset.chunks[0]

        # in manual mode, each process grabs `chunk_size` rows from the file
        start = self.rank * self.chunk_size + self.start_position
        end = self.end_position
        r = range(start, end, self.size * self.chunk_size)
        self.slices = [slice(i, min(i + self.chunk_size, end)) for i in r]

    def copy(self, f0, f1, block=True):
        # copies the whole file for the time being
        if self.rank == 0 and f0 != f1:
            logging.info(f'copy {f0} -> {f1}')
            shutil.copy(f0, f1)
        if block and H5FLOW_MPI:
            self.comm.barrier()
