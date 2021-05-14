import os
import shutil

from h5flow.core import H5FlowGenerator

class H5FlowDatasetLoopGenerator(H5FlowGenerator):
    def __init__(self, **params):
        super(H5FlowDatasetLoopGenerator, self).__init__(**params)

        self.chunk_size = params.get('chunk_size','auto')

        if self.input_filename is None:
            raise RuntimeError('must specify an input filename!')
        self.copy(self.input_filename, self.data_manager.filepath)
        self.setup_slices()
        self.iteration = 0

    def next(self):
        if self.iteration >= len(self.slices):
            curr_slice = H5FlowGenerator.EMPTY
        else:
            curr_slice = self.slices[self.iteration]
        self.iteration += 1
        return self.name, curr_slice

    def __len__(self):
        return len(self.slices)

    def setup_slices(self):
        '''
            Initialize slices for loop

        '''
        # Get the dataset that we will loop over
        dset = self.data_manager.get_dset(self.name)

        if self.chunk_size == 'auto':
            # in auto mode, use the default chunk size in the hdf5 file
            if self.start_position is not None or self.end_position is not None:
                sel = slice(self.start_position, self.end_position)
            else:
                sel = None
            self.slices = [sl[0] for sl in dset.iter_chunks(sel=sel)][self.rank::self.size]
        else:
            # in manual mode, each process grabs chunk_size chunks from the file
            start = self.rank * self.chunk_size + self.start_position if self.start_position \
                else self.rank * self.chunk_size
            end = min(self.end_position, len(dset)) if self.end_position \
                else len(dset)
            r = range(start, end, self.size * self.chunk_size)
            self.slices = [slice(i, min(i+self.chunk_size,end)) for i in r]

    def copy(self, f0, f1, block=True):
        # copies the whole file for the time being
        if self.rank == 0 and f0 != f1:
            shutil.copy(f0, f1)
        if block:
            self.comm.barrier()
