import h5py
import numpy as np
import shutil
import os
from mpi4py import MPI
from tqdm import tqdm

from ..data import H5FlowDataManager
from ..module import get_class

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class H5FlowManager(object):
    def __init__(self, input_filename, output_filename, config, start_position=None, end_position=None):
        self.source = config['flow'].get('source')
        self.start_position = start_position
        self.end_position = end_position

        self.remove(output_filename, block=False)
        self.copy(input_filename, output_filename)
        self.data_manager = H5FlowDataManager(output_filename)

        self.stage_names = config['flow'].get('stages')
        self.stage_args = [config.get(stage_name) for stage_name in self.stage_names]
        self.stages = [
            get_class(args.get('classname'))(
                classname=args.get('classname'),
                name=name,
                source=self.source,
                data_manager=self.data_manager,
                **args.get('params',dict()))
            for name,args in zip(self.stage_names, self.stage_args)
            ]

    def remove(self, f, block=True):
        if rank == 0:
            os.remove(f)
        if block:
            comm.barrier()

    def copy(self, f0, f1, block=True):
        # copies the whole file for the time being
        if rank == 0:
            shutil.copy(f0, f1)
        if block:
            comm.barrier()

    def init(self):
        for stage in self.stages:
            stage.init()
        comm.barrier()

    def run(self):
        sel = slice(self.start_position, self.end_position)
        chunks = list(self.data_manager.get_dset(self.source).iter_chunks(sel=sel if sel.start or sel.stop else None))[rank::size]

        len_chunks = comm.allgather(len(chunks))
        chunks += [(slice(0,0,1),)] * (max(len_chunks) - len(chunks))
        for chunk in tqdm(chunks, desc=f'{rank}'):
            for stage in self.stages:
                stage.run(chunk)
        comm.barrier()

    def finish(self):
        self.data_manager.close_file()
        comm.barrier()


