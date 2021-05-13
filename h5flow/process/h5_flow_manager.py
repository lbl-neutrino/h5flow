import h5py
import numpy as np
import shutil
from mpi4py import MPI
from tqdm import tqdm

from ..data import open_file, get_dset
from ..module import get_class

class H5FlowManager(object):
    def __init__(self, input_filename, output_filename, config, start_position=None, end_position=None):
        self.source = config['flow'].get('source')
        self.start_position = start_position
        self.end_position = end_position

        self.copy(input_filename, output_filename)
        self.output_file = open_file(output_filename)

        self.stage_names = config['flow'].get('stages')
        self.stage_args = [config.get(stage_name) for stage_name in self.stage_names]
        self.stages = [
            get_class(args.get('classname'))(
                classname=args.get('classname'),
                name=name,
                source=self.source,
                output_file=self.output_file,
                **args.get('params',dict()))
            for name,args in zip(self.stage_names, self.stage_args)
            ]

    def copy(self, f0, f1):
        # copies the whole file for the time being
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            shutil.copy(f0, f1)
        comm.barrier()

    def init(self):
        comm = MPI.COMM_WORLD
        for stage in self.stages:
            stage.init()
        comm.barrier()

    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        sel = slice(self.start_position, self.end_position)
        chunks = list(get_dset(self.output_file, self.source).iter_chunks(sel=sel if sel.start or sel.stop else None))[rank::size]

        len_chunks = comm.allgather(len(chunks))
        chunks += [(slice(0,0,1),)] * (max(len_chunks) - len(chunks))
        for chunk in tqdm(chunks, desc=f'Rank:{rank}'):
            for stage in self.stages:
                stage.run(chunk)
        comm.barrier()

    def finish(self):
        comm = MPI.COMM_WORLD
        self.output_file.close()
        comm.barrier()


