import h5py
import numpy as np
import shutil
import os
from mpi4py import MPI
from tqdm import tqdm

from ..data import H5FlowDataManager
from ..modules import get_class

class H5FlowManager(object):
    def __init__(self, input_filename, output_filename, config, start_position=None, end_position=None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # set up the data manager
        self.configure_data_manager(output_filename, config)

        # set up the file chunk generator
        self.configure_generator(input_filename, config, start_position, end_position)

        # set up flow stages
        self.configure_flow(config)

        self.comm.barrier()

    def configure_data_manager(self, output_filename, config):
        self.data_manager = H5FlowDataManager(output_filename)

    def configure_generator(self, input_filename, config, start_position, end_position):
        source_name = config['flow'].get('source')
        source_config = config.get(source_name, self.default_generator_config(source_name))

        self.generator = get_class(source_config.get('classname'))(
            classname=source_config.get('classname'),
            name=source_name,
            data_manager=self.data_manager,
            input_filename=input_filename,
            start_position=start_position,
            end_position=end_position,
            **source_config.get('params',dict())
            )

    def configure_flow(self, config):
        stage_names = config['flow'].get('stages')
        stage_args = [config.get(stage_name) for stage_name in stage_names]
        self.stages = [
            get_class(args.get('classname'))(
                classname=args.get('classname'),
                name=name,
                data_manager=self.data_manager,
                **args.get('params',dict()))
            for name,args in zip(stage_names, stage_args)
            ]

    @staticmethod
    def default_generator_config(source_name):
        return dict(classname='H5FlowDatasetLoopGenerator')

    def init(self):
        for stage in self.stages:
            stage.init()
        self.comm.barrier()

    def run(self):
        for name, chunk in tqdm(self.generator, desc=f'{self.rank}'):
            for stage in self.stages:
                stage.run(name, chunk)
        self.comm.barrier()

    def finish(self):
        self.data_manager.close_file()
        self.comm.barrier()


