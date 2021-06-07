import h5py
import numpy as np
import shutil
import os
from mpi4py import MPI
from tqdm import tqdm
import logging
import subprocess
import time

from ..data.lib import dereference

from ..data import H5FlowDataManager
from ..modules import get_class

class H5FlowManager(object):
    def __init__(self, config, output_filename, input_filename=None, start_position=None, end_position=None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.drop_list = config.get('flow').get('drop',list())

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
        source_config = config[source_name] if source_name in config else self.default_generator_config(source_name)

        self.generator = get_class(source_config.get('classname'))(
            classname=source_config.get('classname'),
            dset_name=source_config.get('dset_name'),
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
                requires=args.get('requires',None),
                **args.get('params',dict()))
            for name,args in zip(stage_names, stage_args)
            ]

    def default_generator_config(self, source_name):
        if self.rank == 0:
            logging.warning(f'Could not find generator description, using default loop behavior on {source_name} dataset')
        return dict(
            classname='H5FlowDatasetLoopGenerator',
            dset_name=source_name
            )

    def init(self):
        logging.debug(f'init generator')
        self.generator.init()
        for stage in self.stages:
            logging.debug(f'init stage {stage.name} source: {self.generator.dset_name}')
            stage.init(self.generator.dset_name)
        self.comm.barrier()

    def run(self):
        loop_gen = tqdm(self.generator) if self.rank == 0 else self.generator
        stage_requirements = [[r for stage in self.stages[:i+1] for r in stage.requires] for i in range(len(self.stages))]
        for chunk in loop_gen:
            logging.debug(f'run on {self.generator.dset_name} chunk: {chunk}')
            cache = dict()
            for i, (stage, requirements) in enumerate(zip(self.stages, stage_requirements)):
                self.update_cache(cache, self.generator.dset_name, chunk, requirements)
                logging.debug(f'run stage {stage.name} source: {self.generator.dset_name} chunk: {chunk} cache contains {len(cache)} objects')
                stage.run(self.generator.dset_name, chunk, cache)
        self.comm.barrier()

    def finish(self):
        logging.debug(f'finish generator')
        self.generator.finish()
        self.comm.barrier()
        for stage in self.stages:
            logging.debug(f'finish stage {stage.name} source: {self.generator.dset_name}')
            stage.finish(self.generator.dset_name)
        self.comm.barrier()

        logging.debug(f'close data manager')
        for drop in self.drop_list:
            self.data_manager.delete(drop)
        self.data_manager.close_file()
        self.comm.barrier()
        if len(self.drop_list) and self.rank == 0:
            # repacks the hdf5 file to recover space from dropped datasets
            tempfile = os.path.join(os.path.dirname(self.data_manager.filepath), '.temp-{}.h5'.format(time.time()))
            subprocess.run(['h5repack', self.data_manager.filepath, tempfile])
            os.replace(tempfile, self.data_manager.filepath)
            os.remove(tempfile)
        self.comm.barrier()

    def update_cache(self, cache, source_name, source_slice, requirements):
        '''
            Load and dereference "required" data associated with a given source
            - first loads the data subset of ``source_name`` specified by the
            ``source_slice``. Then loops over the datasets in ``self.requires``
            and loads data from ``source_name -> required_name`` references.
            Called automatically once per loop, just before calling ``run``.

            Only loads data to the cache if it is not already present

            :param cache: ``dict`` cache to update

            :param source_name: a path to the source dataset group

            :param source_slice: a 1D slice into the source dataset

        '''
        for name in list(cache.keys()).copy():
            if name not in requirements and name != source_name:
                del cache[name]

        if source_name not in cache:
            cache[source_name] = self.data_manager.get_dset(source_name)[source_slice]

        for linked_name in requirements:
            if linked_name not in cache:
                linked_dset = self.data_manager.get_dset(linked_name)
                refs, ref_dir = self.data_manager.get_ref(source_name, linked_name)
                regions = self.data_manager.get_ref_region(source_name, linked_name)

                cache[linked_name] = dereference(linked_dset, refs, regions, sel=source_slice, ref_direction=ref_dir, as_masked=True)
