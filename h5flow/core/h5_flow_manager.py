import h5py
import numpy as np
import shutil
import os
from mpi4py import MPI
from tqdm import tqdm
import logging
import subprocess
import time

from ..data import H5FlowDataManager
from ..modules import get_class

class H5FlowManager(object):
    '''
        Overarching coordination class. Creates data managers, generators, and
        stages according to the specified workflow. After initializing
        the workflow, executes each generator and stages' ``init``, ``run``, and
        ``finish`` methods in sequence.

        Also manages refreshing data in the ``cache`` and dropping datasets from
        the final output file.

        The standard execution sequence (as implemented in ``h5flow.run``) is::

            manager = H5FlowManager(**args)

            manager.init()      # initialize components
            manager.run()       # execute workflow run loop
            manager.finish()    # clean up components

    '''
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
        '''
            Create an ``H5FlowDataManager`` to coordinate access into
            ``output_filename``. Access to the data manager is provided via::

                manager = H5FlowManager(<config>, <filename>)
                manager.data_manager # data manager instance

            :param output_filename: ``str``, output file path

            :param config: ``dict``, parsed yaml config for workflow

        '''
        self.data_manager = H5FlowDataManager(output_filename)

    def configure_generator(self, input_filename, config, start_position, end_position):
        '''
            Create an ``H5FlowGenerator`` to produce slices into the source
            dataset. Access to the generator is provided via::

                manager = H5FlowManager(<config>, <filename>)
                manager.generator # generator instance

            If no generator configuration is found in the ``config``, a default
            dataset loop generator is created.

            :param input_filename: ``str``, input file path passed to the generator

            :param config: ``dict``, parsed yaml config for workflow

            :param start_position: ``int``, dataset start index passed to generator

            :param end_position: ``int``, dataset end index passed to generator

        '''
        source_name = config['flow'].get('source')
        source_config = config[source_name] if source_name in config else self._default_generator_config(source_name)

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
        '''
            Create instances of ``H5FlowStages`` in the sequence specified in
            the ``config``. Access to the stages are provided via::

                manager = H5FlowManager(<config>, <filename>)
                manager.stages # data manager instance

            :param config: ``dict``, parsed yaml config for workflow

        '''
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

    def _default_generator_config(self, source_name):
        if self.rank == 0:
            logging.warning(f'Could not find generator description, using default loop behavior on {source_name} dataset')
        return dict(
            classname='H5FlowDatasetLoopGenerator',
            dset_name=source_name
            )

    def init(self):
        '''
            Execute ``init()`` method of generator and stages, in sequence.

        '''
        logging.debug(f'init generator')
        self.generator.init()
        for stage in self.stages:
            logging.debug(f'init stage {stage.name} source: {self.generator.dset_name}')
            stage.init(self.generator.dset_name)
        self.comm.barrier()

    def run(self):
        '''
            Run loop, executing ``run()`` method of generator and stages, in
            sequence. Terminate once all processes return an
            ``H5FlowGenerator.EMPTY``.

            Also refreshes the cache with required datasets on each stage.

        '''

        loop_gen = tqdm(self.generator) if self.rank == 0 else self.generator
        for chunk in loop_gen:
            logging.debug(f'run on {self.generator.dset_name} chunk: {chunk}')
            cache = dict()
            for stage in self.stages:
                stage.update_cache(cache, self.generator.dset_name, chunk)
                logging.debug(f'run stage {stage.name} source: {self.generator.dset_name} chunk: {chunk} cache contains {len(cache)} objects')
                stage.run(self.generator.dset_name, chunk, cache)
        self.comm.barrier()

    def finish(self):
        '''
            Execute ``finish()`` method of generator and stages. After all
            components have finished, drop datasets that are not wanted in
            output file.

        '''
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
        self.comm.barrier()



