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
                requires=self.format_requirements(args.get('requires',list())),
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

    def format_requirements(self, requirements):
        '''
            Converts list from the "requires" configuration option into an
            list of dicts with::

                name: name of object to place in cache
                path: list of reference datasets (parent, child) to load for this requirement
                indices_only: boolean if dataset should only load reference indices rather than data

        '''
        req = []
        for r in requirements:
            if isinstance(r, str):
                req.append(dict(
                    name= r,
                    path= [r],
                    indices_only= False
                    ))
            elif isinstance(r,dict):
                d = dict(name=r['name'])
                if 'path' in r:
                    if isinstance(r['path'],str):
                        d['path'] = [r['path']]
                    elif isinstance(r['path'],list):
                        d['path'] = r['path']
                    else:
                        raise ValueError(f'Unrecognized path specification in {r}')
                else:
                    d['path'] = [d['name']]
                d['indices_only'] = r['indices_only'] if 'indices_only' in r else False
                req.append(d)
            else:
                raise ValueError(f'Unrecognized requirement {r}')
        return req


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
            ``source_slice``. Then loops over the specification dicts in ``self.requires``
            and loads data from references found in `'path'`.

            Called automatically once per loop, just before calling ``run``.

            Only loads data to the cache if it is not already present.

            :param cache: ``dict`` cache to update

            :param source_name: a path to the source dataset group

            :param source_slice: a 1D slice into the source dataset

        '''
        required_names = [r['name'] for r in requirements]

        for name in list(cache.keys()).copy():
            if name not in required_names and name != source_name:
                del cache[name]

        if source_name not in cache:
            cache[source_name] = self.data_manager.get_dset(source_name)[source_slice]

        for i,linked_name in enumerate(required_names):
            if linked_name not in cache:
                cache[linked_name] = self.load_requirement(requirements[i], source_name, source_slice)

    def load_requirement(self, req, source_name, source_slice):
        '''
            Loads a requirement specified by::

                path: list of references to traverse
                indices_only: True to load only indices and not data

        '''
        path = req['path']
        indices_only = req['indices_only']

        sel = np.r_[source_slice]
        shape = [len(sel),]
        mask = np.zeros(len(sel), dtype=bool)
        dref = None
        for i,(p,c) in enumerate(zip([source_name]+path[:-1], path)):
            dset = self.data_manager.get_dset(c)
            ref, ref_dir = self.data_manager.get_ref(p,c)
            reg = self.data_manager.get_ref_region(p,c)

            dref = dereference(sel.flatten(), ref, dset, region=reg, ref_direction=ref_dir,
                indices_only=True if i != len(path)-1 else indices_only)
            shape += dref.shape[-1:]

            dref = dref.reshape(*shape)
            mask = np.expand_dims(mask, axis=-1) | dref.mask

            if i != len(path)-1:
                sel = dref

        dref.mask = mask
        return dref









