from tqdm import tqdm
import logging
import sys
import numpy as np

from .. import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI

from ..data.lib import dereference_chain

from ..data import H5FlowDataManager
from ..modules import get_class

from .h5_flow_resource import resources, H5FlowResource


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
        self.comm = MPI.COMM_WORLD if H5FLOW_MPI else None
        self.rank = self.comm.Get_rank() if H5FLOW_MPI else 0
        self.size = self.comm.Get_size() if H5FLOW_MPI else 1

        self.drop_list = config['flow'].get('drop', list())

        # set up the data manager
        self.configure_data_manager(output_filename, config)

        # set up resources
        self.configure_resources(config, input_filename, start_position, end_position)

        # set up the file chunk generator
        self.configure_generator(input_filename, config, start_position, end_position)

        # set up flow stages
        self.configure_flow(config)

        if H5FLOW_MPI:
            self.comm.barrier()

    def configure_resources(self, config, input_filename, start_position, end_position):
        '''
            Create ``H5FlowResource`` instance for each object in config that
            inherits from ``H5FlowResource``

            :param input_filename: ``str``, input file path passed to the generator

            :param config: ``dict``, parsed yaml config for workflow

            :param start_position: ``int``, dataset start index passed to generator

            :param end_position: ``int``, dataset end index passed to generator

        '''
        global resources

        for obj_config in config.get('resources', list()):
            obj_classname = obj_config['classname']
            obj_path = obj_config.get('path', None)
            print(obj_classname, obj_path)
            obj_class = get_class(obj_classname, path=obj_path)
            if issubclass(obj_class, H5FlowResource):
                resources[obj_classname] = obj_class(
                    classname=obj_classname,
                    data_manager=self.data_manager,
                    input_filename=input_filename,
                    start_position=start_position,
                    end_position=end_position,
                    **obj_config.get('params', dict())
                )
            else:
                raise RuntimeError(f'failed to load resource {obj_classname} - does not inherit from H5FlowResource')

    def configure_data_manager(self, output_filename, config):
        '''
            Create an ``H5FlowDataManager`` to coordinate access into
            ``output_filename``. Access to the data manager is provided via::

                manager = H5FlowManager(<config>, <filename>)
                manager.data_manager # data manager instance

            :param output_filename: ``str``, output file path

            :param config: ``dict``, parsed yaml config for workflow

        '''
        self.data_manager = H5FlowDataManager(output_filename, drop_list=self.drop_list)

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
        source_name = config['flow']['source']
        source_config = config[source_name] if source_name in config else self._default_generator_config(source_name)

        self.generator = get_class(source_config['classname'], path=source_config.get('path', None))(
            classname=source_config['classname'],
            dset_name=source_config['dset_name'],
            data_manager=self.data_manager,
            input_filename=input_filename,
            start_position=start_position,
            end_position=end_position,
            **source_config.get('params', dict())
        )

    def configure_flow(self, config):
        '''
            Create instances of ``H5FlowStages`` in the sequence specified in
            the ``config``. Access to the stages are provided via::

                manager = H5FlowManager(<config>, <filename>)
                manager.stages # data manager instance

            :param config: ``dict``, parsed yaml config for workflow

        '''
        stage_names = config['flow'].get('stages', list())
        stage_args = [config[stage_name] for stage_name in stage_names]
        self.stages = [
            get_class(args['classname'], path=args.get('path',None))(
                classname=args['classname'],
                name=name,
                data_manager=self.data_manager,
                requires=self.format_requirements(args.get('requires', list())),
                **args.get('params', dict()))
            for name, args in zip(stage_names, stage_args)
        ]

    def _default_generator_config(self, source_name):
        if self.rank == 0:
            print(f'Could not find generator description, using default loop behavior on {source_name} dataset')
        return dict(
            classname='H5FlowDatasetLoopGenerator',
            dset_name=source_name,
            path='h5flow.modules'
        )

    def format_requirements(self, requirements):
        '''
            Converts list from the "requires" configuration option into an
            list of dicts with::

                name: name of object to place in cache
                path: list of reference datasets (parent, child) to load for this requirement
                index_only: boolean if dataset should only load reference indices rather than data

        '''
        req = []
        for r in requirements:
            if isinstance(r, str):
                req.append(dict(
                    name=r,
                    path=[r],
                    index_only=False
                ))
            elif isinstance(r, dict):
                d = dict(name=r['name'])
                if 'path' in r:
                    if isinstance(r['path'], str):
                        d['path'] = [r['path']]
                    elif isinstance(r['path'], list):
                        d['path'] = r['path']
                    else:
                        raise ValueError(f'Unrecognized path specification in {r}')
                else:
                    d['path'] = [d['name']]
                d['index_only'] = r['index_only'] if 'index_only' in r else False
                req.append(d)
            else:
                raise ValueError(f'Unrecognized requirement {r}')
        return req

    def init(self):
        '''
            Execute ``init()`` method of resources, generator, and stages, in
            sequence and in that order.

        '''
        global resources
        for classname, resource in resources.items():
            resource.init(self.generator.dset_name)

        self.generator.init()

        for stage in self.stages:
            stage.init(self.generator.dset_name)

        if H5FLOW_MPI:
            self.comm.barrier()

    def run(self):
        '''
            Run loop, executing ``run()`` method of generator and stages, in
            sequence. Terminate once all processes return an
            ``H5FlowGenerator.EMPTY``.

            Also refreshes the cache with required datasets on each stage.

        '''
        if self.rank == 0:
            print(f'Run loop on {self.generator.dset_name}:')
            print('  ' + ' -> '.join([stage.name for stage in self.stages]))

        loop_gen = tqdm(self.generator, smoothing=1) if self.rank == 0 else self.generator
        stage_requirements = [[r for stage in self.stages[:i + 1] for r in stage.requires] for i in range(len(self.stages))]
        for chunk in loop_gen:
            cache = dict()
            for i, (stage, requirements) in enumerate(zip(self.stages, stage_requirements)):
                self.update_cache(cache, self.generator.dset_name, chunk, requirements)
                stage.run(self.generator.dset_name, chunk, cache)
            sys.stdout.flush()
        if H5FLOW_MPI:
            self.comm.barrier()

    def finish(self):
        '''
            Execute ``finish()`` method of generator and stages. After all
            components have finished, drop datasets that are not wanted in
            output file.

        '''
        self.generator.finish()
        if H5FLOW_MPI:
            self.comm.barrier()
        for stage in self.stages:
            stage.finish(self.generator.dset_name)

        global resources
        for classname, resource in resources.items():
            resource.finish(self.generator.dset_name)

        self.data_manager.finish()

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

        for i, linked_name in enumerate(required_names):
            if linked_name not in cache:
                cache[linked_name] = self.load_requirement(requirements[i], source_name, source_slice)

    def load_requirement(self, req, source_name, source_slice):
        '''
            Loads a requirement specified by::

                path: list of references to traverse
                index_only: True to load only indices and not data

            :param req: ``dict`` with items ``path : list of datasets`` and ``index_only : bool, true to only load index in to final dataset``. Loads a chain of references in a sequence of ``[source_name, *path]``

            :param source_name: ``str`` base dataset to load

            :param source_slice: ``slice`` into ``source_name`` to load

        '''
        path = req['path']
        index_only = req['index_only']

        logging.debug((f'loading requirement {req["name"]}: ' + ' -> '.join([source_name] + path)) +
                      ('' if not index_only else '(index)'))

        if self.data_manager.ref_exists(source_name, path[0]):
            chain = list(zip([source_name] + path[:-1], path))
        elif len(path) > 1:
            chain = list(zip(path[:-1], path[1:]))
        else:
            try:
                if not index_only:
                    return self.data_manager.get_dset(path[0])[source_slice]
                else:
                    return np.r_[source_slice]
            except Exception as e:
                logging.info('failed to load: ' + path[0]
                             + ('' if not index_only else '(index)') + ' : '
                             + str(e))
            return None

        try:
            data = self.data_manager.get_dset(path[-1])
            ref, ref_dir = list(zip(*[self.data_manager.get_ref(p, c) for p, c in chain]))
            regions = [self.data_manager.get_ref_region(p, c) for p, c in chain]

            return dereference_chain(source_slice, ref, data=data, regions=regions, ref_directions=ref_dir, indices_only=index_only)
        except Exception as e:
            logging.info(('failed to load: ' + ' -> '.join([source_name] + path)
                          + ('' if not index_only else '(index)') + ' : '
                          + str(e)))
            return None
