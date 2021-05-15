import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class H5FlowStage(object):
    '''
        Base class for loop stage. Provides the following attributes:
         - ``name``: instance name of stage (declared in configuration file)
         - ``classname``: stage class
         - ``data_manager``: an ``H5FlowDataManager`` instance used to access the output file
         - ``requires``: a list of dataset names to load when calling ``H5FlowStage.load()``
         - ``comm``: MPI world communicator (if needed)
         - ``rank``: MPI group rank
         - ``size``: MPI group size

         To build a custom stage, inherit from this base class and implement
         the ``init()`` and the ``run()`` methods.

         Example::

            class ExampleStage(H5FlowStage):
                custom_param_default_value = None
                default_obj_name = 'obj0'

                def __init__(**params):
                    super(ExampleStage,self).__init__(**params)

                    # grab parameters from configuration file here, e.g.
                    self.custom_param = params.get('custom_param', self.custom_param_default_value)
                    self.obj_name = self.name + '/' + params.get('obj_name', self.default_obj_name)

                def init(self, source_name):
                    # declare any new datasets and set dataset metadata, e.g.

                    self.data_manager.set_attrs(self.obj_name, custom_param=self.custom_param)
                    self.data_manager.create_dset(self.obj_name)

                def run(self, source_name, source_slice):
                    # load, process, and save new data objects

                    data = self.load(source_name, source_slice)



    '''
    def __init__(self, name, classname, data_manager, requires=None, **params):
        self.name = name
        self.classname = classname
        self.data_manager = data_manager
        self.requires = requires if requires is not None else list()

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def init(self, source_name):
        '''
            Called once before starting the loop. Used to create datasets and
            set file meta-data

            :returns: ``None``
        '''
        pass

    def run(self, source_name, source_slice, cache):
        '''
            Called once per ``source_slice`` provided by the ``h5flow``
            generator.

            :param source_name: path to the source dataset group

            :param source_slice: a 1D slice into the source dataset

            :param cache: pre-loaded data from ``requires`` list

            :returns: ``None``
        '''
        pass

    def update_cache(self, cache, source_name, source_slice):
        '''
            Load and dereference "required" data associated with a given source
            - first loads the data subset of ``source_name`` specified by the
            ``source_slice``. Then loops over the datasets in ``self.requires``
            and loads data from ``source_name -> required_name`` references.
            Called automatically once per loop, just before calling ``run``.

            :param cache: ``dict`` cache to update

            :param source_name: a path to the source dataset group

            :param source_slice: a 1D slice into the source dataset

        '''
        if source_name not in cache:
            cache[source_name] = self.data_manager.get_dset(source_name)[source_slice]
        for linked_name in self.requires:
            if linked_name not in cache:
                linked_dset = self.data_manager.get_dset(linked_name)
                cache[linked_name] = [linked_dset[ref] for ref in self.data_manager.get_ref(source_name, linked_name)[source_slice]]

