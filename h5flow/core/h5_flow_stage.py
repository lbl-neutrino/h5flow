import h5py
import numpy as np
from mpi4py import MPI
import logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class H5FlowStage(object):
    '''
        Base class for loop stage. Provides the following attributes:
         - ``name``: instance name of stage (declared in configuration file)
         - ``classname``: stage class
         - ``class_version``: a ``str`` version number (``'major.minor.fix'``, default = ``'0.0.0'``)
         - ``data_manager``: an ``H5FlowDataManager`` instance used to access the output file
         - ``requires``: a list of dataset names to load when calling ``H5FlowStage.load()``
         - ``comm``: MPI world communicator (if needed)
         - ``rank``: MPI group rank
         - ``size``: MPI group size

         To build a custom stage, inherit from this base class and implement
         the ``init()`` and the ``run()`` methods.

         Example::

            class ExampleStage(H5FlowStage):
                class_version = '0.0.0' # keep track of a version number for each class

                default_custom_param = None
                default_obj_name = 'obj0'

                def __init__(**params):
                    super(ExampleStage,self).__init__(**params)

                    # grab parameters from configuration file here, e.g.
                    self.custom_param = params.get('custom_param', self.default_custom_param)
                    self.obj_name = self.name + '/' + params.get('obj_name', self.default_obj_name)

                def init(self, source_name):
                    # declare any new datasets and set dataset metadata, e.g.

                    self.data_manager.set_attrs(self.obj_name,
                        classname=self.classname,
                        class_version=self.class_version,
                        custom_param=self.custom_param,
                        )
                    self.data_manager.create_dset(self.obj_name)

                def run(self, source_name, source_slice):
                    # load, process, and save new data objects

                    data = self.load(source_name, source_slice)



    '''
    class_version = '0.0.0'

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

    def finish(self, source_name):
        '''
            Clean up any open files / etc, called once after run loop finishes

        '''
        pass
