import h5py
import numpy as np

from .. import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI

class H5FlowGenerator(object):
    '''
        Base class for generators. Provides the following attributes:

         - ``classname``: stage class
         - ``class_version``: a ``str`` version number (``'major.minor.fix'``, default = ``'0.0.0'``)
         - ``dset_name``: dataset to be accessed by each stage
         - ``data_manager``: an ``H5FlowDataManager`` instance used to access the output file
         - ``input_filename``: an optional input filename (default = ``None``)
         - ``start_position``: an optional start position to begin iterating (default = ``None``)
         - ``end_position``: an optional end position to stop iterating (default = ``None``)
         - ``comm``: MPI world communicator (if needed, else ``None``)
         - ``rank``: MPI group rank
         - ``size``: MPI group size

        To build a custom generator, inherit from this base class and implement
        the ``next()`` method.

        Example::

            class ExampleGenerator(H5FlowGenerator):
                class_version = '0.0.0'

                default_max_value = 2**32-1
                default_chunk_size = 1024
                default_iterations = 100

                def __init__(self, **params):
                    super(ExampleGenerator,self).__init__(**params)

                    # get config parameters
                    self.max_value = params.get('max_value', self.default_max_value)
                    self.chunk_size = params.get('chunk_size', self.default_chunk_size)

                    # prepare loop
                    if self.end_position is None:
                        self.end_position = self.default_iterations
                    self.iteration = 0

                def init(self):
                    # prepare output file
                    self.data_manager.create_dset(self.dset_name, dtype=int)
                    self.data_manager.set_attrs(self.obj_name,
                        classname=self.classname,
                        class_version=self.class_version,
                        max_value=self.max_value,
                        chunk_size=self.chunk_size,
                        end_position=self.end_position,
                        )

                def next(self):
                    if self.iteration >= self.end_position:
                        return H5FlowGenerator.EMPTY
                    self.iteration += 1

                    next_slice = self.data_manager.reserve_data(self.dset_name, self.chunk_size)
                    self.data_manager.write_data(self.dset_name, next_slice, np.random.randint(self.max_value, self.chunk_size))

                    return next_slice

        This example creates a generator that will fill the ``dset_name``
        dataset with random integer data (max value of ``max_value``) in chunks
        of length ``chunk_size``. The process will continue for ``end_position``
        iterations until it ends. Note that if running with MPI, each *process*
        will run for the same number of iterations (and so the data file will
        be ``N`` times larger).

    '''
    EMPTY = slice(-1,-1)
    class_version = '0.0.0'

    def __init__(self, classname, dset_name, data_manager, input_filename=None, start_position=None, end_position=None, **params):
        self.classname = classname
        self.dset_name = dset_name
        self.data_manager = data_manager
        self.input_filename = input_filename
        self.start_position = start_position
        self.end_position = end_position

        self.comm = MPI.COMM_WORLD if H5FLOW_MPI else None
        self.rank = self.comm.Get_rank() if H5FLOW_MPI else 0
        self.size = self.comm.Get_size() if H5FLOW_MPI else 1

    def __iter__(self):
        return self

    def __next__(self):
        # run next function
        next_slice = self.next()

        # check if all are empty slices
        slices = self.comm.allgather(next_slice) if H5FLOW_MPI else [next_slice]
        if all([sl == H5FlowGenerator.EMPTY for sl in slices]):
            raise StopIteration

        return next_slice

    def init(self):
        '''
            Prepare output file to be written to, called once before
            initializing the flow stages.

        '''
        pass

    def next(self):
        '''
            Generate a new slice into the source dataset in the data file. To
            end loop, return an empty slice (``H5FlowGenerator.EMPTY``).

            :returns: ``<slice>`` into ``self.dset_name`` data
        '''
        raise NotImplementedError

    def finish(self):
        '''
            Clean up any open files / etc, called once after run loop finishes

        '''
        pass
