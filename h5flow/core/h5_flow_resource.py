from .. import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI

comm = MPI.COMM_WORLD if H5FLOW_MPI else None
rank = comm.Get_rank() if H5FLOW_MPI else 0
size = comm.Get_size() if H5FLOW_MPI else 1

resources = dict()

class H5FlowResource(object):
    '''
        Base class for an accessible static resource. Provides:

         - ``classname``: resource class
         - ``class_version``: a ``str`` version number (``'major.minor.fix'``, default = ``'0.0.0'``)
         - ``data_manager``: an ``H5FlowDataManager`` instance used to access the output file
         - ``input_filename``: an optional input filename (default = ``None``)
         - ``start_position``: an optional start position to begin iterating (default = ``None``)
         - ``end_position``: an optional end position to stop iterating (default = ``None``)
         - ``comm``: MPI world communicator (if needed, else ``None``)
         - ``rank``: MPI group rank
         - ``size``: MPI group size

        To build a custom resource, implement the ``init()`` or ``finish()`` methods.

        To access a resource, declare it in the config file under ``resources``::

            resources:
                 - classname: ExampleResource
                   params:
                       a_parameter: example

        And then access it from a stage or generator via::

            from h5flow.core import resources

            resources['ExampleResource']

    '''
    class_version = '0.0.0'

    def __init__(self, classname, data_manager, input_filename=None,
            start_position=None, end_position=None, **params):
        self.classname = classname
        self.data_manager = data_manager
        self.input_filename = input_filename
        self.end_position = end_position
        self.start_position = start_position

        self.comm = MPI.COMM_WORLD if H5FLOW_MPI else None
        self.rank = self.comm.Get_rank() if H5FLOW_MPI else 0
        self.size = self.comm.Get_size() if H5FLOW_MPI else 1

    def init(self, source_name):
        '''
            Called once before starting the loop and before generator has been
            initialized. Used to load data or configure resource.

            :returns: ``None``
        '''
        pass

    def finish(self, source_name):
        '''
            Called once after finishing loop and after generators and stages
            have finished.

            :returns None:
        '''
        pass
