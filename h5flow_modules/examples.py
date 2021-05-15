import numpy as np

from h5flow.core import H5FlowStage, H5FlowGenerator

class ExampleGenerator(H5FlowGenerator):
    class_version = '0.0.0'

    # best practice is to declare default values as class attributes
    default_max_value = 2**32-1
    default_chunk_size = 1024
    default_iterations = 10

    def __init__(self, **params):
        super(ExampleGenerator,self).__init__(**params)

        # get parameters from the params list
        self.max_value = params.get('max_value', self.default_max_value)
        self.chunk_size = params.get('chunk_size', self.default_chunk_size)

        # prepare anything needed for the loop
        if self.end_position is None:
            self.end_position = self.default_iterations
        self.iteration = 0

    def init(self):
        # create any new datasets (including references)
        self.data_manager.create_dset(self.dset_name, dtype=int)
        # best practice to write all config parameters to dataset
        self.data_manager.set_attrs(self.dset_name,
            classname=self.classname,
            class_version=self.class_version,
            max_value=self.max_value,
            chunk_size=self.chunk_size,
            end_position=self.end_position
            )

    def next(self):
        # loop termination condition
        if self.iteration >= self.end_position:
            return H5FlowGenerator.EMPTY
        self.iteration += 1

        # just append random data
        next_slice = self.data_manager.reserve_data(self.dset_name, self.chunk_size)
        self.data_manager.write_data(self.dset_name, next_slice, np.random.randint(self.max_value, size=self.chunk_size))

        # return slice into newly added data
        return next_slice

class ExampleStage(H5FlowStage):
    class_version = '0.0.0'

    def __init__(self, **params):
        super(ExampleStage, self).__init__(**params)

        # get parameters from params list
        self.output_dset = params.get('output_dset')

    def init(self, source_name):
        # best practice is to write all configuration variables to the dataset
        self.data_manager.set_attrs(self.output_dset,
            classname=self.classname,
            class_version=self.class_version,
            input_dset=source_name,
            output_dset=self.output_dset,
            test_attr='test_value'
            )

        # then set up any new datasets that will be added (including references)
        dtype = self.data_manager.get_dset(source_name).dtype
        self.data_manager.create_dset(self.output_dset, dtype=dtype)
        self.data_manager.create_ref(self.output_dset, source_name)
        self.data_manager.create_ref(source_name, self.output_dset)

    def run(self, source_name, source_slice, cache):
        # manipulate data from cache
        data = cache[source_name]

        # remove things from the cache if subsequent stages need to load fresh data
        del cache[source_name]

        # To add data to the output file:
        #  1. reserve a new data region within the output dataset
        new_slice = self.data_manager.reserve_data(self.output_dset, len(data))
        #  2. write the data to the new data region
        self.data_manager.write_data(self.output_dset, new_slice, data)

        # To add references to the output file:
        #  1. reserve the same source data region in the source reference dataset
        self.data_manager.reserve_ref(source_name, self.output_dset, source_slice)
        #  2. write 1:1 old -> new references
        ref = range(new_slice.start, new_slice.stop)
        self.data_manager.write_ref(source_name, self.output_dset, source_slice, ref)

        # To add bi-directional references to the output file:
        #  1. reserve the same data region in the output reference dataset
        self.data_manager.reserve_ref(self.output_dset, source_name, new_slice)
        #  2. write 1:1 new -> old references
        ref = range(source_slice.start, source_slice.stop)
        self.data_manager.write_ref(self.output_dset, source_name, new_slice, ref)
