import numpy as np

from h5flow.core import H5FlowStage, H5FlowGenerator, H5FlowResource

class ExampleResource(H5FlowResource):
    class_version = '0.0.0'

    default_path = 'meta'

    def __init__(self, **params):
        super(ExampleResource,self).__init__(**params)

        # get example parameters
        self.path = params.get('path', self.default_path)
        self.data = params.get('data', dict())

    def init(self, source_name):
        # can save data to output file
        self.data_manager.set_attrs(self.path, **self.data)

    def get(self, name):
        # but allows access to in-memory objects
        return self.data[name]

class ExampleGenerator(H5FlowGenerator):
    class_version = '0.0.0'

    # best practice is to declare default values as class attributes
    default_chunk_size = 1024
    default_iterations = 10

    def __init__(self, **params):
        super(ExampleGenerator,self).__init__(**params)

        # get parameters from the params list
        self.chunk_size = params.get('chunk_size', self.default_chunk_size)

        # prepare anything needed for the loop
        if self.end_position is None:
            self.end_position = self.default_iterations
        self.iteration = 0

    def __len__(self):
        # optional
        return self.end_position

    def init(self):
        # create any new datasets (including references)
        self.data_manager.create_dset(self.dset_name, dtype=int)
        # best practice to write all config parameters to dataset
        self.data_manager.set_attrs(self.dset_name,
            classname=self.classname,
            class_version=self.class_version,
            chunk_size=self.chunk_size,
            end_position=self.end_position
            )

    def next(self):
        # loop termination condition
        if self.iteration >= self.end_position:
            return H5FlowGenerator.EMPTY
        self.iteration += 1

        # just append data equivalent to the current index in the file
        next_slice = self.data_manager.reserve_data(self.dset_name, self.chunk_size)
        data = np.arange(next_slice.start, next_slice.stop)
        self.data_manager.write_data(self.dset_name, next_slice, data)

        # return slice into newly added data
        return next_slice

from h5flow.core import resources

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
        dtype = self.data_manager.get_dset(source_name).dtype # get the dtype of an existing dataset
        self.data_manager.create_dset(self.output_dset, dtype=dtype) # create a new dataset
        self.data_manager.create_ref(source_name, self.output_dset) # create a new reference table (source -> output)

    def run(self, source_name, source_slice, cache):
        # manipulate data from cache
        data = cache[source_name]

        # access resource data
        resources['ExampleResource'].get('val0')

        # To add data to the output file:
        #  1. reserve a new data region within the output dataset
        new_slice = self.data_manager.reserve_data(self.output_dset, len(data))
        #  2. write the data to the new data region
        self.data_manager.write_data(self.output_dset, new_slice, data)

        # To add references to the output file:
        #  1. create an (N,2) array with parent->child indices (this does parent idx->0-10 child index)
        parent_idcs = np.arange(source_slice.start, source_slice.stop).reshape(-1,1,1)
        child_idcs = np.clip(np.arange(new_slice.start, new_slice.start+10).reshape(1,-1,1) + parent_idcs - source_slice.start,0,new_slice.stop-1)
        parent_idcs, child_idcs = np.broadcast_arrays(parent_idcs, child_idcs)
        ref = np.unique(np.concatenate((parent_idcs, child_idcs), axis=-1).reshape(-1,2), axis=0) # reshape to (parent, child), and only use unique references (repeats can be used)
        #  2. then write them into the file (no space reservation needed)
        self.data_manager.write_ref(source_name, self.output_dset, ref)
