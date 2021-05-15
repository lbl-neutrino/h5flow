import time

from h5flow.core import H5FlowStage

class TestStage(H5FlowStage):
    def __init__(self, **params):
        super(TestStage, self).__init__(**params)
        self.output_dset = params.get('output_dset')

    def init(self, source_name):
        # best practice is to write all configuration variables to the dataset
        self.data_manager.set_attrs(
            self.output_dset,
            classname=self.classname,
            input_dset=source_name,
            output_dset=self.output_dset,
            test_attr='test_value'
            )

        dtype = self.data_manager.get_dset(source_name).dtype
        self.data_manager.create_dset(self.output_dset, dtype=dtype)
        self.data_manager.create_ref(self.output_dset, source_name)
        self.data_manager.create_ref(source_name, self.output_dset)

    def run(self, source_name, source_slice):
        # load a copy of the data into memory
        data = self.load(source_name, source_slice)
        data[source_name]

        # reserve a new data region within the output dataset
        new_slice = self.data_manager.reserve_data(self.output_dset, len(data[source_name]))

        # write a copy of the data to the new data region
        self.data_manager.write_data(self.output_dset, new_slice, data[source_name])

        # reserve the same data region in the output reference dataset
        self.data_manager.reserve_ref(self.output_dset, source_name, new_slice)

        # write 1:1 new -> old references
        self.data_manager.write_ref(self.output_dset, source_name, new_slice, range(source_slice.start, source_slice.stop))

        # reserve the same source data region in the source reference dataset
        self.data_manager.reserve_ref(source_name, self.output_dset, source_slice)

        # write 1:1 old -> new references
        self.data_manager.write_ref(source_name, self.output_dset, source_slice, range(new_slice.start, new_slice.stop))
