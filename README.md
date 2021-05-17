# h5flow

A basic MPI framework to create simple sequential workflows, looping over
a dataset within a structured HDF5 file. All MPI calls are hidden behind an API
to allow for (hopefully) seamless running in either a single-process or a
multi-process environment.

## installation

To setup a fresh conda environment::

    conda create --name <env> --file requirements.txt
    pip install .

To update an existing environment::

    conda activate <env>
    conda install --file requirements.txt
    pip install .

To run tests::

    pytest

To run mpi tests::

    mpiexec pytest --with-mpi

## usage

To run a single-process workflow::

    h5flow -o <output file>.h5 -c <config file>.yaml\
        -i <input file, opt.> -s <start position, opt.> -e <end position, opt.>

To run a parallelized workflow::

    mpiexec h5flow -o <output file>.h5 -c <config file>.yaml\
        -i <input file, opt.> -s <start position, opt.> -e <end position, opt.>

# h5flow hdf5 structure

`h5flow` requires a specific, table-like hdf5 structure with references
between datasets. Each dataset is expected to be stored within a group path::

    /<dataset0_path>/data
    /<dataset1_path>/data
    /<dataset2_path>/data

Datasets are expected to be single-dimesional structured arrays. References
between datasets are expected to be stored alongside the parent dataset::

    /<dataset0_path>/data
    /<dataset0_path>/ref/<dataset1_path> # references from dataset0 -> dataset1
    /<dataset0_path>/ref/<dataset2_path> # references from dataset0 -> dataset2
    /<dataset1_path>/data
    /<dataset1_path>/ref/<dataset0_path> # references from dataset1 -> dataset0
    ...

with the same dimensions as the parent dataset.

# h5flow workflow

`h5flow` uses a yaml config file to define the workflow. The main definition of
the workflow is defined under the `flow` key::

    flow:
        source: <dataset to loop over, or generator name>
        stages: [<first sequential stage name>, <second sequential stage name>]

First the `source` defines the loop source. By default, you may specify an
existing dataset and an `H5FlowDatasetLoopGenerator` will be used. `stages`
then defines the names and sequential order of the stages should be executed on
each data chunk provided by the generator.

## generators

To define a generator, specify the name, an `H5FlowGenerator`-inheriting
classname, along with any desired parameters at the top level within the yaml
file::

    dummy_generator:
        classname: DummyGenerator
        dset_name: <dataset to be accessed by each stage>
        params:
            dummy_param: value

For both generators and stages, classes will be discovered for within the
current directory, the `./h5flow_modules/` directory, or the `h5flow/modules`
directory (in that order) and automatically loaded upon runtime.

## stages

To define a stage, specify the name, an `H5FlowStage`-inheriting classname, along
with any desired parameters at the top level within the yaml file::

    flow:
        source: generator_stage_or_path_to_a_dataset
        stages: [dummy_stage0, dummy_stage1]

    dummy_stage0:
        classname: DummyStage
        params:
            dummy_param0: 10
            dummy_param1: [a,list,of,strings]

    dummy_stage1:
        classname: OtherDummyStage

# writing an `H5FlowStage`

Any `H5FlowStage`-inheriting class has 4 main components:
    1. a constructor (`__init__()`)
    2. class attributes
    3. an initialization `init()` method
    4. and a `run()` method


None of the methods are required for the class to function within `h5flow`, but
each provide particular access points into the flow sequence.

First, the constructor is called when the flow sequence is first created and
is passed each of the ``<key>: <value>`` pairs declared in the config yaml. For
example, the parameters declared in the following config file::

    example:
        classname: ExampleStage
        params:
            parameter_name: parameter_value

can be accessed with a constructor::

    class ExampleStage(H5FlowStage):

        default_parameter = 0

        def __init__(self, **params):
            super(ExampleStage,self).__init__(**params) # needed to inherit H5FlowStage functionality

            parameter = params.get('parameter_name', default_parameter)

Next, class attributes (``default_parameter`` above) can be used to declare class-
specific data (e.g. default values for parameters).

Then, the ``init(self, source_name)`` method is called just before entering the
loop. Information about which dataset will be used in the loop is provided to
allow for initialization of dataset-dependent properties (or error out if the
dataset is somehow invalid for the class). Use this function to initialize new
datasets and write meta-data. See the ``h5flow_modules/examples.py`` for an
working example.

Finally, the ``run(self, source_name, source_slice, cache)`` method is called
at each step of the loop. This is where the bulk of the processing occurs.
``source_name`` is a string pointing to the current loop dataset. ``source_slice``
provides a python ``slice`` object into the full ``source_name`` data array for
the current loop iteration. ``cache`` is a python ``dict`` object filled with
pre-loaded data of the ``source_slice`` into the ``source_name`` dataset and any
``required`` datasets specified by the config yaml. Items deleted from the
``cache`` will be reloaded from the underlying hdf5 file, if required by
downstream stages. Reading and writing other data objects from the file can be
done via the ``H5FlowDataManager`` object within ``self.data_manager``. Refer to
the ``h5flow_modules/examples.py`` for a working example.

# writing an `H5FlowGenerator`

