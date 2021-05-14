# h5flow

A basic MPI framework to create simple sequential workflows, looping over
a dataset within a structured HDF5 file. All MPI calls are hidden behind an API
to allow for (hopefully) seamless running in either of a single-process or
multi-process environment.

## installation

To install::

    pip install -r requirements.txt
    pip install .

## usage

To run a workflow::

    h5flow -i <input file>.h5 -o <output file>.h5 -c <config file>.yaml\
        -s <start position, opt.> -e <end position, opt.>

To run a parallelized workflow::

    mpiexec h5flow -i <input file>.h5 -o <output file>.h5 -c <config file>.yaml\
        -s <start position, opt.> -e <end position, opt.>

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



# writing an `H5FlowGenerator`

