h5flow
======

.. image:: https://readthedocs.org/projects/h5flow/badge/?version=latest
    :target: https://h5flow.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/peter-madigan/h5flow/actions/workflows/test.yml/badge.svg
    :target: https://github.com/peter-madigan/h5flow/actions/
    :alt: Test Status

A basic MPI framework to create simple sequential workflows, looping over
a dataset within a structured HDF5 file. All MPI calls are hidden behind an API
to allow for straight-forward implementation of parallelized algorithms without
the need to be familiar with MPI.

installation
------------

First, download this code::

    git clone https://github.com/peter-madigan/h5flow
    cd h5flow

To install dependencies in a fresh conda environment::

    export CC=mpicc
    export HDF5_MPI="ON"
    conda env create --name <env> --file environment.yml
    conda activate <env>

To update an existing environment with necessary dependencies::

    export CC=mpicc
    export HDF5_MPI="ON"
    conda env update --name <env> --file environment.yml
    conda activate <env>

This will attempt to install a parallel-compatible version of HDF5 and h5py. If
you would prefer to install h5flow without parallel capabilities, use the
provided ``environment-nompi.yml`` instead.

To install::

    pip install .

To run tests::

    pytest

To run MPI-enabled tests::

    mpiexec pytest --with-mpi

usage
-----

To run a single-process workflow::

    h5flow -o <output file>.h5 -c <config file>.yaml\
        -i <input file, opt.> -s <start position, opt.> -e <end position, opt.>

The output file here is the destination hdf5 file path. The config file is a
yaml description of the workflow sequence and the parameters for each custom
module in the workflow.

The other arguments are optional, depending on the specifics of the workflow.
Generators may require an input file (specified by the input file argument). The
start and end position arguments allow for partial processing of a given input
file.

To run a parallelized workflow::

    mpiexec h5flow -o <output file>.h5 -c <config file>.yaml\
        -i <input file, opt.> -s <start position, opt.> -e <end position, opt.>

which will launch as many instances of h5flow as there are cores - each instance
are given subsets of the input file to process according to the behavior of the
generator declared in the workflow.

There are also some alternative entry points that can be used to launch ``h5flow``
in the event that one of the above doesn't work for your application::

    python -m h5flow <args>
    run_h5flow.py <args>

You can also use ``h5flow`` without mpi4py by checking the global ``H5FLOW_MPI``
variable::

    from h5flow import H5FLOW_MPI
    if H5FLOW_MPI:
        # mpi-compatible code, e.g.
        from mpi4py import MPI
    else:
        # non-mpi compatible code

h5flow hdf5 structure
=====================

``h5flow`` requires a specific, table-like hdf5 structure with references
between datasets. Each dataset is expected to be stored within a group path::

    /<dataset0_path>/data
    /<dataset1_path>/data
    /<dataset2_path>/data

Datasets are expected to be single-dimensional structured arrays. References
between datasets are expected to be stored alongside the parent dataset::

    /<dataset0_path>/data
    /<dataset0_path>/ref/<dataset1_path>/ref # references from dataset0 -> dataset1
    /<dataset0_path>/ref/<dataset2_path>/ref # references from dataset0 -> dataset2
    /<dataset1_path>/data
    /<dataset1_path>/ref/<dataset0_path>/ref # references from dataset1 -> dataset0
    ...

To facilitate fast + parallel read/writes there is a companion structured
dataset ``ref_region`` at the corresponding position as the ``ref`` dataset that
indicates where to look in the reference dataset for the corresponding row.
E.g.::

    /<dataset0_path>/data
    /<dataset0_path>/ref/<dataset1_path>/ref # references from dataset0 -> dataset1 (and back)
    /<dataset0_path>/ref/<dataset1_path>/ref_region # regions for dataset0 -> dataset1 reference
    /<dataset0_path>/ref/<dataset2_path>/ref # references from dataset0 -> dataset2 (and back)
    /<dataset0_path>/ref/<dataset2_path>/ref_region # regions for dataset0 -> dataset2 reference

The ``.../ref_region`` datasets are a 1D structured array with fields ``'start': int``
and ``'stop': int``. These represent the min and max indices of the ``.../ref`` array
that contain the corresponding index. So for example::

    data0 = np.array([0, 1, 2])
    data1 = np.array([0, 1, 2, 3])

    ref = np.array([[0,1], [1,2]]) # links data0[0] <-> data1[1], data0[1] <-> data1[2]

    ref_region0 = np.array([(0,1), (1,2), (0,0)]) # ref_region for data0, the (0,0) entries correspond to entries without references
    ref_region1 = np.array([(0,0), (0,1), (1,2), (0,0)]) # ref_region for data1

example structure
-----------------

Let's walk through an example in detail. Let's say we have two datasets ``A`` and
``B``::

    /A/data
    /B/data

These must be single dimensional arrays with either a simple or structured type::

    f['/A/data'].dtype # [('id', 'i8'), ('some_val', 'f4')], either a structured array
    f['/B/data'].dtype # 'f4', or a simple array

    f['/A/data'].shape # (N,), only single dimension datasets
    f['/B/data'].shape # (M,)

Now, let's say there are references between the two datasets::

    /A/ref/B/ref
    /A/ref/B/ref_region
    /B/ref/A/ref_region

In particular, we've created references from ``A->B``, so the ``../ref`` is stored
(by convention) at ``/A/ref/B/ref``. This ``../ref`` dataset is 2D of shape ``(L,2)``
where ``L`` is not necessarily equal to ``N`` or ``M`` and contains indices into
each of the corresponding datasets. By convention, index 0 is the "parent"
dataset (``A``) and index 1 is the "child" dataset (``B``)::

    f['/A/ref/B/ref'].shape # (L,2)
    f['/A/ref/B/ref'][:,0] # indices into f['/A/data']
    f['/A/ref/B/ref'][:,1] # indices into f['/B/data']

    linked_a = f['/A/data'][:][ f['/A/ref/B/ref'][:,0] ] # data from A that can be linked to dataset B (note that you must load the dataset before the fancy indexing can be applied)
    linked_b = f['/B/data'][:][ f['/A/ref/B/ref'][:,1] ] # data from B that can be linked to dataset A
    linked_a.shape == linked_b.shape # (L,)

Converting this into a dataset that can be broadcast back into either the ``A`` or
``B`` shape is facilitated with a helper de-referencing function::

    from h5flow.data import dereference

    b2a = dereference(
        slice(0, 1000),     # indices of A to load references for, shape: (n,)
        f['/A/ref/B/ref'],  # references to use, shape: (L,)
        f['/B/data']        # dataset to load, shape: (M,)
        )
    b2a.shape # (n,l), where l is the max number of B items associated with a row in A
    b2a.dtype == f['/B/data'].dtype # True!

    b_sum = b2a.sum(axis=-1) # use numpy masked array interface to operate on the b2a array
    b_sum.shape # (n,), data can be broadcast back onto your selected indices

And inverse relationships can be found by redefining the "ref_direction":::

    a2b = dereference(
        slice(0, 250),      # indices of B to load references for, shape: (m,)
        f['/A/ref/B/ref'],  # references to use, same as before, shape: (L,)
        f['/A/data'],       # dataset to load, shape: (N,)
        ref_direction = (1,0) # now use references from 1->0 (B->A) [default is (0,1)]
        )
    a2b.shape # (m,q), where q is the max number of A items associated with a row in B
    a2b.dtype == f['/A/data'].dtype # True!

This works just fine - until you start needing to keep track of a very large
number of references (~50000). In that case, we use the special
``region`` (or ``../ref_region`` as it is called in the HDF5 file) dataset / array
to facilitate only partially loading from the reference dataset::

    b2a_subset = dereference(
        slice(0, 1000),      # indices of A to load references for, shape: (n,)
        f['/A/ref/B/ref'],  # references to use, shape: (L,)
        f['/B/data'],       # dataset to load, shape: (M,)
        region = f['/A/ref/B/ref_region'] # lookup regions in references, shape: (N,)
        )
    b2a_subset == b2a # same result as before, but internally this is handled in a much more efficient manner

    %timeit dereference(0, f['/A/ref/B/ref'], f['/B/data']) # runtime: max(100ns * len(f['/A/ref/B/ref']), 1ms)
    %timeit dereference(0, f['/A/ref/B/ref'], f['/B/data'], f['/A/ref/B/ref_region']) # runtime: ~5ms

One feature of the dereferencing scheme is that it is relatively easy to follow
references through many complex relationship. In particular, the ``mask`` and
``indices_only`` arguments can be used to selectively load the references that
are returned from one call to ``dereference`` in another::

    a2b_ref = dereference(
        slice(0, 1000),     # indices of A to load references for, shape: (n,)
        f['/A/ref/B/ref'],  # references to use, shape: (L,)
        f['/B/data'],       # dataset to load, shape: (M,)
        region = f['/A/ref/B/ref_region'], # lookup regions in references, shape: (N,)
        indices_only = True
        )
    a2b2c = dereference(
        a2b_ref.ravel(), # convert b2a references into a 1D selection array, shape: (n*l,)
        f['/B/ref/C/ref'], # now use B->C references, shape: (K,)
        f['/C/data'], # and load C data, shape: (J,)
        region = f['/B/ref/C/ref_region'], shape: (M,)
        mask = a2b_ref.mask.ravel() # use the mask that comes along from the previous dereferencing, shape: (n*l,)
    )
    a2b2c.shape # (n*l,k), where k is the max number of a->c references
    a2b2c.reshape(b2a_ref.shape,-1).shape # (n,l,k), broadcast-able back into a2b

This can be repeated many times to access ``B -> A -> C -> D -> ...`` references.

An additional helper function ``dereference_chain`` is provided to make this easier.::

    from h5flow.data import dereference_chain

    sel = slice(0, 1000) # indices of A, shape: (n,)
    refs = [f['/A/ref/B/ref'], f['/B/ref/C/ref']] # chain of references to load (A->B,B->C)
    regions = [f['/A/ref/B/ref_region'], f['/B/ref/C/ref_region']] # lookup regions (for A and B)
    ref_dir = [(0,1),(0,1)] # reference direction to use for each reference (defaults to (0,1))

    a2b2c = dereference_chain(sel, refs, f['/C/data'], region=regions, ref_directions=ref_dir)
    a2b2c.shape # (n,l,k)

h5flow workflow
===============

There are four central components of an ``h5flow`` workflow:
    1. the manager
    2. the generator
    3. stages
    4. the data manager

The manager (see documentation under ``h5flow.core.h5flow_manager``) initializes
components of the workflow (namely, the generator, stages, and the data manager),
and then executes their methods in order:

    1. ``generator.init``
    2. ``stage.init`` (in sequence specified in the flow)
    3. ``generator.run`` (until all processes return ``H5FlowGenerator.EMPTY``)
    4. ``stage.run``
    5. ``generator.finish``
    6. ``stage.finish``

The ``init`` stage creates datasets in the output file and configures each
component for the loop.

The ``run`` stage performs calculations on subsets of the input dataset and
write new data back to the file.

The ``finish`` stage allows components to flush any lingering data in memory to
the data files or finalize and complete any summary calculations.

The generator (see documentation under ``h5flow.core.h5flow_generator``) provides
slices into a source dataset for each stage to execute on. Custom generators can
be written to convert datatypes or generate new datasets, or ``h5flow`` provides
a built-in "loop generator" that can be used to iterate across an existing
dataset in an efficient manner.

Stages are custom, user-built algorithms that take slices into a source dataset
and perform a specific calculation on that slice, typically writing new data into
a different dataset in the hdf5 file.

In order to make the most use of parallel file access provided by ``h5flow`` a
workflow should meet the following requirements:

    1. source dataset slices are `fully` independent of each other
    2. input and output datasets have only 1 dimension (the loop dimension). Note that this does not preclude using compound datatypes with more than one dimension, i.e. ``dset.shape == (N,)`` and ``dset.dtype == [('values','i8(100,')]`` is allowed.

configuration
-------------

``h5flow`` uses a yaml config file to define the workflow. The main definition of
the workflow is defined under the ``flow`` key::

    flow:
        source: <dataset to loop over, or generator name>
        stages: [<first sequential stage name>, <second sequential stage name>]
        drop: [<dataset name, opt.>]

The ``source`` defines the loop source dataset. By default, you may specify an
existing dataset and an ``H5FlowDatasetLoopGenerator`` will be used. ``stages``
defines the names and sequential order of the analysis stages should be executed
on each data chunk provided by the generator. Optionally, ``drop`` defines a list
of dataset paths to save in a temporary file to be deleted at the end of the
workflow.

``h5flow`` also uses `pyyaml-include <https://pypi.org/project/pyyaml-include/>`_
allowing for some simple inheritance from other configuration files in the
current working directory.

generators
~~~~~~~~~~

To define a generator, specify the name, an ``H5FlowGenerator``-inheriting
classname, along with any desired parameters at the top level within the yaml
file::

    dummy_generator:
        classname: DummyGenerator
        dset_name: <dataset to be accessed by each stage>
        params:
            dummy_param: value

For both generators and stages, classes will be discovered for within the
current directory, the ``./h5flow_modules/`` directory, or the ``h5flow/modules``
directory (in that order) and automatically loaded upon runtime.

stages
~~~~~~

To define a stage, specify the name, an ``H5FlowStage``-inheriting classname, along
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

You can also specify specific datasets to load that is linked to the current
loop dataset with the ``requires`` field::

    dummy_stage_requires:
        classname: DummyStage
        requires:
            - <path to a dataset that has source <-> dset references>
            - <path to a second dataset with source <-> dset references>

This will load a ``numpy`` masked array into the ``cache`` under a key of the
same path.

You can specify complex linking paths to load data from references to references
(or references to references to references ...) by specifying a path and a
name::

    dummy_stage_complex_requires:
        classname: DummyStage
        requires:
            - name: <name to use in the cache>
              path: [<path to first dataset>, <path to second dataset>, ...]

which will load the data at ``source -> <first dataset> -> <second dataset>``.

Finally, you can also indicate if you just want to load an index into the final
dataset (rather than the data) with the ``index_only`` flag::

    dummy_stage_index_requires:
        classname: DummyStage
        requires:
            - name: <name to use in cache>
              path: [<first dataset>, <second dataset>]
              index_only: True

resources
~~~~~~~~~

Occasionally, workflow-level, read-only data is needed to be accessed across
multiple stages. For this, an ``H5FlowResource``-inheriting class can be
implemented. Resources can be declared under the ``resources`` field at the top-
level of the configuration yaml::

    resources:
         - classname: DummyResource
           params:
                example_parameter: 'example'

These objects can be accessed within a workflow source via their classname::

    from h5flow.core import resources

    resources['DummyResource'] # access the DummyResource

It is important to note that only one instance of a given resource class is
allowed. Each resource is provided all runtime options and thus can load or
create data that depends on the input file, dataset selection, or output file.

writing an ``H5FlowStage``
==========================

Any ``H5FlowStage``-inheriting class has 4 main components:
    1. a constructor (``__init__()``)
    2. class attributes
    3. an initialization ``init()`` method
    4. and a ``run()`` method


None of the methods are required for the class to function within ``h5flow``, but
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

writing an ``H5FlowGenerator``
==============================

I haven't written this section yet... but in the meantime you can examine the
docstrings of ``h5flow.core.h5_flow_generator``.

writing an ``H5FlowResource``
=============================

I haven't written this section yet... but in the meantime you can examine the
docstrings of ``h5flow.core.h5_flow_resource``.
