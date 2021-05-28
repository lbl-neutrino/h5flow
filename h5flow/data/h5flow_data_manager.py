import h5py
import h5py.h5s as h5s
import h5py.h5r as h5r
import numpy as np
from mpi4py import MPI
import logging

class H5FlowDataManager(object):
    '''
        The ``H5FlowDataManager`` class coordinates access to the output data
        file across multiple processes.

        To initialize::

            hfdm = H5FlowDataManager(<path to file>)

        Opening and closing the underlying resource is handled automatically when
        using the dedicated file access API, e.g.::

            hfdm.dset_exists(...)
            hfdm.create_dset(...)
            hfdm.get_ref(...)
            hfdm.reserve_data(...)
            hfdm.write_ref(...)
            ...

    '''

    def __init__(self, filepath):
        self.filepath = filepath
        self._fh = None
        self.mpi_flag = True

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def _open_file(self, mpi=True):
        if (self._fh is not None or mpi != self.mpi_flag) and self._fh:
            self.close_file()
        if mpi:
            self._fh = h5py.File(self.filepath, 'a', libver='latest', driver='mpio', comm=self.comm)
        else:
            if self.rank == 0:
                self._fh = h5py.File(self.filepath, 'a', libver='latest')
            else:
                self._fh = h5py.File(self.filepath, 'r', libver='latest', swmr=True)
        self.mpi_flag = mpi

    def close_file(self):
        '''
            Force underlying resource to close

        '''
        if self._fh is not None and self._fh:
            self._fh.close()

    @property
    def fh(self):
        '''
            Direct access to the underlying h5py ``File`` object. Not recommended
            for writing to datasets.

        '''
        if self._fh is None or not self._fh:
            self._open_file()
        return self._fh

    def delete(self, name):
        '''
            Delete object at ``name``

            :param name: ``str`` path to dataset to be deleted

        '''
        if name in self.fh:
            del self.fh[name]

    def dset_exists(self, dataset_name):
        '''
            Check if data object of ``dataset_name`` exists

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :returns: ``True`` if data object exists

        '''
        return f'{dataset_name}/data' in self.fh

    def ref_exists(self, parent_dataset_name, child_dataset_name):
        '''
            Check if references for ``parent_dataset_name -> child_dataset_name`` exists

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``True`` if references exists

        '''
        return f'{parent_dataset_name}/ref/{child_dataset_name}/ref' in self.fh \
            and f'{parent_dataset_name}/ref/{child_dataset_name}/ref_valid' in self.fh

    def attr_exists(self, name, key):
        '''
            Check if attribute ``key`` exists for ``name``

            :param name: ``str`` path to object, e.g. ``stage0/obj0`` or ``stage0``

            :param key: ``str`` attribute name

            :returns: ``True`` if attribute exists

        '''
        if f'{name}' in self.fh:
            return key in self.fh[f'{name}'].attrs
        return False

    def get_dset(self, dataset_name):
        '''
            Get dataset of ``dataset_name``

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :returns: ``h5py.Dataset``, e.g. ``stage0/obj0/data``

        '''
        dset = self.fh[f'{dataset_name}/data']
        return dset

    def get_attrs(self, name):
        '''
            Get attributes of ``name``

            :param name: ``str`` path to object, e.g. ``stage0``

            :returns: ``h5py.AttributeManager``

        '''
        return self.fh[f'{name}'].attrs

    def get_ref(self, parent_dataset_name, child_dataset_name):
        '''
            Get refrences of ``parent_dataset_name -> child_dataset_name``

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``h5py.Dataset``, e.g. ``stage0/obj0/ref/stage0/obj1/ref``

        '''
        dset = self.fh[f'{parent_dataset_name}/ref/{child_dataset_name}/ref']
        return dset

    def get_ref_valid(self, parent_dataset_name, child_dataset_name):
        '''
            Get refrences of ``parent_dataset_name -> child_dataset_name``

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``h5py.Dataset``, e.g. ``stage0/obj0/ref/stage0/obj1/ref_valid``

        '''
        dset = self.fh[f'{parent_dataset_name}/ref/{child_dataset_name}/ref_valid']
        return dset

    def set_attrs(self, name, **attrs):
        '''
            Update attributes of ``name``. Attribute ``key: value`` are passed
            in as additional keyword arguments

            :param name: ``str`` path to object, e.g. ``stage0``

        '''
        if name not in self.fh:
            self.fh.create_group(name)
        for key,val in attrs.items():
            self.fh[name].attrs[key] = val

    def create_dset(self, dataset_name, dtype):
        '''
            Create a 1D dataset of ``dataset_name`` with datatype ``dtype``, if
            it doesn't already exist

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param dtype: ``np.dtype`` of dataset, can be a structured dtype

        '''
        path = f'{dataset_name}/data'
        if path not in self.fh:
            self.fh.create_dataset(path, (0,), maxshape=(None,),
                dtype=dtype)

    def create_ref(self, parent_dataset_name, child_dataset_name):
        '''
            Create a 1D dataset of references of
            ``parent_dataset_name -> child_dataset_name``, if it doesn't already
            exist

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

        '''
        path = f'{parent_dataset_name}/ref/{child_dataset_name}/ref'
        if path not in self.fh:
            if f'{parent_dataset_name}/ref' not in self.fh:
                self.fh.create_group(f'{parent_dataset_name}/ref')
                self.fh[f'{parent_dataset_name}/ref'].attrs['parent'] = self.fh[f'{parent_dataset_name}/data'].ref
            self.fh.create_dataset(path, (0,), maxshape=(None,),
                dtype=h5py.regionref_dtype)
            self.fh.create_dataset(path+'_valid', (0,), maxshape=(None,),
                dtype='u1')
            self.fh[path].attrs['child'] = self.fh[f'{child_dataset_name}/data'].ref
            self.fh[path+'_valid'].attrs['child'] = self.fh[f'{child_dataset_name}/data'].ref

    def reserve_data(self, dataset_name, spec):
        '''
            Coordinate access into ``dataset_name``. Depending on the type of
            ``spec`` a different access mode will be performed:

                - ``int``: access in append mode - will grant access to ``spec`` rows at the end of the dataset
                - ``slice`` or list of ``int`` or list of ``slice``: access a specific section(s) of the dataset - will resize dataset if section does not exist

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param spec: see function description

            :returns: ``slice`` into ``dataset_name`` where access is given

        '''
        dset = self.get_dset(dataset_name)
        curr_len = len(dset)
        specs = self.comm.allgather(spec)
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            dset.resize((curr_len + sum(specs),))
            rv = slice(curr_len + sum(specs[:self.rank]), curr_len + sum(specs[:self.rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([spec.stop for spec in specs])
            if new_size > curr_len:
                dset.resize((new_size,))
            rv = spec
        else:
            raise TypeError(f'spec {spec} is not a valid type')
        return rv

    def write_data(self, dataset_name, spec, data):
        '''
            Write ``data`` into ``dataset_name`` at ``spec``

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param spec: ``slice`` into ``dataset_name`` to write ``data``

            :param data: numpy array or iterable to write

        '''
        dset = self.get_dset(dataset_name)
        dset[spec] = data

    @staticmethod
    def merge_region_specs(*specs):
        '''
            Join a region specification into one of:
             - a slice
             - an array of integers (shape: (N, 1))

            Follows the ``numpy.c_[...]`` behavior except prunes for unique values
            and sorts. If indices can be simplified into a ``slice`` selection,
            this method returns a slice. E.g.::

                merge_region_specs(slice(0,10), slice(10,20)) # returns slice(0,20,1)
                merge_region_specs(0,[1,2],3) # returns slice(0,4,1)
                merge_region_specs(slice(0,4,2),[2,4,6],8,2) # returns slice(0,10,2)
                merge_region_specs(slice(0,5),10) # returns np.array([0,1,2,3,4,10])

            Note that empty iterables or empty slices produce empty slices::

                merge_region_specs([], slice(0,0,1)) # returns slice(0,0,1)

            :param specs: a collection of ``int``, ``slice``, iterable of ``int``, or iterable of ``slice`` to join

        '''
        flat_specs = []
        for s in specs:
            if isinstance(s,np.integer) or isinstance(s,int) or isinstance(s,slice):
                flat_specs.append(s)
            else:
                flat_specs.extend(s)

        if len(flat_specs) > 1:
            idcs = np.sort(np.unique(np.r_[(*flat_specs,)].astype(int)))
        elif len(flat_specs):
            idcs = np.sort(np.unique(np.r_[flat_specs[0]].astype(int)))
        else:
            return np.empty((0,1), dtype=int)

        if len(idcs) == 0:
            # refers to no region => default to empty slice behavior
            return np.empty((0,1), dtype=int)
        elif len(idcs) <= 2:
            # only a few => return array
            return idcs[:,np.newaxis]

        step = np.diff(idcs, axis=0)
        if np.all(np.abs(step) == step[0]):
            # can be reduced to a simple slice
            return slice(idcs[0],idcs[-1]+step[0],step[0])
        return idcs[:,np.newaxis]

    @staticmethod
    def create_ref_array(dset, spec):
        array = np.empty((len(spec),), dtype=h5py.regionref_dtype)
        space = h5s.create_simple(dset.id.shape)
        for i in range(len(spec)):
            item = spec[i]
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stride = item.step if item.step is not None else 1
                count = (item.stop-start)//stride if item.stop is not None else (dset.shape[0]-start)//stride
                if count == 0:
                    continue
                space.select_hyperslab((start,), (count,), (stride,))
            elif isinstance(item,int) or isinstance(item,np.integer):
                space.select_hyperslab((item,), (1,))
            elif len(item) == 0:
                continue
            else:
                space.select_elements(item)
            array[i] = h5r.create(dset.id, b'.', h5r.DATASET_REGION, space)
        return array

    def reserve_ref(self, parent_dataset_name, child_dataset_name, spec):
        '''
            Coordinate access into ``parent_dataset_name -> child_dataset_name``
            references. Depending on the type of
            ``spec`` a different access mode will be performed:

                - ``int``: access in append mode - will grant access to ``spec`` rows at the end of the dataset
                - ``slice`` or iterable of ``int`` or iterable of ``slice``: access a specific section - will resize dataset if section does not exist

            Note: access pattern (append or selection) must be the same in all processes

            :param parent_dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param spec: see function description

            :returns: ``slice`` into ``parent_dataset_name/ref/child_dataset_name`` where access is given

        '''
        dset = self.get_ref(parent_dataset_name, child_dataset_name)
        dset_valid = self.get_ref_valid(parent_dataset_name, child_dataset_name)
        curr_len = len(dset)
        if isinstance(spec, slice):
            all_specs = self.comm.allgather([spec])
        else:
            all_specs = self.comm.allgather(spec)
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            dset.resize((curr_len + sum(all_specs),))
            dset_valid.resize((curr_len + sum(all_specs),))
            rv = slice(curr_len + sum(all_specs[:self.rank]), curr_len + sum(all_specs[:self.rank+1]))
        else:
            try:
                # maybe create up to a specific chunk of the dataset
                idcs = np.r_[self.merge_region_specs(*[s for ss in all_specs for s in ss])]+1
                if len(idcs):
                    new_size = np.max(idcs)
                    if new_size > curr_len:
                        dset.resize((new_size,))
                        dset_valid.resize((new_size,))
                rv = spec
            except TypeError:
                raise TypeError(f'spec {spec} is not a valid specification')
        return rv

    def write_ref(self, parent_dataset_name, child_dataset_name, spec, refs):
        '''
            Write refs into ``parent_dataset_name -> child_dataset_name`` at
            ``spec``. As an example of the usage, to write two references
            referring to the 0 position and the first 10 of ``stage0/obj1``,
            respectively, one would call::

                hfdm.write_ref('stage0/obj0', 'stage0/obj1', slice(0,2), [0, slice(0,10)])

            :param parent_dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to dataset, e.g. ``stage0/obj1``

            :param spec: ``slice`` into ``parent_dataset_name/ref/child_dataset_name`` to write ``refs``

            :param refs: an iterable with same length specified by ``spec`` of selections into the child dataset

        '''
        self.close_file()
        self.comm.barrier()

        clean_ref = [r if isinstance(r,slice) or isinstance(r,int) or isinstance(r,np.integer) \
            else self.merge_region_specs(*r) for r in refs]
        valid = np.array([isinstance(r,int) or isinstance(r,np.integer) or (isinstance(r,slice) and r.start != r.stop) or (not isinstance(r,slice) and len(r) > 0) for r in clean_ref], dtype=bool)

        if self.rank == 0:
            self._open_file(mpi=False)
            dset = self.get_ref(parent_dataset_name, child_dataset_name)
            dset_valid = self.get_ref_valid(parent_dataset_name, child_dataset_name)
            child_dset = self.get_dset(child_dataset_name)

            for r in range(self.size):
                if r != self.rank:
                    spec, clean_ref, valid = self.comm.recv(source=r)

                dset[spec] = self.create_ref_array(child_dset, clean_ref)# np.array([child_dset.regionref[r] for r in clean_ref], dtype=h5py.regionref_dtype)
                dset_valid[spec] = valid

            self.close_file()
            self.comm.barrier()
        else:
            self.comm.send((spec, clean_ref, valid), dest=0)
            self.comm.barrier()
