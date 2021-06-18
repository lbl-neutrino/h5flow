import h5py
import h5py.h5s as h5s
import h5py.h5r as h5r
import numpy as np
import logging

from .. import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI

from .lib import ref_region_dtype

class H5FlowDataManager(object):
    '''
        Coordinates access to the output data file across multiple processes.

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
        self.mpi_flag = H5FLOW_MPI

        self.comm = MPI.COMM_WORLD if H5FLOW_MPI else None
        self.rank = self.comm.Get_rank() if H5FLOW_MPI else 0
        self.size = self.comm.Get_size() if H5FLOW_MPI else 1

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
            Force underlying hdf5 resource to close

        '''
        if self._fh is not None and self._fh:
            self._fh.close()

    @property
    def fh(self):
        '''
            Direct access to the underlying h5py ``File`` object. Not recommended
            for writing to datasets (see ``reserve_data``, ``write_data``, and
            ``write_ref``).

        '''
        if self._fh is None or not self._fh:
            self._open_file(mpi=self.mpi_flag)
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
        return (f'{parent_dataset_name}/ref/{child_dataset_name}/ref' in self.fh \
                or f'{child_dataset_name}/ref/{parent_dataset_name}/ref' in self.fh) \
            and f'{parent_dataset_name}/ref/{child_dataset_name}/region' in self.fh \
            and f'{child_dataset_name}/ref/{parent_dataset_name}/region' in self.fh \

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
            Get references of ``parent_dataset_name -> child_dataset_name``

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``tuple`` of ``h5py.Dataset``, reference direction; e.g. ``(stage0/obj0/ref/stage0/obj1/ref, (0,1))``

        '''
        if f'{parent_dataset_name}/ref/{child_dataset_name}/ref' in self.fh:
            dset = self.fh[f'{parent_dataset_name}/ref/{child_dataset_name}/ref']
            return dset, (0,1)
        dset = self.fh[f'{child_dataset_name}/ref/{parent_dataset_name}/ref']
        return dset, (1,0)

    def get_refs(self, dataset_name):
        '''

        '''
        reg_regions = list()
        self.fh.visititems(lambda n,d:
            reg_regions.append(d) \
            if isinstance(d,h5py.Dataset) and n.endswith('/ref_region') \
            else None
            )
        return [self.fh[d.attrs['ref']] for d in reg_regions]

    def get_ref_region(self, parent_dataset_name, child_dataset_name):
        '''
            Get reference lookup regions for ``parent_dataset_name -> child_dataset_name``

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``h5py.Dataset``, ``stage0/obj0/ref/stage0/obj1/ref_region``, (0,1)

        '''
        return self.fh[f'{parent_dataset_name}/ref/{child_dataset_name}/ref_region']


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
            exist. Both datasets must already exist.

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

        '''
        child_path = f'{child_dataset_name}/ref/{parent_dataset_name}'
        if child_path + '/ref' in self.fh:
            raise RuntimeError(f'References for {parent_dataset_name}->{child_dataset_name} already exist under {child_path}')
        path = f'{parent_dataset_name}/ref/{child_dataset_name}'
        if path not in self.fh:
            if f'{parent_dataset_name}/ref' not in self.fh:
                self.fh.create_group(f'{parent_dataset_name}/ref')

            self.fh.create_dataset(path+'/ref', shape=(0,2), maxshape=(None,2),
                dtype='u8')
            self.fh[path+'/ref'].attrs['dset0'] = self.get_dset(parent_dataset_name).ref
            self.fh[path+'/ref'].attrs['dset1'] = self.get_dset(child_dataset_name).ref

            parent_dset = self.get_dset(parent_dataset_name)
            child_dset = self.get_dset(child_dataset_name)
            self.fh.create_dataset(path+'/ref_region', shape=(len(parent_dset),), maxshape=(None,),
                dtype=ref_region_dtype, fillvalue=np.zeros((1,), dtype=ref_region_dtype))
            self.fh.create_dataset(child_path+'/ref_region', shape=(len(child_dset),), maxshape=(None,),
                dtype=ref_region_dtype, fillvalue=np.zeros((1,), dtype=ref_region_dtype))
            self.fh[path+'/ref_region'].attrs['ref'] = self.fh[path+'/ref'].ref
            self.fh[child_path+'/ref_region'].attrs['ref'] = self.fh[path+'/ref'].ref

            self.fh[path+'/ref'].attrs['ref_region0'] = self.fh[f'{path}/ref_region'].ref
            self.fh[path+'/ref'].attrs['ref_region1'] = self.fh[f'{child_path}/ref_region'].ref

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
        grp = self.fh[dataset_name]
        dset = self.get_dset(dataset_name)
        curr_len = len(dset)
        specs = self.comm.allgather(spec) if H5FLOW_MPI else [spec]
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            n = sum(specs)
            dset.resize((curr_len + n,))
            for ref in self.get_refs(dataset_name):
                self.fh[ref.attrs['ref_region0']].resize((len(self.fh[ref.attrs['dset0']]),))
                self.fh[ref.attrs['ref_region1']].resize((len(self.fh[ref.attrs['dset1']]),))

            rv = slice(curr_len + sum(specs[:self.rank]), curr_len + sum(specs[:self.rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([spec.stop for spec in specs])
            if new_size > curr_len:
                dset.resize((new_size,))
                for ref in self.get_refs(dataset_name):
                    self.fh[ref.attrs['ref_region0']].resize((len(self.fh[ref.attrs['dset0']]),))
                    self.fh[ref.attrs['ref_region1']].resize((len(self.fh[ref.attrs['dset1']]),))
            rv = spec
        else:
            raise TypeError(f'spec {spec} is not a valid type, must be slice or integer')
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

    def _update_ref_region(self, region_dset, sel, ref_arr, ref_offset):
        # Note:: ref_arr is the 1D array of indices into region_dset to update, ref_offset is where ref_array is positioned within a larger ref dataset
        max_length = int(np.max(self.comm.allgather(sel.stop))) if H5FLOW_MPI else sel.stop
        if len(region_dset) < max_length:
            region_dset.resize((max_length,))
        region = region_dset[sel]

        _,idcs,start_idcs = np.intersect1d(np.r_[sel],ref_arr,return_indices=True)
        start = np.zeros(len(region), dtype='i8')
        start[idcs] = ref_offset + start_idcs
        region_dset[sel,'start'] = np.where(
            region['start'] != region['stop'],
            np.minimum(region['start'], start),
            start
            )

        _,idcs,stop_idcs = np.intersect1d(np.r_[sel],ref_arr[::-1],return_indices=True)
        stop = np.zeros(len(region), dtype='i8')
        stop[idcs] = ref_offset + len(ref_arr) - stop_idcs
        region_dset[sel,'stop'] = np.where(
            region['start'] != region['stop'],
            np.maximum(region['stop'], stop),
            stop
            )

    def write_ref(self, parent_dataset_name, child_dataset_name, refs):
        '''
            Add refs for ``parent_dataset_name -> child_dataset_name``. Note
            that references are never updated and can't be removed after they
            are created.

            :param refs: an integer array of shape (N,2) with refs[:,0] corresponding to the index in the parent dataset and refs[:,1] corresponding to the index in the child dataset

        '''
        ns = self.comm.allgather(len(refs)) if H5FLOW_MPI else [len(refs)]

        ref_dset, ref_dir = self.get_ref(parent_dataset_name, child_dataset_name)
        ref_offset = len(ref_dset) + sum(ns[:self.rank])
        ref_dset.resize((len(ref_dset) + sum(ns), 2))
        ref_dset[ref_offset:ref_offset + ns[self.rank]] = refs[:,ref_dir]

        parent_ref_region_dset = self.get_ref_region(parent_dataset_name, child_dataset_name)
        child_ref_region_dset = self.get_ref_region(child_dataset_name, parent_dataset_name)

        if len(refs):
            parent_sel = slice(int(np.min(refs[:,0])), int(np.max(refs[:,0])+1))
            child_sel = slice(int(np.min(refs[:,1])), int(np.max(refs[:,1])+1))
        else:
            parent_sel = slice(0,0)
            child_sel = slice(0,0)

        self._update_ref_region(parent_ref_region_dset, parent_sel, refs[:,0], ref_offset)
        self._update_ref_region(child_ref_region_dset, child_sel, refs[:,1], ref_offset)
