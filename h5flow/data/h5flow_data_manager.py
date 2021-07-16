import h5py
import numpy as np
import logging
import os
import time

from .. import H5FLOW_MPI
if H5FLOW_MPI:
    from mpi4py import MPI

from .lib import ref_region_dtype

def cached_function(f):
    cache = dict()
    def new_f(*args, **kwargs):
        h = str((args, kwargs))
        if h in cache:
            return cache[h]
        else:
            rv = f(*args, **kwargs)
            cache[h] = rv
            return rv
    return new_f

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
    _temp_filename_fmt = 'tmp-%y.%m.%d-%H.%M.%S-h5flow.h5'

    def __init__(self, filepath, drop_list=None):
        self.filepath = filepath
        self._fh = None
        self._temp_fh = None
        self.mpi_flag = H5FLOW_MPI

        if drop_list:
            self.drop_list = drop_list
            self._temp_filepath = os.path.join(os.path.dirname(self.filepath),
                time.strftime(self._temp_filename_fmt))
            logging.info(f'writing temporary data to {self._temp_filepath}')
        else:
            self.drop_list = list()
            self._temp_filepath = None

        self.comm = MPI.COMM_WORLD if H5FLOW_MPI else None
        self.rank = self.comm.Get_rank() if H5FLOW_MPI else 0
        self.size = self.comm.Get_size() if H5FLOW_MPI else 1

    def finish(self):
        '''
            Deletes datasets specified in the drop_list before closing file
            handle.

        '''
        for path in self.drop_list:
            self.delete(path)
        self.close_file()

        if H5FLOW_MPI:
            self.comm.barrier()
        if self._temp_filepath and self.rank == 0:
            os.remove(self._temp_filepath)

    def _open_file(self, mpi=True):
        if (self._fh is not None or mpi != self.mpi_flag) and self._fh:
            # close file if mpi mode changes
            self.close_file()

        if mpi:
            # open file with mpi enabled
            self._fh = h5py.File(self.filepath, 'a', libver='latest', driver='mpio', comm=self.comm)
            if self._temp_filepath:
                self._temp_fh = h5py.File(self._temp_filepath, 'a', libver='latest', driver='mpio', comm=self.comm)
        else:
            # open file without mpi enabled
            if self.rank == 0:
                self._fh = h5py.File(self.filepath, 'a', libver='latest')
                if self._temp_filepath:
                    self._temp_fh = h5py.File(self.filepath, 'a', libver='latest')
            else:
                self._fh = h5py.File(self.filepath, 'r', libver='latest', swmr=True)
                if self._temp_filepath:
                    self._temp_fh = h5py.File(self.filepath, 'r', libver='latest', swmr=True)

        # update mpi flag
        self.mpi_flag = mpi

    def close_file(self):
        '''
            Force underlying hdf5 resource to close

        '''
        if self._fh is not None and self._fh:
            self._fh.close()
        if self._temp_fh is not None and self._temp_fh:
            self._temp_fh.close()

    @property
    def fh(self):
        '''
            Direct access to the underlying h5py ``File`` object. Not recommended
            for use. Instead, use ``get_dset(...)``, ``write_data(...)``, etc.

        '''
        if self._fh is None or not self._fh:
            self._open_file(mpi=self.mpi_flag)
        return self._fh

    @cached_function
    def _route_fh(self, path):
        '''
            Return file handle to temp file or output file depending on if
            path is in drop list. Result is cached such that subsequent calls
            return the same handle.

        '''
        if path in self.fh:
            # if it already exists in output file, do not use temp file
            return self.fh
        elif any([d in path for d in self.drop_list]):
            return self._temp_fh
        else:
            return self.fh

    def delete(self, name):
        '''
            Delete object at and references to ``name``. Ignored if path is
            in temp file.

            :param name: ``str`` path to dataset to be deleted

        '''
        for ref in self.get_refs(name):
            fh = self._route_fh(ref.attrs['ref_region0'])
            if ref.attrs['ref_region0'] in fh and fh is not self._temp_fh:
                del fh[ref.attrs['ref_region0'][:-10]] # remove reference group
            fh = self._route_fh(ref.attrs['ref_region1'])
            if ref.attrs['ref_region1'] in fh and fh is not self._temp_fh:
                del fh[ref.attrs['ref_region1'][:-10]] # remove reference group
        fh = self._route_fh(name)
        if name in fh and fh is not self._temp_fh:
            del fh[name] # remove object group

    def dset_exists(self, dataset_name):
        '''
            Check if data object of ``dataset_name`` exists

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :returns: ``True`` if data object exists

        '''
        return f'{dataset_name}/data' in self._route_fh(f'{dataset_name}/data')

    def ref_exists(self, parent_dataset_name, child_dataset_name):
        '''
            Check if references for ``parent_dataset_name -> child_dataset_name`` exists

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``True`` if references exists

        '''
        return (f'{parent_dataset_name}/ref/{child_dataset_name}/ref' in self._route_fh(f'{parent_dataset_name}/ref/{child_dataset_name}/ref') \
                or f'{child_dataset_name}/ref/{parent_dataset_name}/ref' in self._route_fh(f'{child_dataset_name}/ref/{parent_dataset_name}/ref')) \
            and f'{parent_dataset_name}/ref/{child_dataset_name}/ref_region' in self._route_fh(f'{parent_dataset_name}/ref/{child_dataset_name}/ref_region') \
            and f'{child_dataset_name}/ref/{parent_dataset_name}/ref_region' in self._route_fh(f'{child_dataset_name}/ref/{parent_dataset_name}/ref_region') \

    def attr_exists(self, name, key):
        '''
            Check if attribute ``key`` exists for ``name``

            :param name: ``str`` path to object, e.g. ``stage0/obj0`` or ``stage0``

            :param key: ``str`` attribute name

            :returns: ``True`` if attribute exists

        '''
        if f'{name}' in self._route_fh(name):
            return key in self._route_fh(name)[f'{name}'].attrs
        return False

    def get_dset(self, dataset_name):
        '''
            Get dataset of ``dataset_name``

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :returns: ``h5py.Dataset``, e.g. ``stage0/obj0/data``

        '''
        dset = self._route_fh(f'{dataset_name}/data')[f'{dataset_name}/data']
        return dset

    def get_attrs(self, name):
        '''
            Get attributes of ``name``

            :param name: ``str`` path to object, e.g. ``stage0``

            :returns: ``h5py.AttributeManager``

        '''
        return self._route_fh(name)[f'{name}'].attrs

    def get_ref(self, parent_dataset_name, child_dataset_name):
        '''
            Get references of ``parent_dataset_name -> child_dataset_name``

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``tuple`` of ``h5py.Dataset``, reference direction; e.g. ``(stage0/obj0/ref/stage0/obj1/ref, (0,1))``

        '''
        path = f'{parent_dataset_name}/ref/{child_dataset_name}/ref'
        fh = self._route_fh(path)
        if path in fh:
            dset = fh[path]
            return dset, (0,1)
        path = f'{child_dataset_name}/ref/{parent_dataset_name}/ref'
        dset = self._route_fh(path)[path]
        return dset, (1,0)

    def get_refs(self, dataset_name):
        '''
            Get all references involving ``dataset_name -> other``

        '''
        fh = self._route_fh(dataset_name)
        reg_regions = list()
        if dataset_name not in fh:
            return list()
        fh[dataset_name].visititems(lambda n,d:
            reg_regions.append(d) \
            if isinstance(d,h5py.Dataset) and n.endswith('/ref_region') \
            else None
            )
        return [self._route_fh(d.attrs['ref'])[d.attrs['ref']] for d in reg_regions \
            if d.attrs['ref'] in self._route_fh(d.attrs['ref'])]

    def get_ref_region(self, parent_dataset_name, child_dataset_name):
        '''
            Get reference lookup regions for ``parent_dataset_name -> child_dataset_name``

            :param parent_dataset_name: ``str`` path to parent dataset, e.g. ``stage0/obj0``

            :param child_dataset_name: ``str`` path to child dataset, e.g. ``stage0/obj1``

            :returns: ``h5py.Dataset``, ``stage0/obj0/ref/stage0/obj1/ref_region``, (0,1)

        '''
        path = f'{parent_dataset_name}/ref/{child_dataset_name}/ref_region'
        return self._route_fh(path)[path]


    def set_attrs(self, name, **attrs):
        '''
            Update attributes of ``name``. Attribute ``key: value`` are passed
            in as additional keyword arguments

            :param name: ``str`` path to object, e.g. ``stage0``

        '''
        fh = self._route_fh(name)
        if name not in fh:
            fh.create_group(name)
        for key,val in attrs.items():
            fh[name].attrs[key] = val

    def create_dset(self, dataset_name, dtype):
        '''
            Create a 1D dataset of ``dataset_name`` with datatype ``dtype``, if
            it doesn't already exist

            :param dataset_name: ``str`` path to dataset, e.g. ``stage0/obj0``

            :param dtype: ``np.dtype`` of dataset, can be a structured dtype

        '''
        path = f'{dataset_name}/data'

        fh = self._route_fh(path)
        if path not in fh:
            fh.require_dataset(path, (0,), maxshape=(None,),
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
        if child_path + '/ref' in self._route_fh(child_path):
            raise RuntimeError(f'References for {parent_dataset_name}->{child_dataset_name} already exist under {child_path}')
        path = f'{parent_dataset_name}/ref/{child_dataset_name}'

        fh = self._route_fh(path)
        if path not in fh:
            # create reference group, if not present
            if f'{parent_dataset_name}/ref' not in fh:
                fh.create_group(f'{parent_dataset_name}/ref')

            # create bi-directional reference dataset
            fh.require_dataset(path+'/ref', shape=(0,2), maxshape=(None,2),
                dtype='u8')
            # link to source datasets
            fh[path+'/ref'].attrs['dset0'] = self.get_dset(parent_dataset_name).name
            fh[path+'/ref'].attrs['dset1'] = self.get_dset(child_dataset_name).name

            # create lookup table dataset
            parent_dset = self.get_dset(parent_dataset_name)
            child_dset = self.get_dset(child_dataset_name)
            child_fh = self._route_fh(child_path)
            fh.require_dataset(path+'/ref_region', shape=(len(parent_dset),), maxshape=(None,),
                dtype=ref_region_dtype, fillvalue=np.zeros((1,), dtype=ref_region_dtype))
            child_fh.require_dataset(child_path+'/ref_region', shape=(len(child_dset),), maxshape=(None,),
                dtype=ref_region_dtype, fillvalue=np.zeros((1,), dtype=ref_region_dtype))

            # link to references
            fh[path+'/ref_region'].attrs['ref'] = fh[path+'/ref'].name
            child_fh[child_path+'/ref_region'].attrs['ref'] = fh[path+'/ref'].name
            # link back to lookup tables
            fh[path+'/ref'].attrs['ref_region0'] = fh[f'{path}/ref_region'].name
            fh[path+'/ref'].attrs['ref_region1'] = child_fh[f'{child_path}/ref_region'].name

    def _resize_dset(self, dset, new_shape):
        dset.resize(new_shape)

        if dset.name.endswith('/data'):
            for ref in self.get_refs(dset.name[:-5]):
                dset0 = ref.attrs['dset0']
                dset1 = ref.attrs['dset1']
                self._resize_dset(
                    self._route_fh(ref.attrs['ref_region0'])[ref.attrs['ref_region0']],
                    (len(self._route_fh(dset0)[dset0]),)
                    )
                self._resize_dset(
                    self._route_fh(ref.attrs['ref_region1'])[ref.attrs['ref_region1']],
                    (len(self._route_fh(dset1)[dset1]),)
                    )

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
        specs = self.comm.allgather(spec) if H5FLOW_MPI else [spec]
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            n = sum(specs)
            self._resize_dset(dset, (curr_len + n,))
            rv = slice(curr_len + sum(specs[:self.rank]), curr_len + sum(specs[:self.rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([spec.stop for spec in specs])
            if new_size > curr_len:
                self._resize_dset(dset, (new_size,))
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
            self._resize_dset(region_dset, (max_length,))
        region = region_dset[sel]

        _,idcs,start_idcs = np.intersect1d(np.r_[sel],ref_arr,return_indices=True)
        start = np.zeros(len(region), dtype='i8')
        start[idcs] = ref_offset + start_idcs
        start = np.where(
            region['start'] != region['stop'],
            np.minimum(region['start'], start),
            start
            )
        region_dset[sel,'start'] = start

        _,idcs,stop_idcs = np.intersect1d(np.r_[sel],ref_arr[::-1],return_indices=True)
        stop = np.zeros(len(region), dtype='i8')
        stop[idcs] = ref_offset + len(ref_arr) - stop_idcs
        stop = np.where(
            region['start'] != region['stop'],
            np.maximum(region['stop'], stop),
            stop
            )
        region_dset[sel,'stop'] = stop

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
        self._resize_dset(ref_dset, (len(ref_dset) + sum(ns), 2))
        ref_slice = slice(ref_offset,ref_offset + ns[self.rank])
        ref_dset[ref_slice] = refs[:,ref_dir]

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
