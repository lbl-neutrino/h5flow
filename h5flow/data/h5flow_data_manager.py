import h5py
import numpy as np
from mpi4py import MPI

class H5FlowDataManager(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self._fh = None
        self.mpi_flag = True

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def open_file(self, mpi=True):
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
        if self._fh is not None and self._fh:
            self._fh.close()

    @property
    def fh(self):
        if self._fh is None or not self._fh:
            self.open_file()
        return self._fh

    def dset_exists(self, dataset_name):
        return f'{dataset_name}/data' in self.fh

    def ref_exists(self, dataset_name, child_dataset_name):
        return f'{dataset_name}/ref/{child_dataset_name}' in self.fh

    def attr_exists(self, name, key):
        if f'{name}' in self.fh:
            return key in self.fh[f'{name}'].attrs
        return False

    def get_dset(self, dataset_name):
        dset = self.fh[f'{dataset_name}/data']
        dset.id.refresh()
        return dset

    def get_attrs(self, name):
        return self.fh[f'{name}'].attrs

    def get_ref(self, dataset_name, child_dataset_name):
        dset = self.fh[f'{dataset_name}/ref/{child_dataset_name}']
        return dset

    def set_attrs(self, name, **attrs):
        if name not in self.fh:
            self.fh.create_group(name)
        for key,val in attrs.items():
            self.fh[name].attrs[key] = val

    def create_dset(self, dataset_name, dtype):
        path = f'{dataset_name}/data'
        if path not in self.fh:
            if dataset_name not in self.fh:
                self.fh.create_group(dataset_name)
            self.fh.create_dataset(path, (0,), maxshape=(None,),
                dtype=dtype)

    def create_ref(self, dataset_name, child_dataset_name):
        path = f'{dataset_name}/ref/{child_dataset_name}'
        if path not in self.fh:
            if f'{dataset_name}/ref' not in self.fh:
                self.fh.create_group(f'{dataset_name}/ref')
                self.fh[f'{dataset_name}/ref'].attrs['parent'] = self.fh[f'{dataset_name}/data'].ref
            self.fh.create_dataset(path, (0,), maxshape=(None,),
                dtype=h5py.regionref_dtype)
            self.fh[path].attrs['child'] = self.fh[f'{child_dataset_name}/data'].ref

    def reserve_data(self, dataset_name, spec):
        dset = self.get_dset(dataset_name)
        curr_len = len(dset)
        specs = self.comm.allgather(spec)
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            dset.resize((curr_len + sum(specs),))
            rv = slice(curr_len + sum(specs[:self.rank]), curr_len + sum(specs[:self.rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([curr_len] + [spec.stop for spec in specs])
            dset.resize((new_size,))
            rv = spec
        return rv

    def write_data(self, dataset_name, spec, data):
        dset = self.get_dset(dataset_name)
        dset[spec] = data

    def reserve_ref(self, dataset_name, child_dataset_name, spec):
        dset = self.get_ref(dataset_name, child_dataset_name)
        curr_len = len(dset)
        specs = self.comm.allgather(spec)
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            dset.resize((curr_len + sum(specs),))
            rv = slice(curr_len + sum(specs[:self.rank]), curr_len + sum(specs[:self.rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([curr_len] + [spec.stop for spec in specs])
            dset.resize((new_size,))
            rv = spec
        return rv

    def write_ref(self, dataset_name, child_dataset_name, spec, idcs):
        self.close_file()
        gather_idcs = self.comm.gather(idcs, root=0)
        specs = self.comm.gather(spec, root=0)
        if self.rank == 0:
            self.open_file(mpi=False)
            dset = self.get_ref(dataset_name, child_dataset_name)
            child_dset = self.get_dset(child_dataset_name)
            for idcs,spec in zip(gather_idcs,specs):
                dset[spec] = np.array([child_dset.regionref[sel] for sel in idcs], dtype=h5py.regionref_dtype)
            self.close_file()
            self.comm.barrier()
        else:
            self.comm.barrier()
            # self.open_file()
