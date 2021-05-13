import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class H5FlowDataManager(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self._fh = None

    def open_file(self):
        if self._fh is not None:
            self.close_file()
        if rank == 0:
            self._fh = h5py.File(self.filepath, 'a', libver='latest')
        else:
            self._fh = h5py.File(self.filepath, 'r', libver='latest', swmr=True)

    def close_file(self):
        if self._fh is not None and self._fh:
            self._fh.close()

    @property
    def fh(self):
        if self._fh is None or not self._fh:
            self.open_file()
        return self._fh

    def get_dset(self, dataset_name):
        dset = self.fh[f'{dataset_name}/data']
        dset.id.refresh()
        return dset

    def get_attrs(self, stage_name):
        return self.fh[f'{stage_name}']

    def get_ref(self, dataset_name, child_dataset_name):
        dset = self.fh[f'{dataset_name}/ref/{child_dataset_name}']
        return dset

    def set_attrs(self, stage_name, **attrs):
        self.close_file()
        comm.barrier()
        if rank == 0:
            if stage_name not in self.fh:
                self.fh.create_group(stage_name)
            for key,val in attrs.items():
                self.fh[stage_name].attrs[key] = val
            self.fh.swmr_mode = True
            comm.barrier()
        else:
            comm.barrier()
            # self.open_file()

    def create_dset(self, dataset_name, dtype):
        if dataset_name not in self.fh:
            self.close_file()
            comm.barrier()
            if rank == 0:
                self.fh.create_group(dataset_name)
                self.fh.create_dataset(f'{dataset_name}/data', (0,), maxshape=(None,),
                    dtype=dtype)
                self.fh.create_group(f'{dataset_name}/ref')
                self.fh[f'{dataset_name}/ref'].attrs['parent'] = self.fh[f'{dataset_name}/data'].ref
                self.fh.swmr_mode = True
                comm.barrier()
            else:
                comm.barrier()
                # self.open_file()

    def create_ref(self, dataset_name, child_dataset_name):
        path = f'{dataset_name}/ref/{child_dataset_name}'
        if path not in self.fh:
            self.close_file()
            comm.barrier()
            if rank == 0:
                self.fh.create_dataset(path, (0,), maxshape=(None,),
                    dtype=h5py.regionref_dtype)
                self.fh[path].attrs['child'] = self.fh[f'{child_dataset_name}/data'].ref
                self.fh.swmr_mode = True
                comm.barrier()
            else:
                comm.barrier()
                # self.open_file()

    def reserve_data(self, dataset_name, spec):
        dset = self.get_dset(dataset_name)
        curr_len = len(dset)
        specs = comm.allgather(spec)
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            if rank == 0:
                dset.resize((curr_len + sum(specs),))
            rv = slice(curr_len + sum(specs[:rank]), curr_len + sum(specs[:rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([curr_len] + [spec.stop for spec in specs])
            if rank == 0:
                dset.resize((new_size,))
            rv = spec
        if rank == 0:
            dset.flush()
            comm.barrier()
        else:
            comm.barrier()

        return rv

    def write_data(self, dataset_name, spec, data):
        datas = comm.gather(data, root=0)
        specs = comm.gather(spec, root=0)
        dset = self.get_dset(dataset_name)
        if rank == 0:
            for data,spec in zip(datas,specs):
                dset[spec] = data
            dset.flush()
            comm.barrier()
        else:
            comm.barrier()
            dset.id.refresh()

    def reserve_ref(self, dataset_name, child_dataset_name, spec):
        dset = self.get_ref(dataset_name, child_dataset_name)
        curr_len = len(dset)
        specs = comm.allgather(spec)
        if isinstance(spec, int):
            # create a new chunk at the end of the dataset
            if rank == 0:
                dset.resize((curr_len + sum(specs),))
            rv = slice(curr_len + sum(specs[:rank]), curr_len + sum(specs[:rank+1]))
        elif isinstance(spec, slice):
            # maybe create up to a specific chunk of the dataset
            new_size = max([curr_len] + [spec.stop for spec in specs])
            if rank == 0:
                dset.resize((new_size,))
            rv = spec
        if rank == 0:
            dset.flush()
            comm.barrier()
        else:
            comm.barrier()
            dset.id.refresh()
        return rv

    def write_ref(self, dataset_name, child_dataset_name, spec, idcs):
        self.close_file()
        comm.barrier()
        gather_idcs = comm.gather(idcs, root=0)
        specs = comm.gather(spec, root=0)
        if rank == 0:
            dset = self.get_ref(dataset_name, child_dataset_name)
            child_dset = self.get_dset(child_dataset_name)
            for idcs,spec in zip(gather_idcs,specs):
                dset[spec] = np.array([child_dset.regionref[sel] for sel in idcs], dtype=h5py.regionref_dtype)
            self.fh.swmr_mode = True
            comm.barrier()
        else:
            comm.barrier()
            # self.open_file()
