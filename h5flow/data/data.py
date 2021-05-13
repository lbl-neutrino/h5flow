import h5py
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def open_file(filepath, mode='a'):
    fh = h5py.File(filepath, mode, driver='mpio', comm=comm)
    return fh

def get_dset(fh, dataset_name):
    return fh[f'{dataset_name}/data']

def get_attrs(fh, stage_name):
    return fh[f'{stage_name}']

def get_ref(fh, dataset_name, linked_dataset_name):
    return fh[f'{dataset_name}/ref/{linked_dataset_name}']

def set_attrs(fh, stage_name, **attrs):
    if stage_name not in fh:
        fh.create_group(stage_name)
    for key,val in attrs.items():
        fh[stage_name].attrs[key] = val

def create_dset(fh, dataset_name, dtype):
    if dataset_name not in fh:
        fh.create_group(dataset_name)
        fh.create_dataset(f'{dataset_name}/data', (0,), maxshape=(None,),
            dtype=dtype)
        fh.create_group(f'{dataset_name}/ref')
        fh[f'{dataset_name}/ref'].attrs['parent'] = fh[f'{dataset_name}/data'].ref

def create_ref(fh, dataset_name, linked_dataset_name):
    path = f'{dataset_name}/ref/{linked_dataset_name}'
    if path not in fh:
        fh.create_dataset(path, (0,), maxshape=(None,),
            dtype=h5py.regionref_dtype)
        fh[path].attrs['child'] = fh[f'{linked_dataset_name}/data'].ref

def reserve_data(fh, dataset_name, spec):
    dset = get_dset(fh, dataset_name)
    curr_len = len(dset)
    specs = comm.allgather(spec)
    if isinstance(spec, int):
        # create a new chunk at the end of the dataset
        dset.resize((curr_len + sum(specs),))
        return slice(curr_len + sum(specs[:rank]), curr_len + sum(specs[:rank+1]))
    elif isinstance(spec, slice):
        # maybe create up to a specific chunk of the dataset
        new_size = max([curr_len] + [spec.stop for spec in specs])
        dset.resize((new_size,))
        return spec

def write_data(fh, dataset_name, spec, data):
    get_dset(fh, dataset_name)[spec] = data

def reserve_ref(fh, dataset_name, linked_dataset_name, spec):
    dset = get_ref(fh, dataset_name, linked_dataset_name)
    curr_len = len(dset)
    specs = comm.allgather(spec)
    if isinstance(spec, int):
        # create a new chunk at the end of the dataset
        dset.resize((curr_len + sum(specs),))
        return slice(curr_len + sum(specs[:rank]), curr_len + sum(specs[:rank+1]))
    elif isinstance(spec, slice):
        # maybe create up to a specific chunk of the dataset
        new_size = max([curr_len] + [spec.stop for spec in specs])
        dset.resize((new_size,))
        return spec

def generate_ref(fh, dataset_name, idcs):
    dset = get_dset(fh, dataset_name)
    return np.array([dset.regionref[spec] for spec in idcs], dtype=h5py.regionref_dtype)

def write_ref(fh, dataset_name, linked_dataset_name, spec, refs):
    get_ref(fh, dataset_name, linked_dataset_name)[spec] = refs


